package embedder

import (
	"bufio"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Default unix socket used for the shared embedder (Fix B). Each nyawa process
// probes this socket and reuses the running bge_server.py if one exists, so the
// BGE model is loaded only once for the whole stack. Socket path is namespaced
// by model dir so different models don't collide.
func socketPathFor(modelPath string) string {
	h := sha256.Sum256([]byte(modelPath))
	return fmt.Sprintf("/tmp/nyawa-bge-%x.sock", h[:4])
}

type PythonEmbedder struct {
	mu         sync.Mutex
	modelPath  string
	cmd        *exec.Cmd
	conn       net.Conn // unix socket connection (shared mode)
	socketMode bool
	stdin      *json.Encoder
	stdout     *bufio.Scanner
	ready      bool
	dim        int
}

func NewPythonEmbedder(modelPath string) *PythonEmbedder {
	return &PythonEmbedder{modelPath: modelPath, dim: 384}
}

// probeSocket dials an existing unix socket and verifies it answers an "embed"
// probe. On success it wires p.stdin/p.stdout to the socket so Embed() works
// unchanged, and returns true.
func (p *PythonEmbedder) probeSocket(conn net.Conn) bool {
	conn.SetDeadline(time.Now().Add(10 * time.Second))
	enc := json.NewEncoder(conn)
	sc := bufio.NewScanner(conn)
	if err := enc.Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "embed", "params": map[string]string{"text": "probe"}}); err != nil {
		return false
	}
	if !sc.Scan() {
		return false
	}
	var resp struct {
		Result *struct {
			Embedding []float64 `json:"embedding"`
			Dim       int       `json:"dim"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(sc.Bytes(), &resp); err != nil {
		return false
	}
	if resp.Error != nil || resp.Result == nil || resp.Result.Embedding == nil {
		return false
	}
	if resp.Result.Dim > 0 {
		p.dim = resp.Result.Dim
	}
	conn.SetDeadline(time.Time{})
	p.conn = conn
	p.socketMode = true
	p.stdin = json.NewEncoder(conn)
	p.stdout = bufio.NewScanner(conn)
	return true
}

func (p *PythonEmbedder) Start() error {
	scriptPath := findScriptPath()
	if scriptPath == "" {
		return fmt.Errorf("bge_server.py not found")
	}
	pythonPath := findPythonPath()
	if pythonPath == "" {
		return fmt.Errorf("no python with onnxruntime+numpy")
	}

	socketPath := socketPathFor(p.modelPath)

	// 1) Reuse an already-running shared embedder (Fix B).
	if conn, err := net.Dial("unix", socketPath); err == nil {
		if p.probeSocket(conn) {
			p.ready = true
			log.Printf("BGE embedder: reused shared socket %s (dim=%d)", socketPath, p.dim)
			return nil
		}
		conn.Close()
	}

	// 2) Spawn a new shared embedder in socket-serve mode and connect to it.
	// startSocketServer probes the socket itself, wiring p.conn/p.stdin/p.stdout.
	if cmd, _, err := p.startSocketServer(scriptPath, pythonPath, socketPath); err == nil {
		p.cmd = cmd
		p.ready = true
		log.Printf("BGE embedder: started shared socket %s (dim=%d)", socketPath, p.dim)
		return nil
	}

	// 3) Fallback: legacy stdin/stdout subprocess (backward compatible).
	return p.startLegacy(scriptPath, pythonPath)
}

// startSocketServer launches bge_server.py in `serve <socket>` mode, waits for
// READY on stderr, then dials the socket and probes it.
func (p *PythonEmbedder) startSocketServer(scriptPath, pythonPath, socketPath string) (*exec.Cmd, net.Conn, error) {
	cmd := exec.Command(pythonPath, scriptPath, "serve", socketPath)
	cmd.Env = append(os.Environ(), "NYAWA_MODEL_DIR="+p.modelPath)
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("start python: %w", err)
	}
	ready := make(chan struct{}, 1)
	errCh := make(chan error, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "READY") {
				ready <- struct{}{}
			}
			if strings.Contains(line, "Error") || strings.Contains(line, "Traceback") {
				errCh <- fmt.Errorf(line)
			}
		}
	}()
	select {
	case <-ready:
	case err := <-errCh:
		cmd.Process.Kill()
		cmd.Wait()
		return nil, nil, fmt.Errorf("python error: %w", err)
	case <-time.After(30 * time.Second):
		cmd.Process.Kill()
		cmd.Wait()
		return nil, nil, fmt.Errorf("timeout waiting for embedder")
	}

	// Dial the socket, retrying briefly as the server binds.
	deadline := time.Now().Add(5 * time.Second)
	for {
		conn, derr := net.Dial("unix", socketPath)
		if derr == nil {
			if p.probeSocket(conn) {
				return cmd, conn, nil
			}
			conn.Close()
		}
		if time.Now().After(deadline) {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	cmd.Process.Kill()
	cmd.Wait()
	return nil, nil, fmt.Errorf("could not connect to shared embedder socket")
}

// startLegacy is the original behavior: one bge_server.py over stdin/stdout.
func (p *PythonEmbedder) startLegacy(scriptPath, pythonPath string) error {
	p.cmd = exec.Command(pythonPath, scriptPath)
	p.cmd.Env = append(os.Environ(), "NYAWA_MODEL_DIR="+p.modelPath)
	stdin, err := p.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := p.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := p.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}
	if err := p.cmd.Start(); err != nil {
		return fmt.Errorf("start python: %w", err)
	}
	p.stdin = json.NewEncoder(stdin)
	p.stdout = bufio.NewScanner(stdout)
	errCh := make(chan error, 1)
	stderrDone := make(chan struct{}, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "READY") {
				stderrDone <- struct{}{}
			}
			if strings.Contains(line, "Error") || strings.Contains(line, "Traceback") {
				errCh <- fmt.Errorf(line)
			}
		}
	}()
	select {
	case <-stderrDone:
		p.ready = true
		log.Printf("BGE embedder ready (legacy, dim=%d)", p.dim)
		return nil
	case err := <-errCh:
		p.cmd.Process.Kill()
		return fmt.Errorf("python error: %w", err)
	case <-time.After(30 * time.Second):
		p.cmd.Process.Kill()
		return fmt.Errorf("timeout waiting for embedder")
	}
}

func (p *PythonEmbedder) Stop() {
	if p.conn != nil {
		p.conn.Close()
		p.conn = nil
	}
	if p.cmd != nil && p.cmd.Process != nil {
		p.cmd.Process.Kill()
		p.cmd.Wait()
	}
}

func (p *PythonEmbedder) Embed(text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.ready {
		return nil, fmt.Errorf("embedder not ready")
	}
	if p.socketMode && p.conn != nil {
		p.conn.SetWriteDeadline(time.Now().Add(30 * time.Second))
		p.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		defer p.conn.SetDeadline(time.Time{})
	}
	if err := p.stdin.Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "embed", "params": map[string]string{"text": text}}); err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}
	if !p.stdout.Scan() {
		return nil, fmt.Errorf("no response")
	}
	var resp struct {
		Result *struct {
			Embedding []float64 `json:"embedding"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(p.stdout.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("embedder: %s", resp.Error.Message)
	}
	if resp.Result == nil || resp.Result.Embedding == nil {
		return nil, fmt.Errorf("empty result")
	}
	vec := make([]float32, len(resp.Result.Embedding))
	for i, v := range resp.Result.Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

func (p *PythonEmbedder) Name() string    { return "bge-small" }
func (p *PythonEmbedder) Dims() int       { return p.dim }
func (p *PythonEmbedder) Available() bool { return p.ready }

func findScriptPath() string {
	candidates := []string{"internal/embedder/bge_server.py", "/opt/data/nyawa/internal/embedder/bge_server.py"}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func findPythonPath() string {
	candidates := []string{"/opt/hermes/.venv/bin/python3", "/usr/bin/python3", "python3"}
	for _, c := range candidates {
		cmd := exec.Command(c, "-c", "import onnxruntime, numpy; print('ok')")
		if out, err := cmd.Output(); err == nil && strings.TrimSpace(string(out)) == "ok" {
			return c
		}
	}
	return ""
}

package embedder

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type PythonEmbedder struct {
	mu        sync.Mutex
	modelPath string
	cmd       *exec.Cmd
	stdin     *json.Encoder
	stdout    *bufio.Scanner
	ready     bool
	dim       int
}

func NewPythonEmbedder(modelPath string) *PythonEmbedder {
	return &PythonEmbedder{modelPath: modelPath, dim: 384}
}

func (p *PythonEmbedder) Start() error {
	scriptPath := findScriptPath()
	if scriptPath == "" { return fmt.Errorf("bge_server.py not found") }
	pythonPath := findPythonPath()
	if pythonPath == "" { return fmt.Errorf("no python with onnxruntime+numpy") }
	p.cmd = exec.Command(pythonPath, scriptPath)
	p.cmd.Env = append(os.Environ(), "NYAWA_MODEL_DIR="+p.modelPath)
	stdin, err := p.cmd.StdinPipe()
	if err != nil { return fmt.Errorf("stdin pipe: %w", err) }
	stdout, err := p.cmd.StdoutPipe()
	if err != nil { return fmt.Errorf("stdout pipe: %w", err) }
	stderr, err := p.cmd.StderrPipe()
	if err != nil { return fmt.Errorf("stderr pipe: %w", err) }
	if err := p.cmd.Start(); err != nil { return fmt.Errorf("start python: %w", err) }
	p.stdin = json.NewEncoder(stdin)
	p.stdout = bufio.NewScanner(stdout)
	errCh := make(chan error, 1)
	stderrDone := make(chan struct{}, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "READY") { stderrDone <- struct{}{} }
			if strings.Contains(line, "Error") || strings.Contains(line, "Traceback") { errCh <- fmt.Errorf(line) }
		}
	}()
	select {
	case <-stderrDone:
		p.ready = true
		log.Printf("BGE embedder ready (dim=%d)", p.dim)
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
	if p.cmd != nil && p.cmd.Process != nil { p.cmd.Process.Kill(); p.cmd.Wait() }
}

func (p *PythonEmbedder) Embed(text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.ready { return nil, fmt.Errorf("embedder not ready") }
	if err := p.stdin.Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "embed", "params": map[string]string{"text": text}}); err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}
	if !p.stdout.Scan() { return nil, fmt.Errorf("no response") }
	var resp struct {
		Result *struct { Embedding []float64 `json:"embedding"` } `json:"result"`
		Error  *struct { Message string `json:"message"` } `json:"error"`
	}
	if err := json.Unmarshal(p.stdout.Bytes(), &resp); err != nil { return nil, fmt.Errorf("parse: %w", err) }
	if resp.Error != nil { return nil, fmt.Errorf("embedder: %s", resp.Error.Message) }
	if resp.Result == nil || resp.Result.Embedding == nil { return nil, fmt.Errorf("empty result") }
	vec := make([]float32, len(resp.Result.Embedding))
	for i, v := range resp.Result.Embedding { vec[i] = float32(v) }
	return vec, nil
}

func (p *PythonEmbedder) Name() string    { return "bge-small" }
func (p *PythonEmbedder) Dims() int       { return p.dim }
func (p *PythonEmbedder) Available() bool { return p.ready }

func findScriptPath() string {
	candidates := []string{"internal/embedder/bge_server.py", "/opt/data/nyawa/internal/embedder/bge_server.py"}
	for _, c := range candidates { if _, err := os.Stat(c); err == nil { return c } }
	return ""
}

func findPythonPath() string {
	candidates := []string{"/opt/hermes/.venv/bin/python3", "/usr/bin/python3", "python3"}
	for _, c := range candidates {
		cmd := exec.Command(c, "-c", "import onnxruntime, numpy; print('ok')")
		if out, err := cmd.Output(); err == nil && strings.TrimSpace(string(out)) == "ok" { return c }
	}
	return ""
}

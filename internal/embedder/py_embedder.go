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
	p.cmd = exec.Command(findPythonPath(), findScriptPath())
	p.cmd.Env = append(os.Environ(), "NYAWA_MODEL_DIR="+p.modelPath)
	stdin, _ := p.cmd.StdinPipe()
	stdout, _ := p.cmd.StdoutPipe()
	stderr, _ := p.cmd.StderrPipe()
	if err := p.cmd.Start(); err != nil {
		return fmt.Errorf("start: %w", err)
	}
	p.stdin = json.NewEncoder(stdin)
	p.stdout = bufio.NewScanner(stdout)
	errCh := make(chan error, 1)
	doneCh := make(chan struct{}, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "READY") { doneCh <- struct{}{} }
			if strings.Contains(line, "Traceback") || strings.Contains(line, "Error") { errCh <- fmt.Errorf(line) }
		}
	}()
	select {
	case <-doneCh: p.ready = true; log.Printf("BGE embedder ready (dim=%d)", p.dim); return nil
	case err := <-errCh: p.cmd.Process.Kill(); return fmt.Errorf("python: %w", err)
	case <-time.After(30 * time.Second): p.cmd.Process.Kill(); return fmt.Errorf("timeout")
	}
}

func (p *PythonEmbedder) Stop() {
	if p.cmd != nil && p.cmd.Process != nil { p.cmd.Process.Kill(); p.cmd.Wait() }
}

func (p *PythonEmbedder) Embed(text string) ([]float32, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.ready { return nil, fmt.Errorf("not ready") }
	p.stdin.Encode(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "embed", "params": map[string]string{"text": text}})
	if !p.stdout.Scan() { return nil, fmt.Errorf("no response") }
	var resp struct {
		Result *struct{ Embedding []float64 `json:"embedding"` } `json:"result"`
		Error  *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(p.stdout.Bytes(), &resp); err != nil { return nil, fmt.Errorf("parse: %w", err) }
	if resp.Error != nil { return nil, fmt.Errorf("embedder: %s", resp.Error.Message) }
	if resp.Result == nil || resp.Result.Embedding == nil { return nil, fmt.Errorf("empty") }
	vec := make([]float32, len(resp.Result.Embedding))
	for i, v := range resp.Result.Embedding { vec[i] = float32(v) }
	return vec, nil
}

func (p *PythonEmbedder) Name() string    { return "bge-small" }
func (p *PythonEmbedder) Dims() int       { return p.dim }
func (p *PythonEmbedder) Available() bool { return p.ready }

func findScriptPath() string {
	for _, c := range []string{"internal/embedder/bge_server.py", "/opt/data/nyawa/internal/embedder/bge_server.py"} {
		if _, err := os.Stat(c); err == nil { return c }
	}
	return ""
}
func findPythonPath() string {
	for _, c := range []string{"/opt/hermes/.venv/bin/python3", "/usr/bin/python3", "python3"} {
		cmd := exec.Command(c, "-c", "import onnxruntime, numpy; print('ok')")
		if out, err := cmd.Output(); err == nil && strings.TrimSpace(string(out)) == "ok" { return c }
	}
	return ""
}

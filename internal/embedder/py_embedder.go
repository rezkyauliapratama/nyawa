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

// Start owns exactly one BGE server subprocess and connects it through private
// stdin/stdout pipes. The runtime image supplies both the interpreter and the
// model, so startup has no host venv, socket, or second-process discovery path.
func (p *PythonEmbedder) Start() error {
	scriptPath := findScriptPath()
	if scriptPath == "" {
		return fmt.Errorf("bge_server.py not found")
	}
	pythonPath := findPythonPath()
	if pythonPath == "" {
		return fmt.Errorf("no python with onnxruntime+numpy")
	}
	return p.startLegacy(scriptPath, pythonPath)
}

// startLegacy is the sole process-start path: bge_server.py speaks
// line-delimited JSON-RPC over stdin/stdout and reports readiness on stderr.
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
				errCh <- fmt.Errorf("%s", line)
			}
		}
	}()
	select {
	case <-stderrDone:
		p.ready = true
		log.Printf("BGE embedder ready (embedded stdin/stdout, dim=%d)", p.dim)
		return nil
	case err := <-errCh:
		p.cmd.Process.Kill()
		p.cmd.Wait()
		return fmt.Errorf("python error: %w", err)
	case <-time.After(30 * time.Second):
		p.cmd.Process.Kill()
		p.cmd.Wait()
		return fmt.Errorf("timeout waiting for embedder")
	}
}

func (p *PythonEmbedder) Stop() {
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

// Default model locations, in priority order. embedModelDirDefault lives on the
// persistent volume so it survives container recreation; the legacy entry is the
// original in-repo path and remains as a fallback for other deployments.
const (
	embedModelDirDefault = "/opt/data/bge-models/minilm-multilingual"
	embedModelDirLegacy  = "/opt/data/nyawa/internal/embedder/model"
)

// embedModelDirs is a var so tests can substitute their own candidates.
var embedModelDirs = []string{embedModelDirDefault, embedModelDirLegacy}

// DefaultModelDir resolves which BGE model directory to use: the
// NYAWA_EMBED_MODEL_DIR override wins, then the first candidate that exists on
// disk. It returns the chosen path plus a short label for logging.
func DefaultModelDir() (string, string) {
	return resolveModelDir(os.Getenv, os.Stat)
}

func resolveModelDir(getenv func(string) string, stat func(string) (os.FileInfo, error)) (path, source string) {
	if d := strings.TrimSpace(getenv("NYAWA_EMBED_MODEL_DIR")); d != "" {
		return d, "env:NYAWA_EMBED_MODEL_DIR"
	}
	for _, c := range embedModelDirs {
		if _, err := stat(c); err == nil {
			return c, "default"
		}
	}
	return embedModelDirs[0], "default(missing)"
}

func findScriptPath() string {
	candidates := []string{"internal/embedder/bge_server.py", "/opt/data/nyawa/internal/embedder/bge_server.py"}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// embedPythonCandidates are the interpreters shipped with or explicitly
// installed in the runtime image. A host or persistent-volume venv is never
// consulted by the durable runtime.
var embedPythonCandidates = []string{
	"/usr/local/bin/python3",
	"/usr/bin/python3",
	"python3",
}

func findPythonPath() string {
	for _, c := range embedPythonCandidates {
		cmd := exec.Command(c, "-c", "import onnxruntime, numpy; print('ok')")
		if out, err := cmd.Output(); err == nil && strings.TrimSpace(string(out)) == "ok" {
			return c
		}
	}
	return ""
}

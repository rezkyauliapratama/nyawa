package embedder

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

var (
	ErrNoEmbedder                 = errors.New("no embedder available")
	ErrEmbeddingDimensionMismatch = errors.New("embedding dimension mismatch")
)

// LocalModelFingerprint identifies the local model artifacts that define the
// embedding space. Both files are required and must be regular, non-empty files;
// missing or ambiguous artifacts fail closed instead of falling back to a
// directory name.
func LocalModelFingerprint(modelDir string) (string, error) {
	if modelDir == "" {
		return "", fmt.Errorf("local model directory is required")
	}

	hash := sha256.New()
	for _, name := range []string{"model.onnx", "tokenizer.json"} {
		path := filepath.Join(modelDir, name)
		info, err := os.Lstat(path)
		if err != nil {
			return "", fmt.Errorf("inspect local model artifact %s: %w", name, err)
		}
		if !info.Mode().IsRegular() || info.Size() <= 0 {
			return "", fmt.Errorf("local model artifact %s must be a non-empty regular file", name)
		}

		file, err := os.Open(path)
		if err != nil {
			return "", fmt.Errorf("open local model artifact %s: %w", name, err)
		}
		if _, err := io.WriteString(hash, name+string([]byte{0})); err != nil {
			file.Close()
			return "", fmt.Errorf("hash local model artifact %s: %w", name, err)
		}
		if _, err := io.Copy(hash, file); err != nil {
			file.Close()
			return "", fmt.Errorf("hash local model artifact %s: %w", name, err)
		}
		if err := file.Close(); err != nil {
			return "", fmt.Errorf("close local model artifact %s: %w", name, err)
		}
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil)), nil
}

type Embedder interface {
	Embed(text string) ([]float32, error)
	Name() string
	Dims() int
	Available() bool
}

type PriorityChain struct {
	chain   []Embedder
	current Embedder
}

func NewPriorityChain(embedders ...Embedder) *PriorityChain {
	pc := &PriorityChain{chain: embedders}
	pc.selectActive()
	return pc
}

func (pc *PriorityChain) Embed(text string) ([]float32, error) {
	if pc.current != nil && pc.current.Available() {
		if v, err := pc.current.Embed(text); err == nil {
			return v, nil
		}
	}
	return pc.tryFallback(text)
}

// EmbedQuery and EmbedPassage preserve task-aware APIs through a priority
// chain. Embedders without retrieval task support (for example local BGE) use
// their ordinary Embed method.
func (pc *PriorityChain) EmbedQuery(text string) ([]float32, error) {
	return pc.embedTask(text, true)
}

func (pc *PriorityChain) EmbedPassage(text string) ([]float32, error) {
	return pc.embedTask(text, false)
}

// EmbedPassages uses a provider batch API when the active embedder supports it;
// otherwise it preserves the existing single-input task-aware behavior.
func (pc *PriorityChain) EmbedPassages(texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if pc.current == nil || !pc.current.Available() {
		pc.selectActive()
	}
	if pc.current == nil {
		return nil, ErrNoEmbedder
	}
	if batch, ok := pc.current.(interface {
		EmbedPassages([]string) ([][]float32, error)
	}); ok {
		return batch.EmbedPassages(texts)
	}
	vectors := make([][]float32, len(texts))
	for i, text := range texts {
		vector, err := pc.EmbedPassage(text)
		if err != nil {
			return nil, fmt.Errorf("embed passage %d: %w", i, err)
		}
		vectors[i] = vector
	}
	return vectors, nil
}

func (pc *PriorityChain) embedTask(text string, query bool) ([]float32, error) {
	if pc.current == nil || !pc.current.Available() {
		return pc.Embed(text)
	}
	if taskEmbedder, ok := pc.current.(interface {
		EmbedQuery(string) ([]float32, error)
		EmbedPassage(string) ([]float32, error)
	}); ok {
		if query {
			return taskEmbedder.EmbedQuery(text)
		}
		return taskEmbedder.EmbedPassage(text)
	}
	return pc.Embed(text)
}

func (pc *PriorityChain) tryFallback(text string) ([]float32, error) {
	var lastErr error
	for _, e := range pc.chain {
		if !e.Available() {
			continue
		}
		v, err := e.Embed(text)
		if err == nil {
			pc.current = e
			return v, nil
		}
		lastErr = fmt.Errorf("%s: %w", e.Name(), err)
	}
	if lastErr != nil {
		return nil, lastErr
	}
	return nil, ErrNoEmbedder
}

func (pc *PriorityChain) selectActive() {
	for _, e := range pc.chain {
		if e.Available() {
			pc.current = e
			return
		}
	}
	pc.current = nil
}

func (pc *PriorityChain) HealthCheck() (name string, ok bool) {
	if pc.current != nil && pc.current.Available() {
		return pc.current.Name(), true
	}
	pc.selectActive()
	if pc.current != nil {
		return pc.current.Name(), true
	}
	return "", false
}

func (pc *PriorityChain) Current() string {
	if pc.current != nil {
		return pc.current.Name()
	}
	return ""
}

func (pc *PriorityChain) Available() bool { _, ok := pc.HealthCheck(); return ok }
func (pc *PriorityChain) Dims() int {
	if pc.current != nil {
		return pc.current.Dims()
	}
	// Keep the first configured embedder's dimension when none is healthy yet.
	// The default chain starts with BGE/384, so opening an existing BGE index
	// remains safe instead of silently assuming the Ollama fallback's 768 dims.
	for _, e := range pc.chain {
		if dims := e.Dims(); dims > 0 {
			return dims
		}
	}
	return 768
}
func (pc *PriorityChain) StopAll() {
	for _, e := range pc.chain {
		if p, ok := e.(*PythonEmbedder); ok {
			p.Stop()
		}
	}
}

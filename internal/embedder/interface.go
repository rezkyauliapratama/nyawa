package embedder

import (
	"errors"
	"fmt"
)

var (
	ErrNoEmbedder                 = errors.New("no embedder available")
	ErrEmbeddingDimensionMismatch = errors.New("embedding dimension mismatch")
)

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

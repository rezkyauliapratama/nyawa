package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type OllamaEmbedder struct {
	baseURL string
	model   string
	client  *http.Client
	dims    int
}

type OllamaConfig struct {
	BaseURL string `yaml:"base_url"`
	Model   string `yaml:"model"`
	Timeout int    `yaml:"timeout_seconds"`
}

func NewOllamaEmbedder(cfg OllamaConfig) *OllamaEmbedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "nomic-embed-text"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30
	}
	return &OllamaEmbedder{
		baseURL: cfg.BaseURL, model: cfg.Model, dims: 768,
		client: &http.Client{Timeout: time.Duration(cfg.Timeout) * time.Second},
	}
}

type ollamaEmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}
type ollamaEmbedResponse struct {
	Embedding []float64 `json:"embedding"`
	Error     string    `json:"error,omitempty"`
}

func (o *OllamaEmbedder) Embed(text string) ([]float32, error) {
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(ollamaEmbedRequest{Model: o.model, Prompt: text})
	resp, err := o.client.Post(o.baseURL+"/api/embeddings", "application/json", &buf)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama status %d", resp.StatusCode)
	}
	var result ollamaEmbedResponse
	json.Unmarshal(body, &result)
	if result.Error != "" {
		return nil, fmt.Errorf("ollama: %s", result.Error)
	}
	vec := make([]float32, len(result.Embedding))
	for i, v := range result.Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

func (o *OllamaEmbedder) Name() string { return "ollama" }
func (o *OllamaEmbedder) Dims() int    { return o.dims }
func (o *OllamaEmbedder) Available() bool {
	resp, err := o.client.Get(o.baseURL + "/api/tags")
	return err == nil && resp.StatusCode == http.StatusOK
}

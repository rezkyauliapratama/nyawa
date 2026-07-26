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
	if cfg.BaseURL == "" { cfg.BaseURL = "http://localhost:11434" }
	if cfg.Model == "" { cfg.Model = "nomic-embed-text" }
	if cfg.Timeout <= 0 { cfg.Timeout = 30 }
	return &OllamaEmbedder{baseURL: cfg.BaseURL, model: cfg.Model, dims: 768,
		client: &http.Client{Timeout: time.Duration(cfg.Timeout) * time.Second}}
}

func (o *OllamaEmbedder) Embed(text string) ([]float32, error) {
	body := map[string]string{"model": o.model, "prompt": text}
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil { return nil, fmt.Errorf("ollama encode: %w", err) }
	resp, err := o.client.Post(o.baseURL+"/api/embeddings", "application/json", &buf)
	if err != nil { return nil, fmt.Errorf("ollama request: %w", err) }
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil { return nil, fmt.Errorf("ollama read: %w", err) }
	if resp.StatusCode != http.StatusOK { return nil, fmt.Errorf("ollama status %d: %s", resp.StatusCode, string(respBody)) }
	var result struct {
		Embedding []float64 `json:"embedding"`
		Error     string    `json:"error,omitempty"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil { return nil, fmt.Errorf("ollama decode: %w", err) }
	if result.Error != "" { return nil, fmt.Errorf("ollama error: %s", result.Error) }
	vec := make([]float32, len(result.Embedding))
	for i, v := range result.Embedding { vec[i] = float32(v) }
	return vec, nil
}

func (o *OllamaEmbedder) Name() string { return "ollama" }
func (o *OllamaEmbedder) Dims() int { return o.dims }
func (o *OllamaEmbedder) Available() bool {
	resp, err := o.client.Get(o.baseURL + "/api/tags")
	if err != nil { return false }
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// JinaEmbedder uses Jina AI's embedding API (OpenAI-compatible).
// Supports multilingual embeddings including Indonesian.
type JinaEmbedder struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
	dims    int
}

// NewJinaEmbedder creates a Jina AI embedder.
// Config via env vars:
//
//	JINA_API_KEY   — API key (required for Available())
//	JINA_BASE_URL  — API base URL (default: https://api.jina.ai/v1)
//	JINA_MODEL     — Model name (default: jina-embeddings-v3)
//	JINA_DIMS      — Embedding dimension (default: 1024, can use Matryoshka 768/512/256/128)
func NewJinaEmbedder() *JinaEmbedder {
	apiKey := os.Getenv("JINA_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("RERANK_API_KEY") // fallback to shared key
	}
	baseURL := os.Getenv("JINA_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.jina.ai/v1"
	}
	model := os.Getenv("JINA_MODEL")
	if model == "" {
		model = "jina-embeddings-v3"
	}
	dims := 1024
	return &JinaEmbedder{
		apiKey:  apiKey,
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{Timeout: 30 * time.Second},
		dims:    dims,
	}
}

type jinaEmbedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
	Normalized bool     `json:"normalized"`
}

type jinaEmbedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model  string `json:"model"`
	Object string `json:"object"`
	Usage  struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

func (j *JinaEmbedder) Embed(text string) ([]float32, error) {
	if j.apiKey == "" {
		return nil, fmt.Errorf("JINA_API_KEY not set")
	}

	body := jinaEmbedRequest{
		Model:      j.model,
		Input:      []string{text},
		Dimensions: j.dims,
		Normalized: true,
	}
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(body)

	req, err := http.NewRequest("POST", j.baseURL+"/embeddings", &buf)
	if err != nil {
		return nil, fmt.Errorf("jina: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+j.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := j.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("jina: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("jina status %d: %s", resp.StatusCode, string(respBody))
	}

	var result jinaEmbedResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("jina parse: %w", err)
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("jina: empty response")
	}

	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

func (j *JinaEmbedder) Name() string { return "jina/" + j.model }
func (j *JinaEmbedder) Dims() int    { return j.dims }

func (j *JinaEmbedder) Available() bool {
	if j.apiKey == "" {
		return false
	}
	req, err := http.NewRequest("GET", j.baseURL+"/embeddings", nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", "Bearer "+j.apiKey)
	resp, err := j.client.Do(req)
	return err == nil && resp.StatusCode < 500
}

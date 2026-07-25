package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// OpenAIEmbedder uses any OpenAI-compatible embedding API.
// Vendor-agnostic: works with Jina AI, Voyage, OpenAI, Cohere, etc.
//
// Config via generic env vars (not vendor-specific):
//
//	EMBEDDING_API_KEY   — API key (required)
//	EMBEDDING_BASE_URL  — API base URL (default: https://api.jina.ai/v1)
//	EMBEDDING_MODEL     — Model name (default: jina-embeddings-v3)
//	EMBEDDING_DIMS      — Output dimension (default: 1024, Matryoshka if supported)
type OpenAIEmbedder struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
	dims    int
}

func NewOpenAIEmbedder() *OpenAIEmbedder {
	apiKey := os.Getenv("EMBEDDING_API_KEY")
	baseURL := os.Getenv("EMBEDDING_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.jina.ai/v1"
	}
	model := os.Getenv("EMBEDDING_MODEL")
	if model == "" {
		model = "jina-embeddings-v3"
	}
	dims := 1024
	if ds := os.Getenv("EMBEDDING_DIMS"); ds != "" {
		if d, err := strconv.Atoi(ds); err == nil && d > 0 {
			dims = d
		}
	}
	return &OpenAIEmbedder{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		client:  &http.Client{Timeout: 30 * time.Second},
		dims:    dims,
	}
}

type openAIEmbedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions int      `json:"dimensions,omitempty"`
	Normalized bool     `json:"normalized"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
}

func (o *OpenAIEmbedder) Embed(text string) ([]float32, error) {
	if o.apiKey == "" {
		return nil, fmt.Errorf("EMBEDDING_API_KEY not set")
	}
	body := openAIEmbedRequest{
		Model:      o.model,
		Input:      []string{text},
		Dimensions: o.dims,
		Normalized: true,
	}
	var buf bytes.Buffer
	json.NewEncoder(&buf).Encode(body)
	req, err := http.NewRequest("POST", o.baseURL+"/embeddings", &buf)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed status %d: %s", resp.StatusCode, string(respBody))
	}
	var result openAIEmbedResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("embed parse: %w", err)
	}
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("embed: empty response")
	}
	vec := make([]float32, len(result.Data[0].Embedding))
	for i, v := range result.Data[0].Embedding {
		vec[i] = float32(v)
	}
	return vec, nil
}

func (o *OpenAIEmbedder) Name() string     { return "openai/" + o.model }
func (o *OpenAIEmbedder) Dims() int         { return o.dims }
func (o *OpenAIEmbedder) Available() bool   { return o.apiKey != "" }

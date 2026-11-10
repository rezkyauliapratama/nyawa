package embedder

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultEmbeddingBaseURL = "https://api.jina.ai/v1"
	defaultEmbeddingModel   = "jina-embeddings-v3"
	defaultEmbeddingDims    = 1024
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
	apiKey                string
	baseURL               string
	model                 string
	client                *http.Client
	dims                  int
	retrievalTasksEnabled bool
}

// NewOpenAIEmbedder preserves the original permissive constructor for callers
// that only need an optionally available embedder. API mode should use
// NewOpenAIEmbedderFromEnv so invalid configuration is reported instead of
// silently falling back to defaults.
func NewOpenAIEmbedder() *OpenAIEmbedder {
	apiKey := strings.TrimSpace(os.Getenv("EMBEDDING_API_KEY"))
	baseURL := strings.TrimSpace(os.Getenv("EMBEDDING_BASE_URL"))
	if baseURL == "" {
		baseURL = defaultEmbeddingBaseURL
	}
	model := strings.TrimSpace(os.Getenv("EMBEDDING_MODEL"))
	if model == "" {
		model = defaultEmbeddingModel
	}
	dims := defaultEmbeddingDims
	if ds := strings.TrimSpace(os.Getenv("EMBEDDING_DIMS")); ds != "" {
		if d, err := strconv.Atoi(ds); err == nil && d > 0 {
			dims = d
		}
	}
	return newOpenAIEmbedder(apiKey, baseURL, model, dims)
}

// NewOpenAIEmbedderFromEnv builds an API embedder from getenv and validates
// every value that can change the vector shape. It is used when
// MEMORY_EMBEDDER=api is explicit, where silently accepting a typo could lead
// to an incompatible HNSW index.
func NewOpenAIEmbedderFromEnv(getenv func(string) string) (*OpenAIEmbedder, error) {
	return NewOpenAIEmbedderFromEnvWithHTTPClient(getenv, nil)
}

// NewOpenAIEmbedderFromEnvWithHTTPClient builds a validated API embedder and
// optionally accepts an HTTP client. A nil client uses the standard timeout
// configured by the embedder.
func NewOpenAIEmbedderFromEnvWithHTTPClient(getenv func(string) string, client *http.Client) (*OpenAIEmbedder, error) {
	apiKey := strings.TrimSpace(getenv("EMBEDDING_API_KEY"))
	if apiKey == "" {
		return nil, fmt.Errorf("EMBEDDING_API_KEY is required")
	}

	baseURL := strings.TrimSpace(getenv("EMBEDDING_BASE_URL"))
	if baseURL == "" {
		baseURL = defaultEmbeddingBaseURL
	}
	parsedURL, err := url.Parse(baseURL)
	if err != nil || parsedURL.Host == "" || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
		return nil, fmt.Errorf("EMBEDDING_BASE_URL must be an absolute http(s) URL")
	}

	model := strings.TrimSpace(getenv("EMBEDDING_MODEL"))
	if model == "" {
		model = defaultEmbeddingModel
	}

	dims := defaultEmbeddingDims
	if rawDims := strings.TrimSpace(getenv("EMBEDDING_DIMS")); rawDims != "" {
		dims, err = strconv.Atoi(rawDims)
		if err != nil || dims <= 0 {
			return nil, fmt.Errorf("EMBEDDING_DIMS must be a positive integer")
		}
	}

	emb := newOpenAIEmbedder(apiKey, baseURL, model, dims)
	if client != nil {
		emb.client = client
	}
	return emb, nil
}

func newOpenAIEmbedder(apiKey, baseURL, model string, dims int) *OpenAIEmbedder {
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
	Task       string   `json:"task,omitempty"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
}

const (
	TaskRetrievalQuery   = "retrieval.query"
	TaskRetrievalPassage = "retrieval.passage"
)

// Embed retains the task-less behavior used by non-RAG callers. RAG callers
// select the query or passage task explicitly through the methods below.
func (o *OpenAIEmbedder) Embed(text string) ([]float32, error) {
	return o.embed(text, "")
}

// EnableRetrievalTasks allows the RAG API configuration to opt this embedder
// into provider-specific query/passage task fields. It is disabled by default.
func (o *OpenAIEmbedder) EnableRetrievalTasks(enabled bool) {
	o.retrievalTasksEnabled = enabled
}

func (o *OpenAIEmbedder) EmbedQuery(text string) ([]float32, error) {
	if !o.retrievalTasksEnabled {
		return o.embed(text, "")
	}
	return o.embed(text, TaskRetrievalQuery)
}

func (o *OpenAIEmbedder) EmbedPassage(text string) ([]float32, error) {
	if !o.retrievalTasksEnabled {
		return o.embed(text, "")
	}
	return o.embed(text, TaskRetrievalPassage)
}

func (o *OpenAIEmbedder) embed(text, task string) ([]float32, error) {
	vectors, err := o.embedBatch([]string{text}, task)
	if err != nil {
		return nil, err
	}
	return vectors[0], nil
}

// EmbedPassages batches RAG chunks into one task-aware provider request. Memory
// callers continue to use Embed; this method is only used by explicit RAG
// backfill and never changes the configured vector profile.
func (o *OpenAIEmbedder) EmbedPassages(texts []string) ([][]float32, error) {
	task := ""
	if o.retrievalTasksEnabled {
		task = TaskRetrievalPassage
	}
	return o.embedBatch(texts, task)
}

func (o *OpenAIEmbedder) embedBatch(texts []string, task string) ([][]float32, error) {
	if o.apiKey == "" {
		return nil, fmt.Errorf("EMBEDDING_API_KEY not set")
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	body := openAIEmbedRequest{
		Model:      o.model,
		Input:      texts,
		Dimensions: o.dims,
		Normalized: true,
		Task:       task,
	}
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(body); err != nil {
		return nil, fmt.Errorf("embed encode: %w", err)
	}
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
		return nil, fmt.Errorf("embed status %d", resp.StatusCode)
	}
	var result openAIEmbedResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("embed parse: %w", err)
	}
	if len(result.Data) != len(texts) {
		return nil, fmt.Errorf("embed: provider returned %d vectors for %d inputs", len(result.Data), len(texts))
	}
	vectors := make([][]float32, len(texts))
	seen := make([]bool, len(texts))
	for _, item := range result.Data {
		if item.Index < 0 || item.Index >= len(texts) || seen[item.Index] {
			return nil, fmt.Errorf("embed: provider returned invalid or duplicate vector index %d", item.Index)
		}
		if got := len(item.Embedding); got != o.dims {
			return nil, fmt.Errorf("%w: %s returned %d dimensions, expected %d", ErrEmbeddingDimensionMismatch, o.Name(), got, o.dims)
		}
		vector := make([]float32, len(item.Embedding))
		for i, value := range item.Embedding {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("embed: provider returned non-finite vector value at index %d", item.Index)
			}
			vector[i] = float32(value)
		}
		vectors[item.Index] = vector
		seen[item.Index] = true
	}
	for i, ok := range seen {
		if !ok {
			return nil, fmt.Errorf("embed: provider omitted vector index %d", i)
		}
	}
	return vectors, nil
}

func (o *OpenAIEmbedder) Name() string    { return "openai/" + o.model }
func (o *OpenAIEmbedder) Dims() int       { return o.dims }
func (o *OpenAIEmbedder) Available() bool { return o.apiKey != "" }

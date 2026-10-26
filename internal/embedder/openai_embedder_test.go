package embedder

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNewOpenAIEmbedderFromEnvValidatesConfig(t *testing.T) {
	valid := map[string]string{
		"EMBEDDING_API_KEY":  "test-key",
		"EMBEDDING_BASE_URL": "https://api.example.test/v1",
		"EMBEDDING_MODEL":    "test-model",
		"EMBEDDING_DIMS":     "1024",
	}
	getenv := func(key string) string { return valid[key] }

	emb, err := NewOpenAIEmbedderFromEnv(getenv)
	if err != nil {
		t.Fatalf("NewOpenAIEmbedderFromEnv() error = %v", err)
	}
	if got, want := emb.Name(), "openai/test-model"; got != want {
		t.Fatalf("Name() = %q, want %q", got, want)
	}
	if got, want := emb.Dims(), 1024; got != want {
		t.Fatalf("Dims() = %d, want %d", got, want)
	}

	invalidConfigs := []struct {
		name   string
		mutate func(map[string]string)
	}{
		{name: "missing API key", mutate: func(values map[string]string) { values["EMBEDDING_API_KEY"] = "" }},
		{name: "invalid dimensions", mutate: func(values map[string]string) { values["EMBEDDING_DIMS"] = "not-a-number" }},
		{name: "invalid base URL", mutate: func(values map[string]string) { values["EMBEDDING_BASE_URL"] = "not-a-url" }},
	}
	for _, tt := range invalidConfigs {
		t.Run(tt.name, func(t *testing.T) {
			values := make(map[string]string, len(valid))
			for key, value := range valid {
				values[key] = value
			}
			tt.mutate(values)
			if _, err := NewOpenAIEmbedderFromEnv(func(key string) string { return values[key] }); err == nil {
				t.Fatal("NewOpenAIEmbedderFromEnv() succeeded with invalid configuration")
			}
		})
	}
}

func TestOpenAIEmbedderRejectsUnexpectedResponseDimension(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.URL.Path, "/embeddings"; got != want {
			t.Errorf("path = %q, want %q", got, want)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("test-key", server.URL, "test-model", 4)
	emb.client = server.Client()
	if _, err := emb.Embed("test"); !errors.Is(err, ErrEmbeddingDimensionMismatch) {
		t.Fatalf("Embed() error = %v, want ErrEmbeddingDimensionMismatch", err)
	}
}

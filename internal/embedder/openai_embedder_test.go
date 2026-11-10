package embedder

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
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

func TestOpenAIEmbedderUsesFakeHTTPProvider(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/embeddings" {
			t.Errorf("request = %s %s, want POST /embeddings", r.Method, r.URL.Path)
		}
		if got, want := r.Header.Get("Authorization"), "Bearer fake-test-key"; got != want {
			t.Errorf("authorization header = %q, want fake test credential", got)
		}
		var request openAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request.Model != "fake-model" || len(request.Input) != 1 || request.Input[0] != "synthetic text" || request.Dimensions != 4 {
			t.Errorf("unexpected fake request: %+v", request)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("fake-test-key", server.URL, "fake-model", 4)
	emb.client = server.Client()
	vector, err := emb.Embed("synthetic text")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if len(vector) != 4 || vector[0] != 0.1 || vector[3] != 0.4 {
		t.Fatalf("Embed() vector = %v, want four fake dimensions", vector)
	}
}

func TestOpenAIEmbedderUsesRetrievalQueryAndPassageTasks(t *testing.T) {
	seen := make(chan string, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request openAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request.Model != "jina-embeddings-v3" || request.Dimensions != 4 || len(request.Input) != 1 {
			t.Errorf("unexpected task request: %+v", request)
		}
		seen <- request.Task
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("fake-test-key", server.URL, "jina-embeddings-v3", 4)
	emb.client = server.Client()
	emb.EnableRetrievalTasks(true)
	for _, call := range []func(string) ([]float32, error){emb.EmbedPassage, emb.EmbedQuery} {
		vector, err := call("offline synthetic test")
		if err != nil || len(vector) != 4 {
			t.Fatalf("task embedding = %v, %v; want four-dimension vector", vector, err)
		}
	}
	got := map[string]int{}
	for i := 0; i < 2; i++ {
		got[<-seen]++
	}
	if got[TaskRetrievalPassage] != 1 || got[TaskRetrievalQuery] != 1 {
		t.Fatalf("provider tasks = %+v, want retrieval.passage and retrieval.query", got)
	}
}

func TestOpenAIEmbedderBatchesPassagesAndRestoresResponseOrder(t *testing.T) {
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var request openAIEmbedRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode batch request: %v", err)
		}
		if len(request.Input) != 2 || request.Input[0] != "synthetic passage A" || request.Input[1] != "synthetic passage B" {
			t.Errorf("batch input = %q", request.Input)
		}
		if request.Task != TaskRetrievalPassage || request.Model != "jina-embeddings-v3" || request.Dimensions != 2 || !request.Normalized {
			t.Errorf("batch profile = %+v", request)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0.3,0.4]},{"index":0,"embedding":[0.1,0.2]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("fake-test-key", server.URL, "jina-embeddings-v3", 2)
	emb.client = server.Client()
	emb.EnableRetrievalTasks(true)
	vectors, err := emb.EmbedPassages([]string{"synthetic passage A", "synthetic passage B"})
	if err != nil {
		t.Fatalf("EmbedPassages() error = %v", err)
	}
	if requests != 1 || len(vectors) != 2 || vectors[0][0] != 0.1 || vectors[1][0] != 0.3 {
		t.Fatalf("batch result requests=%d vectors=%v", requests, vectors)
	}
}

func TestOpenAIEmbedderRejectsDuplicateBatchResponseIndex(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2]},{"index":0,"embedding":[0.3,0.4]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("fake-test-key", server.URL, "jina-embeddings-v3", 2)
	emb.client = server.Client()
	if _, err := emb.EmbedPassages([]string{"synthetic A", "synthetic B"}); err == nil {
		t.Fatal("EmbedPassages() accepted duplicate response indices")
	}
}

func TestOpenAIEmbedderTasklessMethodsOmitTask(t *testing.T) {
	requests := make(chan map[string]json.RawMessage, 3)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		requests <- request
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}`))
	}))
	defer server.Close()

	emb := newOpenAIEmbedder("fake-test-key", server.URL, "generic-model", 4)
	emb.client = server.Client()
	for _, call := range []func(string) ([]float32, error){emb.Embed, emb.EmbedQuery, emb.EmbedPassage} {
		if _, err := call("offline synthetic test"); err != nil {
			t.Fatalf("taskless embedding call: %v", err)
		}
	}
	for i := 0; i < 3; i++ {
		if request := <-requests; request["task"] != nil {
			t.Errorf("taskless request contains task field: %s", request["task"])
		}
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

func TestLocalModelFingerprintDistinguishesSameBasenameArtifacts(t *testing.T) {
	root := t.TempDir()
	modelA := filepath.Join(root, "first", "shared-model")
	modelB := filepath.Join(root, "second", "shared-model")
	for dir, contents := range map[string]map[string]string{
		modelA: {"model.onnx": "synthetic model A", "tokenizer.json": `{"vocab":"A"}`},
		modelB: {"model.onnx": "synthetic model B", "tokenizer.json": `{"vocab":"B"}`},
	} {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		for name, content := range contents {
			if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	if filepath.Base(modelA) != filepath.Base(modelB) {
		t.Fatal("test fixture model basenames differ")
	}
	fingerprintA, err := LocalModelFingerprint(modelA)
	if err != nil {
		t.Fatalf("fingerprint model A: %v", err)
	}
	fingerprintB, err := LocalModelFingerprint(modelB)
	if err != nil {
		t.Fatalf("fingerprint model B: %v", err)
	}
	if fingerprintA == fingerprintB {
		t.Fatalf("different artifact contents share fingerprint %q", fingerprintA)
	}
}

func TestLocalModelFingerprintFailsClosedOnMissingArtifact(t *testing.T) {
	modelDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(modelDir, "model.onnx"), []byte("synthetic model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LocalModelFingerprint(modelDir); err == nil {
		t.Fatal("LocalModelFingerprint() succeeded without tokenizer.json")
	}
}

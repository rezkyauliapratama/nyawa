package main

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/index"
	"github.com/rezkyauliapratama/nyawa/internal/store"
)

func TestMemoryEmbedderMode(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    string
		wantErr bool
	}{
		{name: "unset defaults to local", raw: "", want: memoryEmbedderLocal},
		{name: "local", raw: "local", want: memoryEmbedderLocal},
		{name: "legacy bge alias", raw: "bge", want: memoryEmbedderLocal},
		{name: "api", raw: "api", want: memoryEmbedderAPI},
		{name: "case and whitespace", raw: " API ", want: memoryEmbedderAPI},
		{name: "reject typo", raw: "remote", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := memoryEmbedderMode(tt.raw)
			if (err != nil) != tt.wantErr {
				t.Fatalf("memoryEmbedderMode(%q) error = %v, wantErr=%v", tt.raw, err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("memoryEmbedderMode(%q) = %q, want %q", tt.raw, got, tt.want)
			}
		})
	}
}

func TestRAGEmbedderMode(t *testing.T) {
	for _, test := range []struct {
		name    string
		raw     string
		want    string
		wantErr bool
	}{
		{name: "unset defaults to local", raw: "", want: ragEmbedderLocal},
		{name: "local", raw: "local", want: ragEmbedderLocal},
		{name: "legacy bge alias", raw: "bge", want: ragEmbedderLocal},
		{name: "api", raw: "api", want: ragEmbedderAPI},
		{name: "case and whitespace", raw: " API ", want: ragEmbedderAPI},
		{name: "reject typo", raw: "remote", wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := ragEmbedderMode(test.raw)
			if (err != nil) != test.wantErr {
				t.Fatalf("ragEmbedderMode(%q) error = %v, wantErr=%v", test.raw, err, test.wantErr)
			}
			if got != test.want {
				t.Fatalf("ragEmbedderMode(%q) = %q, want %q", test.raw, got, test.want)
			}
		})
	}
}

func TestNewRAGAPIEmbedderBuildsProfileWithoutProviderCall(t *testing.T) {
	env := map[string]string{
		"EMBEDDING_API_KEY":  "fake-test-key",
		"EMBEDDING_BASE_URL": "https://api.example.test/v1",
		"EMBEDDING_MODEL":    "test-embedding-model",
		"EMBEDDING_DIMS":     "1024",
	}
	getenv := func(key string) string { return env[key] }
	emb, profile, err := newRAGAPIEmbedder(getenv)
	if err != nil {
		t.Fatalf("newRAGAPIEmbedder() error = %v", err)
	}
	if got, want := emb.Current(), "openai/test-embedding-model"; got != want {
		t.Fatalf("RAG current embedder = %q, want %q", got, want)
	}
	if profile.Provider != "openai-compatible" || profile.Model != "test-embedding-model" || profile.Dimensions != 1024 {
		t.Fatalf("unexpected RAG profile: %+v", profile)
	}
	if profile.BaseURL != "https://api.example.test/v1" {
		t.Fatalf("RAG profile base URL = %q", profile.BaseURL)
	}
	if profile.QueryTask != "" || profile.PassageTask != "" {
		t.Fatalf("generic provider profile unexpectedly enables tasks: %+v", profile)
	}
}

type rewriteEmbeddingTransport struct {
	target string
	base   http.RoundTripper
}

func (rt rewriteEmbeddingTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	rewritten := request.Clone(request.Context())
	target, err := url.Parse(rt.target)
	if err != nil {
		return nil, err
	}
	rewritten.URL.Scheme = target.Scheme
	rewritten.URL.Host = target.Host
	rewritten.Host = target.Host
	return rt.base.RoundTrip(rewritten)
}

func TestRAGAPITaskModesDriveFakeRequestsAndProfile(t *testing.T) {
	requests := make(chan map[string]json.RawMessage, 8)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || !strings.HasSuffix(r.URL.Path, "/embeddings") {
			t.Errorf("request = %s %s, want POST */embeddings", r.Method, r.URL.Path)
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":[{"embedding":[0.1,0.2,0.3,0.4]}]}`))
	}))
	defer server.Close()

	cases := []struct {
		name         string
		env          map[string]string
		wantQuery    bool
		wantPassage  bool
		wantProfile  bool
	}{
		{
			name: "default Jina v3 auto",
			env: map[string]string{
				"EMBEDDING_API_KEY": "fake-test-key", "EMBEDDING_DIMS": "4",
			},
			wantQuery: true, wantPassage: true, wantProfile: true,
		},
		{
			name: "generic provider auto is taskless",
			env: map[string]string{
				"EMBEDDING_API_KEY": "fake-test-key", "EMBEDDING_BASE_URL": server.URL + "/v1",
				"EMBEDDING_MODEL": "generic-model", "EMBEDDING_DIMS": "4",
			},
		},
		{
			name: "explicit compatible proxy opt-in",
			env: map[string]string{
				"EMBEDDING_API_KEY": "fake-test-key", "EMBEDDING_BASE_URL": server.URL + "/v1",
				"EMBEDDING_MODEL": "proxy-jina-v3", "EMBEDDING_DIMS": "4",
				"RAG_EMBEDDING_TASK_MODE": "jina-v3",
			},
			wantQuery: true, wantPassage: true, wantProfile: true,
		},
		{
			name: "none disables known Jina tasks",
			env: map[string]string{
				"EMBEDDING_API_KEY": "fake-test-key", "EMBEDDING_DIMS": "4",
				"RAG_EMBEDDING_TASK_MODE": "none",
			},
		},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			env := test.env
			if env["EMBEDDING_BASE_URL"] == "" {
				env["EMBEDDING_BASE_URL"] = "https://api.jina.ai/v1"
			}
			if env["EMBEDDING_MODEL"] == "" {
				env["EMBEDDING_MODEL"] = "jina-embeddings-v3"
			}
			getenv := func(key string) string { return env[key] }
			client := &http.Client{Transport: rewriteEmbeddingTransport{target: server.URL, base: server.Client().Transport}}
			emb, profile, err := newRAGAPIEmbedderWithHTTPClient(getenv, client)
			if err != nil {
				t.Fatalf("newRAGAPIEmbedderWithHTTPClient() error = %v", err)
			}
			if (profile.QueryTask != "") != test.wantProfile || (profile.PassageTask != "") != test.wantProfile {
				t.Fatalf("profile tasks = %q/%q, want enabled=%t", profile.QueryTask, profile.PassageTask, test.wantProfile)
			}
			for _, call := range []func(string) ([]float32, error){emb.EmbedQuery, emb.EmbedPassage} {
				if _, err := call("offline synthetic RAG input"); err != nil {
					t.Fatalf("fake embedding request: %v", err)
				}
			}
			gotTasks := []string{"", ""}
			for i := range gotTasks {
				body := <-requests
				if raw := body["task"]; raw != nil {
					if err := json.Unmarshal(raw, &gotTasks[i]); err != nil {
						t.Fatalf("decode task: %v", err)
					}
				}
			}
			want := []string{"", ""}
			if test.wantQuery {
				want = []string{"retrieval.query", "retrieval.passage"}
			}
			if gotTasks[0] != want[0] || gotTasks[1] != want[1] {
				t.Fatalf("request tasks = %q, want %q", gotTasks, want)
			}
		})
	}
}

func TestRAGAPIRejectsInvalidTaskMode(t *testing.T) {
	env := map[string]string{
		"EMBEDDING_API_KEY": "fake-test-key", "EMBEDDING_BASE_URL": "https://api.example.test/v1",
		"EMBEDDING_MODEL": "generic-model", "EMBEDDING_DIMS": "4", "RAG_EMBEDDING_TASK_MODE": "maybe",
	}
	if _, _, err := newRAGAPIEmbedder(func(key string) string { return env[key] }); err == nil {
		t.Fatal("newRAGAPIEmbedder() accepted invalid task mode")
	}
}

func TestNewRAGAPIEmbedderRejectsMissingKey(t *testing.T) {
	getenv := func(key string) string {
		if key == "EMBEDDING_BASE_URL" {
			return "https://api.example.test/v1"
		}
		return ""
	}
	if _, _, err := newRAGAPIEmbedder(getenv); err == nil {
		t.Fatal("newRAGAPIEmbedder() succeeded without EMBEDDING_API_KEY")
	}
}

func TestGetEmbedderUsesExplicitAPI(t *testing.T) {
	t.Setenv("MEMORY_EMBEDDER", "api")
	t.Setenv("EMBEDDING_API_KEY", "test-key")
	t.Setenv("EMBEDDING_BASE_URL", "https://example.invalid/v1")
	t.Setenv("EMBEDDING_MODEL", "test-embedding-model")
	t.Setenv("EMBEDDING_DIMS", "1024")

	emb, err := getEmbedder()
	if err != nil {
		t.Fatalf("getEmbedder() error = %v", err)
	}
	if got, want := emb.Current(), "openai/test-embedding-model"; got != want {
		t.Fatalf("current embedder = %q, want %q", got, want)
	}
	if got, want := emb.Dims(), 1024; got != want {
		t.Fatalf("embedder dimensions = %d, want %d", got, want)
	}
}

func TestGetEmbedderRejectsAPIWithoutKey(t *testing.T) {
	t.Setenv("MEMORY_EMBEDDER", "api")
	t.Setenv("EMBEDDING_API_KEY", "")

	if _, err := getEmbedder(); err == nil {
		t.Fatal("getEmbedder() succeeded without EMBEDDING_API_KEY")
	}
}

func TestExplicitAPIEmbedderRejectsExistingBGEIndex(t *testing.T) {
	t.Setenv("MEMORY_EMBEDDER", "api")
	t.Setenv("EMBEDDING_API_KEY", "test-key")
	t.Setenv("EMBEDDING_BASE_URL", "https://api.example.test/v1")
	t.Setenv("EMBEDDING_MODEL", "test-embedding-model")
	t.Setenv("EMBEDDING_DIMS", "1024")

	emb, err := getEmbedder()
	if err != nil {
		t.Fatalf("getEmbedder() error = %v", err)
	}

	dbPath := filepath.Join(t.TempDir(), "memory.db")
	h := index.NewHNSW(index.DefaultHNSWConfig(384))
	if err := h.Insert("bge-memory", make([]float32, 384)); err != nil {
		t.Fatalf("seed BGE index: %v", err)
	}
	if err := h.Save(dbPath + ".hnsw"); err != nil {
		t.Fatalf("save BGE index: %v", err)
	}

	st, err := store.NewStore(dbPath, emb)
	if st != nil {
		st.Close()
		t.Fatal("Store opened an existing BGE index with the API embedder")
	}
	if !errors.Is(err, index.ErrDimensionMismatch) {
		t.Fatalf("NewStore() error = %v, want ErrDimensionMismatch", err)
	}
}

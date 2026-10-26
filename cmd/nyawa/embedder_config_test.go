package main

import (
	"errors"
	"path/filepath"
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

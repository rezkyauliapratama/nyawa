package rag

import (
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/index"
)

func openRAGManifestTestDB(t *testing.T, dir string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(dir, "memory.db"))
	if err != nil {
		t.Fatalf("open metadata DB: %v", err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func TestOpenVectorIndexCreatesSeparateProfiledIndex(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	path := filepath.Join(dir, "memory.db.rag.hnsw")
	profile := EmbeddingProfile{
		Provider:    "openai-compatible",
		Model:       "jina-embeddings-v3",
		BaseURL:     "https://api.jina.ai/v1",
		Dimensions:  1024,
		QueryTask:   "retrieval.query",
		PassageTask: "retrieval.passage",
	}

	h, err := OpenVectorIndex(db, path, profile)
	if err != nil {
		t.Fatalf("OpenVectorIndex() error = %v", err)
	}
	if got := h.Dimension(); got != profile.Dimensions {
		t.Fatalf("index dimension = %d, want %d", got, profile.Dimensions)
	}
	var manifest ragIndexManifest
	if err := db.QueryRow(`SELECT version,index_path,provider,model,base_url,dimensions,query_task,passage_task FROM rag_index_manifest WHERE singleton=1`).Scan(
		&manifest.Version, &manifest.IndexPath, &manifest.Profile.Provider, &manifest.Profile.Model, &manifest.Profile.BaseURL, &manifest.Profile.Dimensions, &manifest.Profile.QueryTask, &manifest.Profile.PassageTask); err != nil {
		t.Fatalf("read persisted RAG manifest: %v", err)
	}
	if manifest.IndexPath != path || manifest.Profile != profile {
		t.Fatalf("persisted manifest = %+v, want profile %+v at %s", manifest, profile, path)
	}

	if err := h.Insert("rag_chk_test_0", make([]float32, profile.Dimensions)); err != nil {
		t.Fatalf("insert vector: %v", err)
	}
	if err := h.Save(path); err != nil {
		t.Fatalf("save vector: %v", err)
	}
	reopened, err := OpenVectorIndex(db, path, profile)
	if err != nil {
		t.Fatalf("reopen RAG index: %v", err)
	}
	if got := reopened.Size(); got != 1 {
		t.Fatalf("reopened vector count = %d, want 1", got)
	}
}

func TestOpenVectorIndexRejectsProfileMismatch(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	path := filepath.Join(dir, "memory.db.rag.hnsw")
	profile := EmbeddingProfile{Provider: "bge-onnx", Model: "minilm-multilingual", Dimensions: 384}
	if _, err := OpenVectorIndex(db, path, profile); err != nil {
		t.Fatalf("create BGE index: %v", err)
	}

	wrong := EmbeddingProfile{Provider: "openai-compatible", Model: "jina-embeddings-v3", BaseURL: "https://api.jina.ai/v1", Dimensions: 1024}
	if _, err := OpenVectorIndex(db, path, wrong); !errors.Is(err, ErrRAGIndexProfileMismatch) {
		t.Fatalf("OpenVectorIndex() error = %v, want ErrRAGIndexProfileMismatch", err)
	}
}

func TestOpenVectorIndexRejectsDifferentLocalArtifactsWithSameBasename(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	indexPath := filepath.Join(dir, "local.rag.hnsw")
	modelA := filepath.Join(dir, "one", "shared-model")
	modelB := filepath.Join(dir, "two", "shared-model")
	for modelDir, contents := range map[string]map[string]string{
		modelA: {"model.onnx": "synthetic model A", "tokenizer.json": `{"vocab":"A"}`},
		modelB: {"model.onnx": "synthetic model B", "tokenizer.json": `{"vocab":"B"}`},
	} {
		if err := os.MkdirAll(modelDir, 0o700); err != nil {
			t.Fatal(err)
		}
		for name, content := range contents {
			if err := os.WriteFile(filepath.Join(modelDir, name), []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	if filepath.Base(modelA) != filepath.Base(modelB) {
		t.Fatal("test fixture model basenames differ")
	}
	fingerprintA, err := embedder.LocalModelFingerprint(modelA)
	if err != nil {
		t.Fatalf("fingerprint model A: %v", err)
	}
	fingerprintB, err := embedder.LocalModelFingerprint(modelB)
	if err != nil {
		t.Fatalf("fingerprint model B: %v", err)
	}
	profileA := EmbeddingProfile{Provider: "bge-onnx", Model: "minilm-multilingual@" + fingerprintA, Dimensions: 384}
	profileB := EmbeddingProfile{Provider: "bge-onnx", Model: "minilm-multilingual@" + fingerprintB, Dimensions: 384}
	if _, err := OpenVectorIndex(db, indexPath, profileA); err != nil {
		t.Fatalf("create model A index: %v", err)
	}
	if _, err := OpenVectorIndex(db, indexPath, profileB); !errors.Is(err, ErrRAGIndexProfileMismatch) {
		t.Fatalf("OpenVectorIndex() error = %v, want local artifact profile mismatch", err)
	}
}

func TestOpenVectorIndexRejectsTaskProfileMismatch(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	path := filepath.Join(dir, "task-profile.rag.hnsw")
	profile := EmbeddingProfile{Provider: "openai-compatible", Model: "jina-embeddings-v3", BaseURL: "https://api.jina.ai/v1", Dimensions: 1024, QueryTask: "retrieval.query", PassageTask: "retrieval.passage"}
	if _, err := OpenVectorIndex(db, path, profile); err != nil {
		t.Fatalf("create task-profile index: %v", err)
	}
	wrong := profile
	wrong.QueryTask = "text-matching"
	if _, err := OpenVectorIndex(db, path, wrong); !errors.Is(err, ErrRAGIndexProfileMismatch) {
		t.Fatalf("OpenVectorIndex() error = %v, want task-profile mismatch", err)
	}
}

func TestOpenVectorIndexProfileMismatchLeavesLegacyHNSWBytesUntouched(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	path := filepath.Join(dir, "legacy-profile.rag.hnsw")
	profile := EmbeddingProfile{Provider: "provider-a", Model: "model-a", Dimensions: 1}
	if _, err := OpenVectorIndex(db, path, profile); err != nil {
		t.Fatalf("initialize manifest: %v", err)
	}

	legacy := []byte(`{"ep":"legacy","ml":0,"cfg":{"M":2,"Mmax":2,"EfConstruction":16,"EfSearch":8,"ML":0.5,"Dim":1},"ns":{"legacy":{"ID":"legacy","Vec":[0],"Level":0}},"g":[{"legacy":{}}],"rag_backbone_order":["legacy"]}`)
	if err := os.WriteFile(path, legacy, 0o600); err != nil {
		t.Fatalf("write legacy HNSW: %v", err)
	}
	wrongProfile := EmbeddingProfile{Provider: "provider-b", Model: "model-b", Dimensions: 1}
	if _, err := OpenVectorIndex(db, path, wrongProfile); !errors.Is(err, ErrRAGIndexProfileMismatch) {
		t.Fatalf("OpenVectorIndex() error = %v, want ErrRAGIndexProfileMismatch", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read HNSW after mismatch: %v", err)
	}
	if string(got) != string(legacy) {
		t.Fatal("profile mismatch changed legacy HNSW file bytes")
	}
}

func TestOpenVectorIndexRejectsNonEmptyUnmanifestedIndex(t *testing.T) {
	dir := t.TempDir()
	db := openRAGManifestTestDB(t, dir)
	path := filepath.Join(dir, "legacy.rag.hnsw")
	h := index.NewHNSW(index.DefaultHNSWConfig(384))
	if err := h.Insert("rag_chk_legacy_0", make([]float32, 384)); err != nil {
		t.Fatalf("insert legacy vector: %v", err)
	}
	if err := h.Save(path); err != nil {
		t.Fatalf("save legacy index: %v", err)
	}

	profile := EmbeddingProfile{Provider: "bge-onnx", Model: "minilm-multilingual", Dimensions: 384}
	if _, err := OpenVectorIndex(db, path, profile); !errors.Is(err, ErrRAGIndexManifestMissing) {
		t.Fatalf("OpenVectorIndex() error = %v, want ErrRAGIndexManifestMissing", err)
	}
}

func TestEmbeddingProfileDoesNotPersistURLCredentials(t *testing.T) {
	profile := EmbeddingProfile{
		Provider:   "openai-compatible",
		Model:      "jina-embeddings-v3",
		BaseURL:    "https://user:private@example.test/v1?api_key=placeholder#fragment",
		Dimensions: 1024,
	}
	got, err := normalizeEmbeddingProfile(profile)
	if err != nil {
		t.Fatalf("normalizeEmbeddingProfile() error = %v", err)
	}
	if got.BaseURL != "https://example.test/v1" {
		t.Fatalf("normalized base URL = %q, want %q", got.BaseURL, "https://example.test/v1")
	}
}

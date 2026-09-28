package server

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	_ "github.com/mattn/go-sqlite3"

	"github.com/rezkyauliapratama/nyawa/internal/index"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
)

// newTestServerForRAG builds a Server backed by an in-memory SQLite RAG store
// so handleRAGIngest can be exercised end-to-end without the heavy
// store/pipeline/embedder wiring used in production. The embedder is nil:
// IngestFile skips vector insertion when no embedder is available, which is
// all the handler path needs.
func newTestServerForRAG(t *testing.T) *Server {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatalf("open sqlite: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })
	rs := rag.NewRAGStore(db, index.NewHNSW(index.DefaultHNSWConfig(4)),
		filepath.Join(t.TempDir(), "test.hnsw"), nil)
	return &Server{ragStore: rs}
}

// writeTempDoc writes a small text file to ingest and returns its path.
func writeTempDoc(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "doc.txt")
	if err := os.WriteFile(p, []byte("hello rag ingest test content"), 0o644); err != nil {
		t.Fatalf("write temp doc: %v", err)
	}
	return p
}

func doIngest(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.handleRAGIngest(rec, req)
	return rec
}

// TestHandleRAGIngest_AcceptsCanonicalFilePath is the regression test for the
// bug: the documented/contract field name is `file_path` (it is what the
// dashboard and the handler's own error message use), so sending it must
// succeed rather than being silently ignored and rejected with 400.
func TestHandleRAGIngest_AcceptsCanonicalFilePath(t *testing.T) {
	s := newTestServerForRAG(t)
	fp := writeTempDoc(t)
	body, _ := json.Marshal(map[string]string{"file_path": fp, "collection": "documents"})

	rec := doIngest(t, s, string(body))
	if rec.Code == http.StatusBadRequest {
		t.Fatalf("file_path was rejected with 400 (bug); body=%s", rec.Body.String())
	}
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body=%s", rec.Code, rec.Body.String())
	}
	var doc struct {
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &doc); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if doc.Filename != "doc.txt" {
		t.Errorf("filename = %q, want %q", doc.Filename, "doc.txt")
	}
}

// TestHandleRAGIngest_AcceptsLegacyFilepathAlias guards backward compatibility
// for the `filepath` spelling that worked before this fix.
func TestHandleRAGIngest_AcceptsLegacyFilepathAlias(t *testing.T) {
	s := newTestServerForRAG(t)
	fp := writeTempDoc(t)
	body, _ := json.Marshal(map[string]string{"filepath": fp, "collection": "documents"})

	rec := doIngest(t, s, string(body))
	if rec.Code == http.StatusBadRequest {
		t.Fatalf("legacy filepath was rejected with 400; body=%s", rec.Body.String())
	}
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body=%s", rec.Code, rec.Body.String())
	}
}

// TestHandleRAGIngest_RequiresFilePath ensures that omitting both spellings
// still yields 400 with a message that names the canonical `file_path` field.
func TestHandleRAGIngest_RequiresFilePath(t *testing.T) {
	s := newTestServerForRAG(t)
	rec := doIngest(t, s, `{"collection":"documents"}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "file_path") {
		t.Errorf("error message should name file_path, got: %s", rec.Body.String())
	}
}

// TestHandleRAGIngest_RequiresCollection ensures collection stays mandatory.
func TestHandleRAGIngest_RequiresCollection(t *testing.T) {
	s := newTestServerForRAG(t)
	fp := writeTempDoc(t)
	body, _ := json.Marshal(map[string]string{"file_path": fp})
	rec := doIngest(t, s, string(body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
}

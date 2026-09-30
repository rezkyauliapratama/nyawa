package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/store"
)

// TestHTTPRequestCanonicalKeysBind is the regression guard for the HTTP side
// of the snake_case binding bug: every key documented in the README /
// dashboard must populate the request struct.
func TestHTTPRequestCanonicalKeysBind(t *testing.T) {
	tests := []struct {
		name  string
		body  string
		newR  func() any
		check func(t *testing.T, r any)
	}{
		{
			name: "ingest file_path",
			body: `{"file_path":"/tmp/doc.md","collection":"docs"}`,
			newR: func() any { return &ragIngestRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*ragIngestRequest)
				r.resolve()
				if r.FilePath != "/tmp/doc.md" || r.Collection != "docs" {
					t.Fatalf("file_path/collection not bound: %+v", r)
				}
			},
		},
		{
			name: "ingest legacy filepath",
			body: `{"filepath":"/tmp/legacy.md","collection":"docs"}`,
			newR: func() any { return &ragIngestRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*ragIngestRequest)
				r.resolve()
				if r.FilePath != "/tmp/legacy.md" {
					t.Fatalf("legacy filepath alias not resolved: %+v", r)
				}
			},
		},
		{
			name: "query top_k",
			body: `{"query":"q","collection":"docs","top_k":7}`,
			newR: func() any { return &ragQueryRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*ragQueryRequest)
				r.resolve()
				if r.TopK != 7 {
					t.Fatalf("top_k not bound: %+v", r)
				}
			},
		},
		{
			name: "query legacy topK",
			body: `{"query":"q","topK":9}`,
			newR: func() any { return &ragQueryRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*ragQueryRequest)
				r.resolve()
				if r.TopK != 9 {
					t.Fatalf("legacy topK alias not resolved: %+v", r)
				}
			},
		},
		{
			name: "collection chunk_size",
			body: `{"name":"docs","chunk_size":250}`,
			newR: func() any { return &createCollectionRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*createCollectionRequest)
				r.resolve()
				if r.ChunkSize != 250 {
					t.Fatalf("chunk_size not bound: %+v", r)
				}
			},
		},
		{
			name: "collection legacy chunkSize",
			body: `{"name":"docs","chunkSize":333}`,
			newR: func() any { return &createCollectionRequest{} },
			check: func(t *testing.T, v any) {
				r := v.(*createCollectionRequest)
				r.resolve()
				if r.ChunkSize != 333 {
					t.Fatalf("legacy chunkSize alias not resolved: %+v", r)
				}
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := tc.newR()
			if err := json.Unmarshal([]byte(tc.body), r); err != nil {
				t.Fatalf("unmarshal %s: %v", tc.body, err)
			}
			tc.check(t, r)
		})
	}
}

// TestHandleRAGIngestHTTPCanonicalFilePath proves the dashboard flow
// ({"file_path", "collection"}) returns 201 instead of 400.
func TestHandleRAGIngestHTTPCanonicalFilePath(t *testing.T) {
	srv, path := ragIngestTestServer(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest",
		strings.NewReader(`{"file_path":`+quote(path)+`,"collection":"scratch_http"}`))
	srv.handleRAGIngest(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("POST /v1/rag/ingest {file_path} = %d, want 201; body: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"chunk_count"`) {
		t.Fatalf("ingest response missing chunk_count: %s", rec.Body.String())
	}
}

// TestHandleRAGIngestHTTPLegacyFilePathAlias proves the pre-v1.2.0
// single-word spelling still returns 201.
func TestHandleRAGIngestHTTPLegacyFilePathAlias(t *testing.T) {
	srv, path := ragIngestTestServer(t)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/rag/ingest",
		strings.NewReader(`{"filepath":`+quote(path)+`,"collection":"scratch_http"}`))
	srv.handleRAGIngest(rec, req)
	if rec.Code != http.StatusCreated {
		t.Fatalf("POST /v1/rag/ingest {filepath} = %d, want 201; body: %s", rec.Code, rec.Body.String())
	}
}

func ragIngestTestServer(t *testing.T) (*Server, string) {
	t.Helper()
	st, err := store.NewStore(filepath.Join(t.TempDir(), "http-params.db"), nil)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	rs := rag.NewRAGStore(st.GetDB(), st.GetHNSW(), st.GetHNSWPath(), nil)

	path := filepath.Join(t.TempDir(), "doc.md")
	if err := os.WriteFile(path, []byte("# Doc\n\nHTTP parameter binding fixture.\n"), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	return &Server{ragStore: rs}, path
}

func quote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

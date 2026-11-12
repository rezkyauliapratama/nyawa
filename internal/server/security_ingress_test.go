package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/security"
	"github.com/rezkyauliapratama/nyawa/internal/store"
)

func newSecurityIngressStore(t *testing.T) *store.Store {
	t.Helper()
	st, err := store.NewStore(t.TempDir()+"/security-ingress.db", nil)
	if err != nil {
		t.Fatalf("open temporary SQLite store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

// This is deliberately synthetic and assembled from pieces so no live-looking
// credential is embedded in a test failure or repository diff.
func syntheticClassifiedContent() string {
	return "synthetic ingress fixture " + "s" + "k-" + strings.Repeat("A", 32)
}

func storedContentCount(t *testing.T, st *store.Store, content string) int {
	t.Helper()
	var count int
	if err := st.GetDB().QueryRow(`SELECT COUNT(*) FROM memories WHERE content = ?`, content).Scan(&count); err != nil {
		t.Fatalf("count stored test fixture: %v", err)
	}
	return count
}

func postJSON(t *testing.T, handler http.Handler, path string, value any) (int, []byte) {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec.Code, rec.Body.Bytes()
}

func TestHTTPStoreIngressSecurityParity(t *testing.T) {
	st := newSecurityIngressStore(t)
	var pipe *search.Pipeline
	var emb *embedder.PriorityChain
	var rs *rag.RAGStore
	api := New(st, pipe, emb, rs, Config{})
	handler := api.withMiddleware(api.mux)

	classifier := security.NewFilter()
	classified := syntheticClassifiedContent()
	if class, _ := classifier.Classify(classified); class != security.Secret {
		t.Fatal("synthetic fixture no longer triggers the secret classifier")
	}

	singleStatus, singleBody := postJSON(t, handler, "/v1/memories", map[string]string{"content": classified})
	singleStored := storedContentCount(t, st, classified)
	if singleStatus != http.StatusForbidden || singleStored != 0 {
		t.Errorf("single-store ingress: status=%d, matching SQLite rows=%d; want forbidden and no persistence", singleStatus, singleStored)
	}
	if bytes.Contains(singleBody, []byte(classified)) {
		t.Error("single-store ingress echoed classified content")
	}

	batchStatus, batchBody := postJSON(t, handler, "/v1/memories/batch", map[string]any{"memories": []map[string]string{{"content": classified}}})
	batchStored := storedContentCount(t, st, classified)
	batchRejected := batchStatus >= http.StatusBadRequest
	if !batchRejected {
		var result struct {
			Results []struct {
				Status string `json:"status"`
				Error  string `json:"error"`
			} `json:"results"`
		}
		if err := json.Unmarshal(batchBody, &result); err != nil {
			t.Errorf("decode batch-store response (status=%d): %v", batchStatus, err)
		} else {
			for _, row := range result.Results {
				if row.Status == "error" || row.Error != "" {
					batchRejected = true
					break
				}
			}
		}
	}
	if !batchRejected || batchStored != 0 {
		t.Errorf("batch-store ingress: status=%d, rejected=%t, matching SQLite rows=%d; want HTTP or per-row rejection and no persistence", batchStatus, batchRejected, batchStored)
	}
	if bytes.Contains(batchBody, []byte(classified)) {
		t.Error("batch-store ingress echoed classified content")
	}

	// Benign controls ensure the test is guarding classification, not disabling
	// either storage route. Both accepted values must reach SQLite.
	benignSingle := "ordinary HTTP single-store regression control"
	benignSingleStatus, _ := postJSON(t, handler, "/v1/memories", map[string]string{"content": benignSingle})
	if benignSingleStatus != http.StatusCreated || storedContentCount(t, st, benignSingle) != 1 {
		t.Errorf("benign single-store control: status=%d, rows=%d; want 201 and one row", benignSingleStatus, storedContentCount(t, st, benignSingle))
	}

	benignBatch := "ordinary HTTP batch-store regression control"
	benignBatchStatus, _ := postJSON(t, handler, "/v1/memories/batch", map[string]any{"memories": []map[string]string{{"content": benignBatch}}})
	if benignBatchStatus != http.StatusCreated || storedContentCount(t, st, benignBatch) != 1 {
		t.Errorf("benign batch-store control: status=%d, rows=%d; want 201 and one row", benignBatchStatus, storedContentCount(t, st, benignBatch))
	}
}

func TestSecurityClassifierHashFalsePositiveIsExplicit(t *testing.T) {
	// A SHA-256-shaped identifier is benign in this fixture, but the current
	// broad 40-character key pattern classifies it as secret. This records a
	// classifier false positive; it is not an ingress contract or a whitelist.
	sha256Like := strings.Repeat("a", 64)
	class, reason := security.NewFilter().Classify("artifact digest " + sha256Like)
	if class != security.Secret || reason != "AWS Secret Key" {
		t.Fatalf("expected the documented broad-pattern false positive, got class=%q reason=%q", class, reason)
	}
}

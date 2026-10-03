package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// newRecallHTTPServer builds a real SQLite store with three memories that all
// match the same FTS5 query, backed by a pipeline without an embedder (FTS5
// only), so the recall filters can be exercised end to end over HTTP.
func newRecallHTTPServer(t *testing.T) *Server {
	t.Helper()
	st, err := store.NewStore(filepath.Join(t.TempDir(), "http-recall-filters.db"), nil)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })

	mems := []*types.Memory{
		{ID: "memA", Content: "alpha project planning decision record", Type: types.TypeDecision, Namespace: "hermes"},
		{ID: "memB", Content: "alpha project planning note entry", Type: types.TypeNote, Namespace: "hermes"},
		{ID: "memC", Content: "alpha project planning conversation transcript", Type: types.MemoryType("conversation"), Namespace: "hermes"},
	}
	for _, m := range mems {
		if err := st.InsertMemory(m); err != nil {
			t.Fatalf("insert %s: %v", m.ID, err)
		}
	}

	p := search.NewPipeline(st, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})
	return &Server{store: st, pipeline: p}
}

func postRecall(t *testing.T, srv *Server, body string) (int, []string) {
	t.Helper()
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/recall", strings.NewReader(body))
	srv.handleRecall(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("POST /v1/recall %s = %d, body: %s", body, rec.Code, rec.Body.String())
	}
	var payload struct {
		Results []struct {
			ID   string `json:"id"`
			Type string `json:"type"`
		} `json:"results"`
		Count int `json:"count"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response %q: %v", rec.Body.String(), err)
	}
	if payload.Count != len(payload.Results) {
		t.Fatalf("count %d does not match results %d", payload.Count, len(payload.Results))
	}
	gotTypes := make([]string, len(payload.Results))
	for i, r := range payload.Results {
		gotTypes[i] = r.Type
	}
	return payload.Count, gotTypes
}

// TestRecallRequestBindsCanonicalKeys guards the HTTP side of the binding bug:
// min_score and exclude_types must reach the request struct.
func TestRecallRequestBindsCanonicalKeys(t *testing.T) {
	var req recallRequest
	body := `{"query":"q","namespace":"hermes","limit":7,"min_score":0.15,"exclude_types":["note","conversation"]}`
	if err := json.Unmarshal([]byte(body), &req); err != nil {
		t.Fatalf("unmarshal %s: %v", body, err)
	}
	if req.Query != "q" || req.Namespace != "hermes" || req.Limit != 7 {
		t.Errorf("base keys not bound: %+v", req)
	}
	if req.MinScore != 0.15 {
		t.Errorf("min_score not bound: %+v", req)
	}
	if len(req.ExcludeTypes) != 2 || req.ExcludeTypes[0] != "note" || req.ExcludeTypes[1] != "conversation" {
		t.Errorf("exclude_types not bound: %+v", req)
	}
}

// TestHandleRecallAppliesFilters is the end-to-end HTTP proof that the declared
// filters reach the pipeline and shrink the result set.
func TestHandleRecallAppliesFilters(t *testing.T) {
	srv := newRecallHTTPServer(t)

	count, types := postRecall(t, srv, `{"query":"alpha project planning","namespace":"hermes","limit":20}`)
	if count != 3 {
		t.Fatalf("unfiltered count = %d (types %v), want 3", count, types)
	}

	count, _ = postRecall(t, srv, `{"query":"alpha project planning","namespace":"hermes","limit":20,"min_score":2.0}`)
	if count != 0 {
		t.Fatalf("min_score=2.0 count = %d, want 0", count)
	}

	count, types = postRecall(t, srv, `{"query":"alpha project planning","namespace":"hermes","limit":20,"exclude_types":["note","conversation"]}`)
	if count != 1 || len(types) != 1 || types[0] != "decision" {
		t.Fatalf("exclude_types count/types = %d %v, want 1 [decision]", count, types)
	}

	// Filter before limit: the top-scoring type is excluded, so limit=1 must
	// return the best surviving result instead of nothing.
	count, types = postRecall(t, srv, `{"query":"alpha project planning","namespace":"hermes","limit":1,"exclude_types":["decision"]}`)
	if count != 1 || len(types) != 1 {
		t.Fatalf("filter-before-limit count/types = %d %v, want 1 result", count, types)
	}
}

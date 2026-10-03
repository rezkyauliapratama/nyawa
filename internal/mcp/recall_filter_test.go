package mcp

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// recallFakeEmbedder keeps the vector leg deterministic without a real model.
type recallFakeEmbedder struct{}

func (recallFakeEmbedder) Embed(text string) ([]float32, error) { return []float32{1, 0, 0}, nil }
func (recallFakeEmbedder) Name() string                         { return "fake" }
func (recallFakeEmbedder) Dims() int                            { return 3 }
func (recallFakeEmbedder) Available() bool                      { return true }

// recallFakeStore serves a fixed retrieval set: memA (decision) also comes
// back from the vector leg so it scores ~1.05, while memB (note) and memC
// (conversation) stay around 0.54 via FTS5 only.
type recallFakeStore struct {
	fts    []string
	vector []string
	mems   map[string]*types.Memory
}

func (f *recallFakeStore) FTS5Search(query string, topK int, namespace string) ([]string, error) {
	return f.fts, nil
}
func (f *recallFakeStore) FTS5SearchAt(query string, tq store.TimeQuery) ([]string, error) {
	return f.fts, nil
}
func (f *recallFakeStore) VectorSearch(queryVector []float32, topK int, namespace string) ([]string, error) {
	return f.vector, nil
}
func (f *recallFakeStore) VectorSearchAt(query []float32, tq store.TimeQuery) ([]string, error) {
	return f.vector, nil
}
func (f *recallFakeStore) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	out := make([]*types.Memory, 0, len(ids))
	for _, id := range ids {
		if m, ok := f.mems[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}
func (f *recallFakeStore) IncrementAccessCount(id string) error              { return nil }
func (f *recallFakeStore) ListNamespaces() (map[string]int, error)           { return nil, nil }
func (f *recallFakeStore) ArchiveSuperseded(archivePath string) (int, error) { return 0, nil }
func (f *recallFakeStore) TraverseGraph(seeds []string, depth, limit int) ([]graph.TraversalResult, error) {
	return nil, nil
}
func (f *recallFakeStore) ListEntityNames(limit int) ([]string, error) { return nil, nil }

func newRecallTestServer(t *testing.T) (*Server, *bytes.Buffer) {
	t.Helper()
	now := time.Now()
	st := &recallFakeStore{
		fts:    []string{"memA", "memB", "memC"},
		vector: []string{"memA"},
		mems: map[string]*types.Memory{
			"memA": {ID: "memA", Content: "A", Type: types.TypeDecision, Namespace: "hermes", CreatedAt: now},
			"memB": {ID: "memB", Content: "B", Type: types.TypeNote, Namespace: "hermes", CreatedAt: now},
			"memC": {ID: "memC", Content: "C", Type: types.MemoryType("conversation"), Namespace: "hermes", CreatedAt: now},
		},
	}
	p := search.NewPipeline(st, embedder.NewPriorityChain(recallFakeEmbedder{}),
		types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})
	buf := &bytes.Buffer{}
	return &Server{pipeline: p, writer: json.NewEncoder(buf)}, buf
}

// recallPayload decodes the JSON-RPC envelope written by handleRecall and
// returns the result count and the types actually returned.
func recallPayload(t *testing.T, buf *bytes.Buffer) (int, []string) {
	t.Helper()
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(buf.Bytes(), &resp); err != nil {
		t.Fatalf("decode JSON-RPC response %q: %v", buf.String(), err)
	}
	if resp.Error != nil {
		t.Fatalf("handleRecall returned error: %s", resp.Error.Message)
	}
	if len(resp.Result.Content) == 0 {
		t.Fatalf("no content in response: %s", buf.String())
	}
	var payload struct {
		Results []struct {
			ID, Content, Type string
		}
		Count int
	}
	if err := json.Unmarshal([]byte(resp.Result.Content[0].Text), &payload); err != nil {
		t.Fatalf("decode tool payload %q: %v", resp.Result.Content[0].Text, err)
	}
	gotTypes := make([]string, len(payload.Results))
	for i, r := range payload.Results {
		gotTypes[i] = r.Type
	}
	return payload.Count, gotTypes
}

// TestRecallArgsBindDeclaredSchemaProperties is the explicit guard that every
// property the nyawa_recall schema declares reaches the handler's arg struct
// (min_score and exclude_types were declared nowhere and bound nowhere).
func TestRecallArgsBindDeclaredSchemaProperties(t *testing.T) {
	raw := json.RawMessage(`{"query":"q","namespace":"hermes","limit":7,"min_score":0.15,"exclude_types":["note","conversation"]}`)
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		t.Fatalf("unmarshal recall args: %v", err)
	}
	if args.Query != "q" || args.Namespace != "hermes" || args.Limit != 7 {
		t.Errorf("base properties not bound: %+v", args)
	}
	if args.MinScore != 0.15 {
		t.Errorf("min_score not bound: %+v", args)
	}
	if len(args.ExcludeTypes) != 2 || args.ExcludeTypes[0] != "note" || args.ExcludeTypes[1] != "conversation" {
		t.Errorf("exclude_types not bound: %+v", args)
	}

	// The schema must declare both properties, otherwise clients cannot send
	// them at all.
	s := &Server{}
	var schema inputSchema
	for _, tool := range s.tools() {
		if tool.Name == "nyawa_recall" {
			schema = tool.InputSchema
		}
	}
	for _, prop := range []string{"query", "namespace", "limit", "min_score", "exclude_types"} {
		if _, ok := schema.Properties[prop]; !ok {
			t.Errorf("nyawa_recall schema does not declare %q", prop)
		}
	}
	if it := schema.Properties["exclude_types"].Items; it == nil || it.Type != "string" {
		t.Errorf("exclude_types schema missing items:string: %+v", schema.Properties["exclude_types"])
	}
}

// TestHandleRecallFiltersReduceResultCount is the end-to-end proof that the
// declared filters reach the pipeline: each filtered call returns strictly
// fewer results than the unfiltered one, with the excluded types gone.
func TestHandleRecallFiltersReduceResultCount(t *testing.T) {
	cases := []struct {
		name  string
		args  string
		count int
		types []string
	}{
		{
			name:  "no filter",
			args:  `{"query":"q","namespace":"hermes","limit":20}`,
			count: 3,
			types: []string{"decision", "note", "conversation"},
		},
		{
			name:  "min_score high",
			args:  `{"query":"q","namespace":"hermes","limit":20,"min_score":1.0}`,
			count: 1,
			types: []string{"decision"},
		},
		{
			name:  "exclude_types conversation",
			args:  `{"query":"q","namespace":"hermes","limit":20,"exclude_types":["conversation"]}`,
			count: 2,
			types: []string{"decision", "note"},
		},
		{
			name:  "plugin combination",
			args:  `{"query":"q","namespace":"hermes","limit":20,"min_score":1.0,"exclude_types":["note","conversation"]}`,
			count: 1,
			types: []string{"decision"},
		},
		{
			name:  "filter before limit",
			args:  `{"query":"q","namespace":"hermes","limit":1,"exclude_types":["decision","conversation"]}`,
			count: 1,
			types: []string{"note"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, buf := newRecallTestServer(t)
			srv.handleRecall(1, json.RawMessage(tc.args))
			count, types := recallPayload(t, buf)
			if count != tc.count {
				t.Fatalf("handleRecall(%s) count = %d, want %d (types %v)", tc.args, count, tc.count, types)
			}
			if len(types) != len(tc.types) {
				t.Fatalf("handleRecall(%s) types = %v, want %v", tc.args, types, tc.types)
			}
			for i := range types {
				if types[i] != tc.types[i] {
					t.Fatalf("handleRecall(%s) types = %v, want %v", tc.args, types, tc.types)
				}
			}
		})
	}
}

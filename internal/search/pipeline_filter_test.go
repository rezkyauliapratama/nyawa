package search

import (
	"testing"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// fakeEmbedder is always available and returns a fixed vector, so the vector
// leg of the recall pipeline runs without touching a real model.
type fakeEmbedder struct{}

func (fakeEmbedder) Embed(text string) ([]float32, error) { return []float32{1, 0, 0}, nil }
func (fakeEmbedder) Name() string                         { return "fake" }
func (fakeEmbedder) Dims() int                            { return 3 }
func (fakeEmbedder) Available() bool                      { return true }

// fakeStore serves a fixed retrieval result: three memories from FTS5, one of
// which (memA) also comes back from the vector leg, so scores are well
// separated (memA ~1.05, memB ~0.54, memC ~0.53).
type fakeStore struct {
	fts    []string
	vector []string
	mems   map[string]*types.Memory
}

func (f *fakeStore) FTS5Search(query string, topK int, namespace string) ([]string, error) {
	return f.fts, nil
}
func (f *fakeStore) FTS5SearchAt(query string, tq store.TimeQuery) ([]string, error) {
	return f.fts, nil
}
func (f *fakeStore) VectorSearch(queryVector []float32, topK int, namespace string) ([]string, error) {
	return f.vector, nil
}
func (f *fakeStore) VectorSearchAt(query []float32, tq store.TimeQuery) ([]string, error) {
	return f.vector, nil
}
func (f *fakeStore) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	out := make([]*types.Memory, 0, len(ids))
	for _, id := range ids {
		if m, ok := f.mems[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}
func (f *fakeStore) IncrementAccessCount(id string) error              { return nil }
func (f *fakeStore) ListNamespaces() (map[string]int, error)           { return nil, nil }
func (f *fakeStore) ArchiveSuperseded(archivePath string) (int, error) { return 0, nil }
func (f *fakeStore) TraverseGraph(seeds []string, depth, limit int) ([]graph.TraversalResult, error) {
	return nil, nil
}
func (f *fakeStore) ListEntityNames(limit int) ([]string, error) { return nil, nil }

func newFilterPipeline() *Pipeline {
	now := time.Now()
	st := &fakeStore{
		fts:    []string{"memA", "memB", "memC"},
		vector: []string{"memA"},
		mems: map[string]*types.Memory{
			"memA": {ID: "memA", Content: "A", Type: types.TypeDecision, Namespace: "hermes", CreatedAt: now},
			"memB": {ID: "memB", Content: "B", Type: types.TypeNote, Namespace: "hermes", CreatedAt: now},
			"memC": {ID: "memC", Content: "C", Type: types.MemoryType("conversation"), Namespace: "hermes", CreatedAt: now},
		},
	}
	return NewPipeline(st, embedder.NewPriorityChain(fakeEmbedder{}), types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})
}

func idsOf(results []*types.MemoryResult) []string {
	out := make([]string, len(results))
	for i, r := range results {
		out[i] = r.ID
	}
	return out
}

func equalIDs(got []string, want ...string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestRecallMinScoreFilterDropsLowScores is the unit guard for the filter the
// README documents but that nothing ever bound: a min_score above the lower
// results must drop them while keeping the high-scoring one.
func TestRecallMinScoreFilterDropsLowScores(t *testing.T) {
	p := newFilterPipeline()

	all, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10})
	if err != nil {
		t.Fatalf("unfiltered search: %v", err)
	}
	if len(all) != 3 {
		t.Fatalf("unfiltered results = %v, want 3", idsOf(all))
	}
	scores := map[string]float64{}
	for _, r := range all {
		scores[r.ID] = r.Score
	}

	kept, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10, MinScore: 0.75})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if !equalIDs(idsOf(kept), "memA") {
		t.Fatalf("min_score=0.75 results = %v (scores %v), want [memA]", idsOf(kept), scores)
	}

	none, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10, MinScore: 1.2})
	if err != nil {
		t.Fatalf("over-threshold search: %v", err)
	}
	if len(none) != 0 {
		t.Fatalf("min_score=1.2 results = %v, want none", idsOf(none))
	}
}

// TestRecallExcludeTypesFilterKeepsOtherTypes proves excluded types disappear
// while every other type still comes through.
func TestRecallExcludeTypesFilterKeepsOtherTypes(t *testing.T) {
	p := newFilterPipeline()

	got, err := p.Search(types.StoreQuery{
		QueryText: "q", Namespace: "hermes", Limit: 10,
		ExcludeTypes: []string{"note", "conversation"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !equalIDs(idsOf(got), "memA") {
		t.Fatalf("exclude note+conversation results = %v, want [memA]", idsOf(got))
	}

	got, err = p.Search(types.StoreQuery{
		QueryText: "q", Namespace: "hermes", Limit: 10,
		ExcludeTypes: []string{"note"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !equalIDs(idsOf(got), "memA", "memC") {
		t.Fatalf("exclude note results = %v, want [memA memC]", idsOf(got))
	}
}

// TestRecallFiltersApplyBeforeLimit proves ordering: with the top-scoring type
// excluded, limit=1 must return the best *surviving* result, not an empty set
// produced by truncating first and filtering afterwards.
func TestRecallFiltersApplyBeforeLimit(t *testing.T) {
	p := newFilterPipeline()

	got, err := p.Search(types.StoreQuery{
		QueryText: "q", Namespace: "hermes", Limit: 1,
		ExcludeTypes: []string{"decision", "conversation"},
	})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !equalIDs(idsOf(got), "memB") {
		t.Fatalf("filter-then-limit results = %v, want [memB]", idsOf(got))
	}
}

// TestRecallLimitCapsResults proves Limit is honoured after filtering.
func TestRecallLimitCapsResults(t *testing.T) {
	p := newFilterPipeline()

	got, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 2})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if !equalIDs(idsOf(got), "memA", "memB") {
		t.Fatalf("limit=2 results = %v, want [memA memB]", idsOf(got))
	}
}

// TestRecallCacheIsKeyedOnFilters guards the interaction that made the filters
// unobservable: an unfiltered query and a filtered one share the same query
// text, so a cache keyed only on the text serves the wrong entry.
func TestRecallCacheIsKeyedOnFilters(t *testing.T) {
	p := newFilterPipeline()

	first, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10})
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	p.ReleaseResults(first)

	filtered, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10, MinScore: 0.75})
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if !equalIDs(idsOf(filtered), "memA") {
		t.Fatalf("filtered search after unfiltered one = %v, want [memA]", idsOf(filtered))
	}
	p.ReleaseResults(filtered)

	// A different namespace must not reuse the cached result either.
	other, err := p.Search(types.StoreQuery{QueryText: "q", Namespace: "other", Limit: 10})
	if err != nil {
		t.Fatalf("other namespace search: %v", err)
	}
	if len(other) != 3 {
		t.Fatalf("other namespace results = %v, want the fake store's 3 (namespace is not filtered by the fake)", idsOf(other))
	}
}

// TestRecallCacheSurvivesResultRelease guards the pool/cache interaction: the
// caller releases results back into the result pool (which resets them), so
// a cache that stores those same objects serves blanked entries on the second
// identical query.
func TestRecallCacheSurvivesResultRelease(t *testing.T) {
	p := newFilterPipeline()
	q := types.StoreQuery{QueryText: "q", Namespace: "hermes", Limit: 10}

	first, err := p.Search(q)
	if err != nil {
		t.Fatalf("cold search: %v", err)
	}
	p.ReleaseResults(first)

	cached, err := p.Search(q)
	if err != nil {
		t.Fatalf("cache-hit search: %v", err)
	}
	if len(cached) != 3 {
		t.Fatalf("cache-hit results = %d, want 3", len(cached))
	}
	for _, r := range cached {
		if r.ID == "" || r.Content == "" {
			t.Fatalf("cache-hit result was reset by the result pool: %+v", r)
		}
	}
}

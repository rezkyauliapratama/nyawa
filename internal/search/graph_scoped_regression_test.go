package search

import (
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type scopedGraphRegressionStore struct {
	*fakeStore
	entityNames       []string
	graphCandidates   []graph.TraversalResult
	ignoreScopeFilter bool
	scopedCalls       int
	unscopedCalls     int
}

func (s *scopedGraphRegressionStore) ListEntityNames(int) ([]string, error) {
	return s.entityNames, nil
}

func (s *scopedGraphRegressionStore) TraverseGraph(_ []string, _, limit int) ([]graph.TraversalResult, error) {
	s.unscopedCalls++
	return scopedGraphRegressionTopK(s.graphCandidates, "", limit, s.fakeStore.mems)
}

func (s *scopedGraphRegressionStore) TraverseGraphScoped(_ []string, _, limit int, namespace string) ([]graph.TraversalResult, error) {
	s.scopedCalls++
	if s.ignoreScopeFilter {
		if len(s.graphCandidates) > limit {
			return s.graphCandidates[:limit], nil
		}
		return s.graphCandidates, nil
	}
	return scopedGraphRegressionTopK(s.graphCandidates, namespace, limit, s.fakeStore.mems)
}

func scopedGraphRegressionTopK(candidates []graph.TraversalResult, namespace string, limit int, memories map[string]*types.Memory) ([]graph.TraversalResult, error) {
	eligible := make([]graph.TraversalResult, 0, len(candidates))
	for _, candidate := range candidates {
		memory := memories[candidate.MemoryID]
		if namespace == "" || (memory != nil && memory.Namespace == namespace) {
			eligible = append(eligible, candidate)
		}
	}
	if limit > 0 && len(eligible) > limit {
		eligible = eligible[:limit]
	}
	return eligible, nil
}

func newScopedGraphRegressionPipeline(t *testing.T, store *scopedGraphRegressionStore) *Pipeline {
	t.Helper()
	return NewPipeline(store, embedder.NewPriorityChain(fakeEmbedder{}), types.DefaultConfig().Search)
}

func scopedGraphRegressionFixtures(now time.Time) (map[string]*types.Memory, []graph.TraversalResult) {
	memories := map[string]*types.Memory{
		"alpha-graph": {ID: "alpha-graph", Content: "alpha graph-only match", Namespace: "alpha", Type: types.TypeNote, CreatedAt: now},
		"alpha-fts":   {ID: "alpha-fts", Content: "alpha keyword result", Namespace: "alpha", Type: types.TypeNote, CreatedAt: now},
		"beta-fts":    {ID: "beta-fts", Content: "beta keyword result", Namespace: "beta", Type: types.TypeNote, CreatedAt: now},
	}
	candidates := make([]graph.TraversalResult, 0, 7)
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("beta-graph-%d", i)
		memories[id] = &types.Memory{ID: id, Content: "beta graph-only match", Namespace: "beta", Type: types.TypeNote, CreatedAt: now}
		candidates = append(candidates, graph.TraversalResult{MemoryID: id, Score: float64(9 - i)})
	}
	candidates = append(candidates, graph.TraversalResult{MemoryID: "alpha-graph", Score: 0.01})
	return memories, candidates
}

func TestGraphScopedRecallAndCacheIsolationRegression(t *testing.T) {
	now := time.Now()
	memories, candidates := scopedGraphRegressionFixtures(now)
	store := &scopedGraphRegressionStore{
		fakeStore:       &fakeStore{mems: memories},
		entityNames:     []string{"SharedEntity"},
		graphCandidates: candidates,
	}
	pipeline := newScopedGraphRegressionPipeline(t, store)
	query := types.StoreQuery{QueryText: "SharedEntity", Namespace: "alpha", Limit: 1}

	for call := 0; call < 2; call++ {
		results, err := pipeline.Search(query)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 1 || results[0].ID != "alpha-graph" || results[0].Namespace != "alpha" {
			t.Fatalf("alpha graph-only scoped result call %d = %#v, want only alpha-graph", call+1, results)
		}
		pipeline.ReleaseResults(results)
	}
	if store.scopedCalls != 1 {
		t.Fatalf("same query should use isolated cache; scoped traversal calls = %d, want 1", store.scopedCalls)
	}

	betaResults, err := pipeline.Search(types.StoreQuery{QueryText: "SharedEntity", Namespace: "beta", Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(betaResults) != 1 || betaResults[0].ID != "beta-graph-0" || betaResults[0].Namespace != "beta" {
		t.Fatalf("beta scoped results = %#v, want beta-graph-0", betaResults)
	}
	pipeline.ReleaseResults(betaResults)
	if store.scopedCalls != 2 || store.unscopedCalls != 0 {
		t.Fatalf("scoped/unscoped graph calls = %d/%d, want 2/0", store.scopedCalls, store.unscopedCalls)
	}
}

func TestGraphScopedCacheKeyNULCollisionCannotCrossNamespaceRegression(t *testing.T) {
	now := time.Now()
	memories, candidates := scopedGraphRegressionFixtures(now)
	store := &scopedGraphRegressionStore{
		fakeStore:       &fakeStore{mems: memories},
		entityNames:     []string{"SharedEntity"},
		graphCandidates: candidates,
	}
	pipeline := newScopedGraphRegressionPipeline(t, store)

	alphaQuery := types.StoreQuery{QueryText: "SharedEntity\x00foreign", Namespace: "alpha", Limit: 1}
	foreignQuery := types.StoreQuery{QueryText: "SharedEntity", Namespace: "foreign\x00alpha", Limit: 1}
	if alphaQuery.CacheKey() != foreignQuery.CacheKey() {
		t.Fatalf("expected NUL-delimited cache-key collision, keys differ")
	}

	alphaResults, err := pipeline.Search(alphaQuery)
	if err != nil {
		t.Fatal(err)
	}
	if len(alphaResults) != 1 || alphaResults[0].ID != "alpha-graph" || alphaResults[0].Namespace != "alpha" {
		t.Fatalf("first scoped query results = %#v, want alpha-graph", alphaResults)
	}
	pipeline.ReleaseResults(alphaResults)
	if store.scopedCalls != 1 {
		t.Fatalf("first query scoped traversal calls = %d, want 1", store.scopedCalls)
	}

	foreignResults, err := pipeline.Search(foreignQuery)
	if err != nil {
		t.Fatal(err)
	}
	if len(foreignResults) != 0 {
		t.Fatalf("colliding foreign namespace received cached results: %#v", foreignResults)
	}
	if store.scopedCalls != 1 {
		t.Fatalf("second query missed cache; scoped traversal calls = %d, want unchanged at 1", store.scopedCalls)
	}
}

func TestGraphScopedRecallAdverseAndUnscopedControlsRegression(t *testing.T) {
	now := time.Now()
	memories, candidates := scopedGraphRegressionFixtures(now)

	t.Run("no local graph matches", func(t *testing.T) {
		store := &scopedGraphRegressionStore{fakeStore: &fakeStore{mems: memories}, entityNames: []string{"SharedEntity"}, graphCandidates: candidates}
		results, err := newScopedGraphRegressionPipeline(t, store).Search(types.StoreQuery{QueryText: "SharedEntity", Namespace: "missing", Limit: 3})
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 0 {
			t.Fatalf("no-match namespace returned %#v", results)
		}
	})

	t.Run("hostile scoped store cannot inject foreign graph memory", func(t *testing.T) {
		store := &scopedGraphRegressionStore{fakeStore: &fakeStore{mems: memories}, entityNames: []string{"SharedEntity"}, graphCandidates: candidates, ignoreScopeFilter: true}
		results, err := newScopedGraphRegressionPipeline(t, store).Search(types.StoreQuery{QueryText: "SharedEntity", Namespace: "alpha", Limit: 1})
		if err != nil {
			t.Fatal(err)
		}
		for _, result := range results {
			if result.Namespace != "alpha" {
				t.Fatalf("foreign graph injection survived defense-in-depth filter: %#v", result)
			}
		}
		if len(results) != 0 {
			t.Fatalf("hostile top-K contained only foreign records; got %#v", results)
		}
	})

	t.Run("FTS control is namespace-filtered without graph", func(t *testing.T) {
		store := &scopedGraphRegressionStore{
			fakeStore: &fakeStore{fts: []string{"beta-fts", "alpha-fts"}, mems: memories},
		}
		results, err := newScopedGraphRegressionPipeline(t, store).Search(types.StoreQuery{QueryText: "keyword", Namespace: "alpha", Limit: 5})
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != 1 || results[0].ID != "alpha-fts" || results[0].Namespace != "alpha" {
			t.Fatalf("no-graph FTS scoped results = %#v, want alpha-fts only", results)
		}
	})

	t.Run("empty namespace preserves intentionally mixed graph results", func(t *testing.T) {
		store := &scopedGraphRegressionStore{
			fakeStore:       &fakeStore{mems: memories},
			entityNames:     []string{"SharedEntity"},
			graphCandidates: candidates,
		}
		results, err := newScopedGraphRegressionPipeline(t, store).Search(types.StoreQuery{QueryText: "SharedEntity", Limit: 10})
		if err != nil {
			t.Fatal(err)
		}
		seen := map[string]bool{}
		for _, result := range results {
			seen[result.Namespace] = true
		}
		if !seen["alpha"] || !seen["beta"] {
			t.Fatalf("empty namespace graph result did not retain mixed namespaces: %#v", results)
		}
		if store.unscopedCalls != 1 || store.scopedCalls != 0 {
			t.Fatalf("empty namespace traversal calls scoped/unscoped = %d/%d, want 0/1", store.scopedCalls, store.unscopedCalls)
		}
	})
}

type legacyGraphRegressionStore struct {
	*fakeStore
	unscopedCalls int
}

func (s *legacyGraphRegressionStore) ListEntityNames(int) ([]string, error) {
	return []string{"SharedEntity"}, nil
}

func (s *legacyGraphRegressionStore) TraverseGraph(seeds []string, depth, limit int) ([]graph.TraversalResult, error) {
	s.unscopedCalls++
	return []graph.TraversalResult{{MemoryID: "beta-graph-0", Score: 1}}, nil
}

func TestNamespacedSearchSkipsGraphWhenStoreLacksScopedCapability(t *testing.T) {
	legacy := &legacyGraphRegressionStore{fakeStore: &fakeStore{}}
	pipeline := NewPipeline(legacy, embedder.NewPriorityChain(fakeEmbedder{}), types.DefaultConfig().Search)

	results, err := pipeline.Search(types.StoreQuery{QueryText: "SharedEntity", Namespace: "alpha", Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Fatalf("legacy store without scoped traversal returned graph results: %#v", results)
	}
	if legacy.unscopedCalls != 0 {
		t.Fatalf("namespaced search made %d unscoped traversal calls, want 0", legacy.unscopedCalls)
	}
}

func TestUnscopedPipelineSearchStillTruncatesAtRequestedLimit(t *testing.T) {
	now := time.Now()
	memories, candidates := scopedGraphRegressionFixtures(now)
	regressionStore := &scopedGraphRegressionStore{
		fakeStore:       &fakeStore{mems: memories},
		entityNames:     []string{"SharedEntity"},
		graphCandidates: candidates,
	}
	pipeline := newScopedGraphRegressionPipeline(t, regressionStore)

	results, err := pipeline.Search(types.StoreQuery{QueryText: "SharedEntity", Limit: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Fatalf("unscoped result count = %d, want requested top-5", len(results))
	}
	pipeline.ReleaseResults(results)
}

func TestSQLitePipelineSearchScopesBeforeGraphLimit(t *testing.T) {
	sqliteStore, err := store.NewStore(filepath.Join(t.TempDir(), "pipeline-graph-scope.sqlite"), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sqliteStore.Close() })

	db := sqliteStore.GetDB()
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("pipeline-beta-%d", i)
		if _, err := db.Exec(`INSERT INTO memories(id, content, mem_type, namespace) VALUES (?, ?, ?, 'beta')`, id, "beta graph-only candidate", string(types.TypeNote)); err != nil {
			t.Fatalf("insert beta candidate %q: %v", id, err)
		}
	}
	if _, err := db.Exec(`INSERT INTO memories(id, content, mem_type, namespace) VALUES ('pipeline-alpha', 'alpha graph-only candidate', ?, 'alpha')`, string(types.TypeNote)); err != nil {
		t.Fatalf("insert alpha candidate: %v", err)
	}
	result, err := db.Exec(`INSERT INTO entity_nodes(name, category) VALUES ('PipelineSharedEntity', 'tech')`)
	if err != nil {
		t.Fatalf("insert graph entity: %v", err)
	}
	entityID, err := result.LastInsertId()
	if err != nil {
		t.Fatalf("read graph entity ID: %v", err)
	}
	for i := 0; i < 6; i++ {
		id := fmt.Sprintf("pipeline-beta-%d", i)
		if _, err := db.Exec(`INSERT INTO entity_edges(memory_id, entity_id, weight) VALUES (?, ?, ?)`, id, entityID, float64(20-i)); err != nil {
			t.Fatalf("link beta candidate %q: %v", id, err)
		}
	}
	if _, err := db.Exec(`INSERT INTO entity_edges(memory_id, entity_id, weight) VALUES ('pipeline-alpha', ?, 0.01)`, entityID); err != nil {
		t.Fatalf("link alpha candidate: %v", err)
	}

	pipeline := NewPipeline(sqliteStore, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})
	alphaResults, err := pipeline.Search(types.StoreQuery{QueryText: "PipelineSharedEntity", Namespace: "alpha", Limit: 1})
	if err != nil {
		t.Fatalf("SQLite alpha scoped pipeline search: %v", err)
	}
	if len(alphaResults) != 1 || alphaResults[0].ID != "pipeline-alpha" || alphaResults[0].Namespace != "alpha" {
		t.Fatalf("SQLite alpha scoped results = %#v, want only pipeline-alpha", alphaResults)
	}
	pipeline.ReleaseResults(alphaResults)

	unscopedResults, err := pipeline.Search(types.StoreQuery{QueryText: "PipelineSharedEntity", Limit: 10})
	if err != nil {
		t.Fatalf("SQLite unscoped pipeline search: %v", err)
	}
	seenAlpha, seenBeta := false, false
	for _, candidate := range unscopedResults {
		seenAlpha = seenAlpha || candidate.ID == "pipeline-alpha"
		seenBeta = seenBeta || candidate.Namespace == "beta"
	}
	if !seenAlpha || !seenBeta {
		t.Fatalf("SQLite unscoped results should contain alpha and beta graph candidates; got %#v", unscopedResults)
	}
	pipeline.ReleaseResults(unscopedResults)
}

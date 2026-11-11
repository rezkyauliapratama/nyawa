package search

import (
	"path/filepath"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// newGraphNamespaceIsolationStore builds an isolated SQLite store whose real
// graph traversal returns memories from both namespaces for the Quartz entity.
func newGraphNamespaceIsolationStore(t *testing.T) *store.Store {
	t.Helper()

	s, err := store.NewStore(filepath.Join(t.TempDir(), "namespace-regression.sqlite"), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })

	db := s.GetDB()
	memories := []struct {
		id, content, namespace string
		importance             float64
	}{
		{"alpha-graph", "alpha local graph memory", "alpha", 0},
		{"beta-graph", "beta private graph memory", "beta", 1},
		{"alpha-plain", "plain alpha namespace result", "alpha", 0.4},
		{"beta-plain", "plain beta namespace result", "beta", 0.4},
		{"alpha-shared", "sharedword alpha namespace result", "alpha", 0.4},
		{"beta-shared", "sharedword beta namespace result", "beta", 0.4},
	}
	for _, memory := range memories {
		if _, err := db.Exec(
			`INSERT INTO memories (id, content, mem_type, namespace, importance) VALUES (?, ?, ?, ?, ?)`,
			memory.id, memory.content, string(types.TypeNote), memory.namespace, memory.importance,
		); err != nil {
			t.Fatalf("insert synthetic memory %q: %v", memory.id, err)
		}
	}

	result, err := db.Exec(`INSERT INTO entity_nodes (name, category) VALUES ('Quartz', 'tech')`)
	if err != nil {
		t.Fatalf("insert Quartz entity: %v", err)
	}
	entityID, err := result.LastInsertId()
	if err != nil {
		t.Fatalf("read Quartz entity ID: %v", err)
	}
	for _, memoryID := range []string{"alpha-graph", "beta-graph"} {
		if _, err := db.Exec(
			`INSERT INTO entity_edges (memory_id, entity_id, weight, created_at) VALUES (?, ?, 1.0, datetime('now'))`,
			memoryID, entityID,
		); err != nil {
			t.Fatalf("link %q to Quartz: %v", memoryID, err)
		}
	}
	return s
}

func TestGraphNamespaceIsolationControls(t *testing.T) {
	t.Run("graph traversal reaches both namespace candidates", func(t *testing.T) {
		s := newGraphNamespaceIsolationStore(t)

		// Prove this synthetic setup reaches the real SQLite-backed graph path
		// and presents both scoped and foreign candidates to traversal.
		graphResults, err := s.TraverseGraph([]string{"Quartz"}, 2, 10)
		if err != nil {
			t.Fatalf("traverse Quartz graph: %v", err)
		}
		graphIDs := make(map[string]bool, len(graphResults))
		for _, result := range graphResults {
			graphIDs[result.MemoryID] = true
		}
		for _, id := range []string{"alpha-graph", "beta-graph"} {
			if !graphIDs[id] {
				t.Fatalf("Quartz traversal did not return fixture candidate %q: %v", id, graphIDs)
			}
		}
	})

	t.Run("alpha-scoped graph search excludes beta at low limit", func(t *testing.T) {
		s := newGraphNamespaceIsolationStore(t)
		pipeline := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

		// The graph-only fixture content deliberately omits "Quartz"; only the
		// entity edge should make these candidates available for this query.
		results, err := pipeline.Search(types.StoreQuery{QueryText: "Quartz", Namespace: "alpha", Limit: 1})
		if err != nil {
			t.Fatalf("scoped graph search: %v", err)
		}
		if len(results) != 1 || results[0].ID != "alpha-graph" {
			t.Fatalf("scoped low-limit results = %v, want only alpha-graph (never beta-graph)", idsOf(results))
		}
		for _, result := range results {
			if result.ID == "beta-graph" {
				t.Fatalf("foreign graph candidate leaked into alpha namespace: %v", idsOf(results))
			}
		}
	})

	t.Run("no-graph FTS search remains alpha-scoped", func(t *testing.T) {
		s := newGraphNamespaceIsolationStore(t)
		pipeline := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

		// With no matching entity seed, regular FTS recall remains namespace-scoped.
		results, err := pipeline.Search(types.StoreQuery{QueryText: "plain", Namespace: "alpha", Limit: 10})
		if err != nil {
			t.Fatalf("no-graph scoped search: %v", err)
		}
		ids := make(map[string]bool, len(results))
		for _, result := range results {
			ids[result.ID] = true
		}
		if !ids["alpha-plain"] || ids["beta-plain"] {
			t.Fatalf("no-graph alpha results = %v, want alpha-plain and no beta-plain", idsOf(results))
		}
	})

	t.Run("empty namespace remains intentionally unscoped", func(t *testing.T) {
		s := newGraphNamespaceIsolationStore(t)
		pipeline := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

		results, err := pipeline.Search(types.StoreQuery{QueryText: "sharedword", Limit: 10})
		if err != nil {
			t.Fatalf("empty-namespace search: %v", err)
		}
		ids := make(map[string]bool, len(results))
		for _, result := range results {
			ids[result.ID] = true
		}
		if !ids["alpha-shared"] || !ids["beta-shared"] {
			t.Fatalf("empty-namespace results = %v, want both alpha-shared and beta-shared", idsOf(results))
		}
	})
}

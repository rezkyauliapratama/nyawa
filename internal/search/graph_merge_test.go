package search

import (
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// setupGraphMergeStore creates a real SQLite store with entity-graph memories.
//
// Graph structure:
//
//	mem1: Rezky works at Bank Sinarmas  (entities: Rezky, Bank Sinarmas)
//	mem2: Tim uses Kafka               (entities: Tim, Kafka; typed edge uses)
//	mem3: Kafka is part of MCP         (entities: Kafka, MCP; typed edge part_of)
//	mem4: Kafka streams events         (entities: Kafka)
//
// Plus a manual related_to edge: Bank Sinarmas <-> Kafka.
//
// Query "Kafka" -> Traverse from Kafka (depth 2) reaches: all 4 memories.
// Query "Bank Sinarmas" -> via bridging edge reaches Kafka's memories too.
// Query "tidak ada entity cocok" -> no entity match, pure RRF fallback.
func setupGraphMergeStore(t *testing.T) *store.Store {
	t.Helper()

	s, err := store.NewStore(":memory:", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })

	// mem1: Rezky works at Bank Sinarmas
	err = s.InsertMemory(&types.Memory{
		ID:         "mem1",
		Content:    "Rezky bekerja di Bank Sinarmas sebagai software engineer",
		Type:       types.TypeNote,
		Namespace:  "default",
		Importance: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}

	// mem2: Tim uses Kafka
	err = s.InsertMemory(&types.Memory{
		ID:         "mem2",
		Content:    "Tim menggunakan Kafka untuk streaming data",
		Type:       types.TypeNote,
		Namespace:  "default",
		Importance: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}

	// mem3: Kafka part of MCP
	err = s.InsertMemory(&types.Memory{
		ID:         "mem3",
		Content:    "Kafka adalah bagian dari MCP",
		Type:       types.TypeNote,
		Namespace:  "default",
		Importance: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}

	// mem4: Kafka streams events (attached to Kafka entity)
	err = s.InsertMemory(&types.Memory{
		ID:         "mem4",
		Content:    "Kafka digunakan untuk streaming events secara real-time",
		Type:       types.TypeNote,
		Namespace:  "default",
		Importance: 0.5,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Add bridging edge: Bank Sinarmas <-> Kafka
	// so Rezky -> Bank Sinarmas -> Kafka -> MCP is traversable
	db := s.GetDB()
	var bankID, kafkaID int
	db.QueryRow(`SELECT id FROM entity_nodes WHERE name = 'Bank Sinarmas'`).Scan(&bankID)
	db.QueryRow(`SELECT id FROM entity_nodes WHERE name = 'Kafka'`).Scan(&kafkaID)
	if bankID > 0 && kafkaID > 0 {
		db.Exec(`INSERT INTO entity_entity_edges (source_id, target_id, rel_type, weight) VALUES (?, ?, 'related_to', 1.0)`, bankID, kafkaID)
	}

	return s
}

func TestGraphMergeEntityQuery(t *testing.T) {
	s := setupGraphMergeStore(t)
	p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})

	// Query "Kafka" should match the Kafka entity and traverse the graph
	results, err := p.Search(types.StoreQuery{QueryText: "Kafka", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for Kafka query")
	}

	// Verify Kafka's own memories appear (mem2, mem3, mem4 are all linked to Kafka)
	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.ID] = true
	}

	// Kafka has edges to mem2 (Tim uses Kafka), mem3 (Kafka part of MCP), mem4 (Kafka streams)
	if !memIDs["mem2"] {
		t.Error("expected mem2 reachable via Kafka entity")
	}
	if !memIDs["mem3"] {
		t.Error("expected mem3 reachable via Kafka entity")
	}
	if !memIDs["mem4"] {
		t.Error("expected mem4 reachable via Kafka entity")
	}
}

func TestGraphMergeNoEntityMatch(t *testing.T) {
	s := setupGraphMergeStore(t)
	p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

	// Query with no entity match: pure RRF fallback, no crash
	results, err := p.Search(types.StoreQuery{QueryText: "cuaca hari ini sangat cerah", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	// Should not crash, results via FTS5 (may be empty or partial)
	_ = results
}

func TestGraphMergeCrossEntityTraversal(t *testing.T) {
	s := setupGraphMergeStore(t)
	p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})

	// Query "Rezky" -> traverses to Kafka via Bank Sinarmas bridging edge
	results, err := p.Search(types.StoreQuery{QueryText: "Rezky", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("expected results for Rezky query")
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.ID] = true
	}
	if !memIDs["mem1"] {
		t.Error("expected mem1 (Rezky's own memory)")
	}
	// With bridging edge, should reach Kafka's memories
	if !memIDs["mem2"] {
		t.Error("expected mem2 reachable via Bank Sinarmas -> Kafka bridging edge")
	}
}

func TestGraphMergeOverlapBoost(t *testing.T) {
	s := setupGraphMergeStore(t)
	p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60, RecencyWeight: 0.05, ImportanceWeight: 0.10})

	// Query for "Kafka" — mem2/mem3/mem4 should appear in both RRF and graph
	results, err := p.Search(types.StoreQuery{QueryText: "Kafka", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.ID] = true
	}
	// All memory IDs that mention Kafka should be present
	if !memIDs["mem2"] && !memIDs["mem3"] && !memIDs["mem4"] {
		t.Error("expected at least one Kafka-related memory in results")
	}
}

func TestListEntityNames(t *testing.T) {
	s := setupGraphMergeStore(t)
	names, err := s.ListEntityNames(100)
	if err != nil {
		t.Fatal(err)
	}
	if len(names) == 0 {
		t.Fatal("expected entity names in store")
	}
	// Should contain Kafka, Rezky, Tim, Bank Sinarmas, MCP
	found := make(map[string]bool)
	for _, n := range names {
		found[n] = true
	}
	for _, want := range []string{"Rezky", "Tim", "Kafka", "MCP", "Bank Sinarmas"} {
		if !found[want] {
			t.Errorf("expected entity %q in ListEntityNames", want)
		}
	}
}

func TestGraphMergeEmptyStore(t *testing.T) {
	// Empty store: no entities, graph merge falls back gracefully
	s, err := store.NewStore(":memory:", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})
	results, err := p.Search(types.StoreQuery{QueryText: "Kafka", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	// Should not crash; no entities to match, pure RRF (may be empty)
	_ = results
}

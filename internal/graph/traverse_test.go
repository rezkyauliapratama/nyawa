package graph

import (
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
)

// setupTraverseGraph builds a graph with:
//
//	mem1: Rezky --[works_at]--> Bank Sinarmas
//	mem2: Tim   --[uses]-----> Kafka
//	mem3: Kafka --[part_of]--> MCP
//
// Plus a manual related_to edge connecting Bank Sinarmas <-> Kafka
// so the full chain Rezky -> Bank Sinarmas -> Kafka -> MCP is traversable.
func setupTraverseGraph(t *testing.T) (*Store, *sql.DB) {
	t.Helper()
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// mem1: Rezky works at Bank Sinarmas
	s.InsertMemoryEntities("mem1", extract.Entities{
		People: []string{"Rezky"},
		Tech:   []string{"Bank Sinarmas"},
	})
	s.InferTypedEdges("mem1", "Rezky bekerja di Bank Sinarmas")

	// mem2: Tim uses Kafka
	s.InsertMemoryEntities("mem2", extract.Entities{
		People: []string{"Tim"},
		Tech:   []string{"Kafka"},
	})
	s.InferTypedEdges("mem2", "Tim menggunakan Kafka")

	// mem3: Kafka part of MCP
	s.InsertMemoryEntities("mem3", extract.Entities{
		Tech: []string{"Kafka", "MCP"},
	})
	s.InferTypedEdges("mem3", "Kafka bagian dari MCP")

	// mem4: attached only to MCP (leaf, no outgoing edges)
	// used to verify depth limits: reachable at depth 2 from Tim,
	// NOT reachable at depth 1.
	s.InsertMemoryEntities("mem4", extract.Entities{
		Tech: []string{"MCP"},
	})

	// Add bridging edge: Bank Sinarmas <-> Kafka (related_to)
	// so the full graph is connected for multi-hop traversal tests.
	var bankID, kafkaID int
	db.QueryRow(`SELECT id FROM entity_nodes WHERE name = 'Bank Sinarmas'`).Scan(&bankID)
	db.QueryRow(`SELECT id FROM entity_nodes WHERE name = 'Kafka'`).Scan(&kafkaID)
	if bankID > 0 && kafkaID > 0 {
		db.Exec(`INSERT INTO entity_entity_edges (source_id, target_id, rel_type, weight) VALUES (?, ?, 'related_to', 1.0)`, bankID, kafkaID)
	}

	return s, db
}

func TestTraverseFindsRelatedMemory(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	results, err := s.Traverse([]string{"Rezky"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.MemoryID] = true
	}

	if !memIDs["mem1"] {
		t.Error("expected mem1 (seed's own memory) in results")
	}
	if !memIDs["mem2"] {
		t.Error("expected mem2 reachable via Bank Sinarmas -> Kafka path")
	}
	if !memIDs["mem3"] {
		t.Error("expected mem3 reachable via Bank Sinarmas -> Kafka -> MCP path")
	}

	// Verify path lengths make sense
	for _, r := range results {
		switch r.MemoryID {
		case "mem1":
			if r.PathLength != 0 {
				t.Errorf("mem1 PathLength expected 0, got %d", r.PathLength)
			}
		case "mem2":
			if r.PathLength > 2 {
				t.Errorf("mem2 PathLength expected <= 2, got %d", r.PathLength)
			}
		case "mem3":
			if r.PathLength > 3 {
				t.Errorf("mem3 PathLength expected <= 3, got %d", r.PathLength)
			}
		}
	}
}

func TestTraverseDepthLimit(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Graph chain from Tim: Tim ->[uses]-> Kafka ->[part_of]-> MCP
	// mem4 is attached ONLY to MCP (depth 2 from Tim).
	// At depth 1, Kafka is reached -> mem2/mem3 found, mem4 NOT found.
	results, err := s.Traverse([]string{"Tim"}, 1, 10)
	if err != nil {
		t.Fatal(err)
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.MemoryID] = true
	}

	if !memIDs["mem2"] {
		t.Error("expected mem2 (Tim's own memory) at depth 1")
	}
	if !memIDs["mem3"] {
		t.Error("expected mem3 reachable at depth 1 via Tim -> Kafka")
	}
	if memIDs["mem4"] {
		t.Error("mem4 (MCP-only, depth 2) must NOT be reachable at depth 1")
	}

	// At depth 2, MCP is reached -> mem4 must appear.
	results2, err := s.Traverse([]string{"Tim"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	foundMem4 := false
	for _, r := range results2 {
		if r.MemoryID == "mem4" {
			foundMem4 = true
		}
	}
	if !foundMem4 {
		t.Error("expected mem4 reachable at depth 2 via Tim -> Kafka -> MCP")
	}
}

func TestTraverseDeepDepth(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Depth 2 from Rezky: should discover MCP entity at hop 2
	results, err := s.Traverse([]string{"Rezky"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.MemoryID] = true
	}

	// mem3 reachable both via Kafka (hop 1) and MCP (hop 2)
	if !memIDs["mem3"] {
		t.Error("expected mem3 in results at depth 2")
	}

	// At depth 2 we should discover MCP entity too
	// (mem3 is also attached to MCP, scored via both paths)
}

func TestTraverseUnknownSeed(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	results, err := s.Traverse([]string{"Nobody"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected empty results for unknown seed, got %d results", len(results))
	}

	// Mixed known + unknown: should work with known seeds
	results, err = s.Traverse([]string{"Nobody", "Rezky"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Error("expected results when at least one seed resolves")
	}
}

func TestTraverseLimit(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	results, err := s.Traverse([]string{"Rezky"}, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) > 1 {
		t.Errorf("expected at most 1 result with limit=1, got %d", len(results))
	}
}

func TestTraverseBothDirections(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Traverse from MCP backwards: edges are directed Kafka->MCP (part_of),
	// but Traverse follows BOTH directions so MCP -> Kafka works.
	results, err := s.Traverse([]string{"MCP"}, 2, 10)
	if err != nil {
		t.Fatal(err)
	}

	memIDs := make(map[string]bool)
	for _, r := range results {
		memIDs[r.MemoryID] = true
	}

	if !memIDs["mem3"] {
		t.Error("expected mem3 (MCP's own memory)")
	}
	if !memIDs["mem2"] {
		t.Error("expected mem2 reachable via reverse direction MCP->Kafka")
	}
	if !memIDs["mem1"] {
		t.Error("expected mem1 reachable via MCP->Kafka->Bank Sinarmas->Rezky")
	}
}

func TestTraverseDefaultParams(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// depth=0, limit=0 should use defaults (depth=2, limit=10)
	results, err := s.Traverse([]string{"Rezky"}, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Error("expected results with default params")
	}
}

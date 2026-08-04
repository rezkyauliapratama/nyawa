package graph

import (
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
)

func TestRebuildCreatesEdges(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert 2 memories sharing entity pair A-B -> count reaches 2 -> related_to edge
	e1 := extract.Entities{Tech: []string{"A", "B"}}
	s.InsertMemoryEntities("mem1", e1)

	e2 := extract.Entities{Tech: []string{"A", "B"}}
	s.InsertMemoryEntities("mem2", e2)

	// Verify edge exists before rebuild
	var count int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&count)
	if count == 0 {
		t.Fatal("expected related_to edge before rebuild")
	}

	// Rebuild
	stats, err := s.RebuildGraph()
	if err != nil {
		t.Fatal(err)
	}

	// Edge should still exist
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&count)
	if count == 0 {
		t.Error("expected related_to edge after rebuild")
	}
	if stats.MemoriesScanned < 2 {
		t.Errorf("expected MemoriesScanned >= 2, got %d", stats.MemoriesScanned)
	}
	if stats.EdgesTotal < 1 {
		t.Errorf("expected EdgesTotal >= 1, got %d", stats.EdgesTotal)
	}
}

func TestRebuildPrunesStale(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert a single memory with pair A-B (count=1, no edge created)
	e1 := extract.Entities{Tech: []string{"A", "B"}}
	s.InsertMemoryEntities("mem1", e1)

	// Manually insert a stale related_to edge with weight=1 (stale, count < 2)
	db.Exec(`INSERT OR IGNORE INTO entity_entity_edges (source_id, target_id, rel_type, weight)
		SELECT epc.source_id, epc.target_id, 'related_to', 1.0
		FROM entity_pair_counts epc WHERE epc.count < 2 LIMIT 1`)

	var staleBefore int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&staleBefore)
	if staleBefore == 0 {
		t.Fatal("expected stale edge to be inserted for test setup")
	}

	// Rebuild should prune the stale edge (count < 2 is not promoted)
	stats, err := s.RebuildGraph()
	if err != nil {
		t.Fatal(err)
	}

	var afterCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&afterCount)
	if afterCount != 0 {
		t.Errorf("expected 0 related_to edges after rebuild (stale pruned), got %d", afterCount)
	}
	if stats.EdgesPruned < 1 {
		t.Errorf("expected EdgesPruned >= 1, got %d", stats.EdgesPruned)
	}
}

func TestRebuildPreservesTyped(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert entities and create a typed edge
	e := extract.Entities{People: []string{"Rezky"}, Tech: []string{"Bank Sinarmas"}}
	s.InsertMemoryEntities("mem1", e)
	s.InferTypedEdges("mem1", "Rezky bekerja di Bank Sinarmas")

	// Also create some co-occurrence edges
	e2 := extract.Entities{Tech: []string{"Go", "Docker"}}
	s.InsertMemoryEntities("mem2", e2)
	s.InsertMemoryEntities("mem3", extract.Entities{Tech: []string{"Go", "Docker"}})

	// Verify typed edge exists before rebuild
	var typedBefore int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'works_at'`).Scan(&typedBefore)
	if typedBefore == 0 {
		t.Fatal("expected works_at edge before rebuild")
	}

	// Rebuild
	stats, err := s.RebuildGraph()
	if err != nil {
		t.Fatal(err)
	}

	// Typed edge MUST still exist
	var typedAfter int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'works_at'`).Scan(&typedAfter)
	if typedAfter == 0 {
		t.Error("expected works_at edge preserved after rebuild, got 0")
	}

	// related_to edge should also be rebuilt for Go-Docker pair (count=2)
	var relatedCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&relatedCount)
	if relatedCount == 0 {
		t.Error("expected related_to edge after rebuild for co-occurring pair")
	}

	_ = stats
}

func TestRebuildStatsSane(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert enough data for meaningful stats
	s.InsertMemoryEntities("mem1", extract.Entities{Tech: []string{"A", "B", "C"}})
	s.InsertMemoryEntities("mem2", extract.Entities{Tech: []string{"A", "B", "C"}})
	s.InsertMemoryEntities("mem3", extract.Entities{Tech: []string{"A", "D"}})

	stats, err := s.RebuildGraph()
	if err != nil {
		t.Fatal(err)
	}

	// Verify stats align with DB state
	var nodesTotal int
	db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&nodesTotal)
	if stats.NodesTotal != nodesTotal {
		t.Errorf("NodesTotal mismatch: stats=%d db=%d", stats.NodesTotal, nodesTotal)
	}

	var edgesTotal int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&edgesTotal)
	if stats.EdgesTotal != edgesTotal {
		t.Errorf("EdgesTotal mismatch: stats=%d db=%d", stats.EdgesTotal, edgesTotal)
	}

	if stats.MemoriesScanned != 3 {
		t.Errorf("expected MemoriesScanned=3, got %d", stats.MemoriesScanned)
	}

	if stats.NodesTotal > 0 && stats.EdgesTotal > 0 && stats.AvgDegree <= 0 {
		t.Error("expected AvgDegree > 0 when edges exist")
	}

	// With 4 nodes (A,B,C,D) and edges for pairs with count >= 2:
	// Pairs: A-B (2x in mem1+mem2 -> count=2 -> edge), A-C (2x -> edge), B-C (2x -> edge), A-D (1x -> no edge)
	// So should be 3 edges
	if stats.EdgesTotal != 3 {
		t.Errorf("expected 3 edges (A-B, A-C, B-C all count>=2), got %d", stats.EdgesTotal)
	}
}

func TestRebuildEmptyGraph(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	stats, err := s.RebuildGraph()
	if err != nil {
		t.Fatal(err)
	}

	if stats.NodesTotal != 0 {
		t.Errorf("expected 0 nodes for empty graph, got %d", stats.NodesTotal)
	}
	if stats.EdgesTotal != 0 {
		t.Errorf("expected 0 edges for empty graph, got %d", stats.EdgesTotal)
	}
	if stats.AvgDegree != 0 {
		t.Errorf("expected AvgDegree=0 for empty graph, got %f", stats.AvgDegree)
	}
}

// TestReextractBackfillsTech verifies that memories stored before the tech
// dictionary grew (e.g. DeepSeek, AWS Bedrock, MCP) get their entities
// backfilled by ReextractEntities, with alias normalization to canonical
// names.
func TestReextractBackfillsTech(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Graph store does not own the memories table — create a minimal copy
	// for the re-extract test.
	if _, err := db.Exec(
		`CREATE TABLE IF NOT EXISTS memories (
			id TEXT PRIMARY KEY, content TEXT NOT NULL,
			mem_type TEXT, namespace TEXT, importance REAL DEFAULT 0.5,
			pinned INTEGER DEFAULT 0, created_at TEXT, updated_at TEXT,
			superseded_at TEXT, access_count INTEGER DEFAULT 0)`); err != nil {
		t.Fatal(err)
	}

	// Simulate an OLD memory stored before the dictionary had DeepSeek/Bedrock:
	// insert the memory row directly (no entities) — the old pipeline missed them.
	if _, err := db.Exec(
		`INSERT INTO memories (id, content, mem_type, namespace, importance, pinned, created_at, updated_at)
		 VALUES ('mem_old', 'running deepseek via AWS Bedrock in jakarta', 'note', 'default', 0.5, 0, datetime('now'), datetime('now'))`,
	); err != nil {
		t.Fatal(err)
	}

	// A newer memory with MCP (also in new dictionary)
	if _, err := db.Exec(
		`INSERT INTO memories (id, content, mem_type, namespace, importance, pinned, created_at, updated_at)
		 VALUES ('mem_new', 'hermes uses MCP over stdio', 'note', 'default', 0.5, 0, datetime('now'), datetime('now'))`,
	); err != nil {
		t.Fatal(err)
	}

	// Before re-extract: no entity nodes at all
	var nodesBefore int
	db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&nodesBefore)
	if nodesBefore != 0 {
		t.Fatalf("expected 0 nodes before reextract, got %d", nodesBefore)
	}

	stats, err := s.ReextractEntities()
	if err != nil {
		t.Fatal(err)
	}
	if stats.MemoriesScanned != 2 {
		t.Errorf("expected 2 memories scanned, got %d", stats.MemoriesScanned)
	}
	if stats.EntitiesAdded < 3 {
		t.Errorf("expected >= 3 entities added (DeepSeek, AWS, AWS Bedrock, MCP), got %d", stats.EntitiesAdded)
	}

	// Verify canonical nodes exist
	for _, want := range []string{"DeepSeek", "AWS Bedrock", "MCP"} {
		var id int
		err := db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, want).Scan(&id)
		if err != nil {
			t.Errorf("expected entity node %q to exist after reextract (err=%v)", want, err)
		}
	}

	// Idempotent: second run should not duplicate entity_edges rows
	var edgesBefore int
	db.QueryRow(`SELECT COUNT(*) FROM entity_edges`).Scan(&edgesBefore)
	if _, err := s.ReextractEntities(); err != nil {
		t.Fatal(err)
	}
	var edgesAfter int
	db.QueryRow(`SELECT COUNT(*) FROM entity_edges`).Scan(&edgesAfter)
	if edgesAfter != edgesBefore {
		t.Errorf("reextract not idempotent: edges %d -> %d", edgesBefore, edgesAfter)
	}
}

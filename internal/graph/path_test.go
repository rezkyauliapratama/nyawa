package graph

import (
	"testing"
)

func TestFindPathRezkyToMCP(t *testing.T) {
	s, db := setupTraverseGraph(t)
	_ = db

	path, err := s.FindPath("Rezky", "MCP", 4)
	if err != nil {
		t.Fatal(err)
	}
	if path == nil {
		// Debug: check entities and edges exist
		var n int
		db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&n)
		t.Logf("entity_nodes: %d", n)
		rows, _ := db.Query(`SELECT id, name FROM entity_nodes ORDER BY id`)
		for rows.Next() {
			var id int; var name string
			rows.Scan(&id, &name)
			t.Logf("  node: id=%d name=%s", id, name)
		}
		rows.Close()
		var e int
		db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&e)
		t.Logf("entity_entity_edges: %d", e)
		erows, _ := db.Query(`SELECT source_id, target_id, rel_type FROM entity_entity_edges`)
		for erows.Next() {
			var sid2, tid2 int; var r string
			erows.Scan(&sid2, &tid2, &r)
			t.Logf("  edge: %d -> %d [%s]", sid2, tid2, r)
		}
		erows.Close()
		t.Fatal("expected a path from Rezky to MCP")
	}
	t.Logf("path length: %d", len(path))
	for i, h := range path {
		t.Logf("  hop %d: %s -> %s [%s]", i, h.From, h.To, h.RelType)
	}
	if len(path) != 3 {
		t.Fatalf("expected 3 hops, got %d: %+v", len(path), path)
	}

	// Verify hops: Rezky -> Bank Sinarmas -> Kafka -> MCP
	if path[0].From != "Rezky" || path[0].To != "Bank Sinarmas" {
		t.Errorf("hop 0 expected Rezky->Bank Sinarmas, got %s->%s", path[0].From, path[0].To)
	}
	if path[0].RelType != "works_at" {
		t.Errorf("hop 0 expected works_at, got %s", path[0].RelType)
	}

	if path[1].To != "Kafka" {
		t.Errorf("hop 1 expected to Kafka, got %s", path[1].To)
	}

	if path[2].From != "Kafka" || path[2].To != "MCP" {
		t.Errorf("hop 2 expected Kafka->MCP, got %s->%s", path[2].From, path[2].To)
	}
	if path[2].RelType != "part_of" {
		t.Errorf("hop 2 expected part_of, got %s", path[2].RelType)
	}
}

func TestFindPathTimToMCP(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	path, err := s.FindPath("Tim", "MCP", 4)
	if err != nil {
		t.Fatal(err)
	}
	if path == nil {
		t.Fatal("expected a path from Tim to MCP")
	}
	if len(path) != 2 {
		t.Fatalf("expected 2 hops, got %d: %+v", len(path), path)
	}

	if path[0].RelType != "uses" {
		t.Errorf("hop 0 expected uses, got %s", path[0].RelType)
	}
	if path[1].RelType != "part_of" {
		t.Errorf("hop 1 expected part_of, got %s", path[1].RelType)
	}
}

func TestFindPathNoPath(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Nobody doesn't exist
	path, err := s.FindPath("Rezky", "Nobody", 4)
	if err != nil {
		t.Fatal(err)
	}
	if path != nil {
		t.Errorf("expected nil path for unknown target, got %+v", path)
	}
}

func TestFindPathUnknownSource(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	path, err := s.FindPath("Nobody", "MCP", 4)
	if err != nil {
		t.Fatal(err)
	}
	if path != nil {
		t.Errorf("expected nil path for unknown source, got %+v", path)
	}
}

func TestFindPathDepthTooSmall(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Rezky -> MCP requires 3 hops. At depth 2 it should fail.
	path, err := s.FindPath("Rezky", "MCP", 2)
	if err != nil {
		t.Fatal(err)
	}
	if path != nil {
		t.Errorf("expected nil path with maxDepth=2 (needs 3), got %+v", path)
	}

	// Tim -> MCP requires 2 hops. At depth 1 it should fail.
	path, err = s.FindPath("Tim", "MCP", 1)
	if err != nil {
		t.Fatal(err)
	}
	if path != nil {
		t.Errorf("expected nil path with maxDepth=1 (needs 2), got %+v", path)
	}
}

func TestFindPathDefaultDepth(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	// Default depth (4) should be enough for Rezky -> MCP (3 hops)
	path, err := s.FindPath("Rezky", "MCP", 0)
	if err != nil {
		t.Fatal(err)
	}
	if path == nil {
		t.Fatal("expected path with default depth=4")
	}
	if len(path) != 3 {
		t.Errorf("expected 3 hops, got %d", len(path))
	}
}

func TestFindPathSameEntity(t *testing.T) {
	s, _ := setupTraverseGraph(t)

	path, err := s.FindPath("Rezky", "Rezky", 4)
	if err != nil {
		t.Fatal(err)
	}
	if path != nil {
		t.Errorf("expected nil for same entity, got %+v", path)
	}
}

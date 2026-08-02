package graph

import (
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
)

func TestMigrateCreatesEntityEntityEdges(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	var cnt int
	if err := db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&cnt); err != nil {
		t.Fatal("entity_entity_edges table missing:", err)
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM entity_pair_counts`).Scan(&cnt); err != nil {
		t.Fatal("entity_pair_counts table missing:", err)
	}
	_ = s
}

func TestSingleMemoryNoEdge(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	e := extract.Entities{Tech: []string{"Go", "Docker"}}
	s.InsertMemoryEntities("mem1", e)

	var eeCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&eeCount)
	if eeCount != 0 {
		t.Errorf("expected 0 entity_entity_edges (co-occurrence < 2), got %d", eeCount)
	}
}

func TestCooccurrenceCreatesEdge(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// First memory: Go + Docker
	s.InsertMemoryEntities("mem1", extract.Entities{Tech: []string{"Go", "Docker"}})

	var eeCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&eeCount)
	if eeCount != 0 {
		t.Errorf("after 1 memory, expected 0 edges, got %d", eeCount)
	}

	// Second memory: Go + Docker again (co-occurrence >= 2)
	s.InsertMemoryEntities("mem2", extract.Entities{Tech: []string{"Go", "Docker"}})

	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&eeCount)
	if eeCount == 0 {
		t.Error("expected at least 1 entity_entity_edges row after co-occurrence >= 2")
	}

	// Verify edge has correct rel_type and weight >= 2
	var relType string
	var weight float64
	db.QueryRow(`SELECT rel_type, weight FROM entity_entity_edges LIMIT 1`).Scan(&relType, &weight)
	if relType != "related_to" {
		t.Errorf("expected rel_type='related_to', got %q", relType)
	}
	if weight < 2.0 {
		t.Errorf("expected weight >= 2, got %f", weight)
	}
}

func TestPairNormalization(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// First memory: A,B order
	s.InsertMemoryEntities("mem1", extract.Entities{Tech: []string{"A", "B"}})

	// Second memory: B,A order (reversed)
	s.InsertMemoryEntities("mem2", extract.Entities{Tech: []string{"B", "A"}})

	var eeCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&eeCount)
	if eeCount != 1 {
		t.Errorf("expected exactly 1 normalized edge for (A,B), got %d", eeCount)
	}

	// Verify pair_counts also has exactly 1 row
	var pcCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_pair_counts`).Scan(&pcCount)
	if pcCount != 1 {
		t.Errorf("expected 1 pair count row, got %d", pcCount)
	}
}

func TestStatsIncludesEntityEntityEdges(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert 2 memories sharing a pair to create entity_entity_edges
	s.InsertMemoryEntities("mem1", extract.Entities{Tech: []string{"Go", "Docker"}})
	s.InsertMemoryEntities("mem2", extract.Entities{Tech: []string{"Go", "Docker"}})

	stats, err := s.Stats()
	if err != nil {
		t.Fatal(err)
	}

	ee, ok := stats["entity_entity_edges"].(int)
	if !ok {
		t.Fatal("entity_entity_edges key missing from Stats()")
	}
	if ee == 0 {
		t.Error("expected entity_entity_edges > 0 in Stats()")
	}
}

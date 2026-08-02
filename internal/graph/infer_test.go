package graph

import (
	"database/sql"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
)

func TestInferWorksAt(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert entities first so InferTypedEdges can resolve them
	e := extract.Entities{
		People: []string{"Rezky"},
		Tech:   []string{"Bank Sinarmas"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Rezky bekerja di Bank Sinarmas"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var relType string
	err = db.QueryRow(`SELECT rel_type FROM entity_entity_edges WHERE rel_type = 'works_at'`).Scan(&relType)
	if err != nil {
		t.Fatalf("expected works_at edge, got error: %v", err)
	}
	if relType != "works_at" {
		t.Errorf("expected rel_type='works_at', got %q", relType)
	}
}

func TestInferUses(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert entities: "Tim backend" as people, "Go" and "Docker" as tech
	e := extract.Entities{
		People: []string{"Tim backend"},
		Tech:   []string{"Go", "Docker"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Tim backend menggunakan Go dan Docker"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var count int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'uses'`).Scan(&count)
	if count < 1 {
		t.Errorf("expected at least 1 uses edge, got %d", count)
	}
}

func TestInferNoTypedOnPlainText(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert some entities
	e := extract.Entities{
		People: []string{"Rezky"},
		Tech:   []string{"Go"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Hari ini cuaca sangat cerah dan menyenangkan"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var count int
	db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type != 'related_to'`).Scan(&count)
	if count != 0 {
		t.Errorf("expected 0 typed edges for plain text, got %d", count)
	}
}

func TestInferEnglish(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	e := extract.Entities{
		People: []string{"Rezky"},
		Tech:   []string{"Bank Sinarmas"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Rezky works at Bank Sinarmas"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var relType string
	err = db.QueryRow(`SELECT rel_type FROM entity_entity_edges WHERE rel_type = 'works_at'`).Scan(&relType)
	if err != nil {
		t.Fatalf("expected works_at edge for English text, got error: %v", err)
	}
	if relType != "works_at" {
		t.Errorf("expected rel_type='works_at', got %q", relType)
	}
}

func TestStatsTypedEdges(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// Insert entities and create typed edge
	e := extract.Entities{
		People: []string{"Rezky"},
		Tech:   []string{"Bank Sinarmas"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Rezky bekerja di Bank Sinarmas"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	stats, err := s.Stats()
	if err != nil {
		t.Fatal(err)
	}

	typedEdges, ok := stats["typed_edges"].(int)
	if !ok {
		t.Fatal("typed_edges key missing from Stats()")
	}
	if typedEdges == 0 {
		t.Error("expected typed_edges > 0 in Stats()")
	}

	_, ok = stats["infer_total"].(int64)
	if !ok {
		t.Fatal("infer_total key missing from Stats()")
	}

	missRate, ok := stats["infer_miss_rate"].(float64)
	if !ok {
		t.Fatal("infer_miss_rate key missing from Stats()")
	}
	// With 1 call and 1 matched, miss_rate should be 0.0
	if missRate != 0.0 {
		t.Errorf("expected infer_miss_rate=0.0, got %f", missRate)
	}
}

func TestInferLocatedIn(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	e := extract.Entities{
		Tech:      []string{"Kantor Pusat"},
		Locations: []string{"Jakarta"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Kantor Pusat berlokasi di Jakarta"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var relType string
	err = db.QueryRow(`SELECT rel_type FROM entity_entity_edges WHERE rel_type = 'located_in'`).Scan(&relType)
	if err != nil {
		t.Fatalf("expected located_in edge, got error: %v", err)
	}
	if relType != "located_in" {
		t.Errorf("expected rel_type='located_in', got %q", relType)
	}
}

func TestInferPartOf(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	e := extract.Entities{
		Tech: []string{"Tim Infra", "Divisi Teknologi"},
	}
	s.InsertMemoryEntities("mem1", e)

	content := "Tim Infra bagian dari Divisi Teknologi"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var relType string
	err = db.QueryRow(`SELECT rel_type FROM entity_entity_edges WHERE rel_type = 'part_of'`).Scan(&relType)
	if err != nil {
		t.Fatalf("expected part_of edge, got error: %v", err)
	}
	if relType != "part_of" {
		t.Errorf("expected rel_type='part_of', got %q", relType)
	}
}

func TestInferAutoRegistersEntities(t *testing.T) {
	db, err := sql.Open("sqlite3", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	s, err := NewStore(db)
	if err != nil {
		t.Fatal(err)
	}

	// No entities pre-inserted: InferTypedEdges must auto-register
	// source/target nodes so typed inference works end-to-end.
	content := "Rezky bekerja di Bank Sinarmas"
	if err := s.InferTypedEdges("mem1", content); err != nil {
		t.Fatal(err)
	}

	var nodeCount int
	db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&nodeCount)
	if nodeCount < 2 {
		t.Errorf("expected >= 2 auto-registered entity nodes, got %d", nodeCount)
	}

	var relType string
	err = db.QueryRow(`SELECT rel_type FROM entity_entity_edges WHERE rel_type = 'works_at'`).Scan(&relType)
	if err != nil {
		t.Fatalf("expected works_at edge from auto-registered entities, got error: %v", err)
	}
}

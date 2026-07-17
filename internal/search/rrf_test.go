package search

import (
	"testing"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestRRFFusion(t *testing.T) {
	rrf := NewRRF(60)
	results := rrf.Fuse([]string{"A", "B", "C", "D"}, []string{"B", "D", "E", "F"})
	if results[0].MemoryID != "B" {
		t.Errorf("expected B at top, got %s", results[0].MemoryID)
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	if s := cosineSimilarity(a, b); s > 0.01 {
		t.Errorf("orthogonal should be 0, got %.4f", s)
	}
	if s := cosineSimilarity(a, a); s < 0.99 {
		t.Errorf("identical should be 1, got %.4f", s)
	}
}

func TestPostProcessor(t *testing.T) {
	pp := NewPostProcessor(0.05, 0.10, pool.NewResultPool(4))
	mem := &types.Memory{ID: "t1", Content: "test", Type: types.TypeNote}
	results := pp.Process(
		[]FusionResult{{MemoryID: "t1", Score: 0.05, VectorRank: 1, FTS5Rank: 5}},
		map[string]*types.Memory{"t1": mem}, 1000,
	)
	if len(results) != 1 || results[0].Score < 0.05 {
		t.Fatal("expected valid score")
	}
}

package search

import (
	"testing"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestRRFFusion(t *testing.T) {
	rrf := NewRRF(60)
	vectorIDs := []string{"A", "B", "C", "D"}
	fts5IDs := []string{"B", "D", "E", "F"}
	results := rrf.Fuse(vectorIDs, fts5IDs)
	if len(results) < 4 { t.Fatalf("expected at least 4 results, got %d", len(results)) }
	if results[0].MemoryID != "B" { t.Errorf("expected B at top, got %s", results[0].MemoryID) }
	for _, r := range results {
		expectedScore := 0.0
		if r.VectorRank <= len(vectorIDs) { expectedScore += 1.0 / float64(60+r.VectorRank) }
		if r.FTS5Rank <= len(fts5IDs) { expectedScore += 1.0 / float64(60+r.FTS5Rank) }
		if r.Score != expectedScore { t.Errorf("memory %s: expected score %.6f, got %.6f", r.MemoryID, expectedScore, r.Score) }
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float32{1, 0, 0}; b := []float32{0, 1, 0}; c := []float32{0.5, 0.5, 0}
	if score := cosineSimilarity(a, b); score > 0.01 { t.Errorf("orthogonal vectors should have ~0 similarity, got %.4f", score) }
	if score := cosineSimilarity(a, a); score < 0.99 { t.Errorf("identical vectors should have ~1 similarity, got %.4f", score) }
	if score := cosineSimilarity(a, c); score < 0.69 || score > 0.72 { t.Errorf("expected ~0.707, got %.4f", score) }
}

func TestPostProcessor(t *testing.T) {
	pp := NewPostProcessor(0.05, 0.10, pool.NewResultPool(4))
	mem := &types.Memory{ID: "test_1", Content: "test memory", Type: types.TypeNote, Importance: 0.4}
	memories := map[string]*types.Memory{"test_1": mem}
	fused := []FusionResult{{MemoryID: "test_1", Score: 0.05, VectorRank: 1, FTS5Rank: 5}}
	results := pp.Process(fused, memories, 1000)
	if len(results) != 1 { t.Fatalf("expected 1 result, got %d", len(results)) }
	if r := results[0]; r.Score <= 0 { t.Errorf("expected positive score, got %.4f", r.Score) }
}

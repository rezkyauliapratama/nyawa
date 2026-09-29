package search

import (
	"fmt"
	"testing"
	"time"

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
		expectedScore /= 2.0 / 61.0 // Fuse normalises the fused score to [0,1]
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

// The nearest vector hit must outrank a distractor retrieved far down the list
// but carrying large recency/importance/graph boosts. Before the fused score was
// normalised to [0,1] the raw RRF term (~1/(k+rank)) was smaller than a single
// boost, so "important" memories outranked the true nearest neighbour.
func TestPostProcessorRelevanceOutranksBoosts(t *testing.T) {
	pp := NewPostProcessor(0.05, 0.10, pool.NewResultPool(4))
	near := &types.Memory{ID: "near", Type: types.TypeInsight, CreatedAt: time.Now()}
	far := &types.Memory{ID: "far", Type: types.TypeDecision, AccessCount: 100, EdgeCount: 100, CreatedAt: time.Now()}
	memories := map[string]*types.Memory{"near": near, "far": far}

	vecIDs := []string{"near"}
	for i := 0; i < 148; i++ {
		vecIDs = append(vecIDs, fmt.Sprintf("distractor-%d", i))
	}
	vecIDs = append(vecIDs, "far")
	fused := NewRRF(60).Fuse(vecIDs, nil)
	// "far" must actually be the weakest vector hit.
	var nearRank, farRank int
	for _, fr := range fused {
		if fr.MemoryID == "near" {
			nearRank = fr.VectorRank
		}
		if fr.MemoryID == "far" {
			farRank = fr.VectorRank
		}
	}
	if nearRank != 1 || farRank <= 100 {
		t.Fatalf("test setup: expected near rank 1 and far rank >100, got %d and %d", nearRank, farRank)
	}

	results := pp.Process(fused, memories, float64(time.Now().Unix())/3600.0)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].ID != "near" {
		for _, r := range results {
			t.Logf("%s score=%.4f rrf=%.4f temp=%.4f imp=%.4f", r.ID, r.Score, r.RRFScore, r.TemporalBoost, r.ImportanceBoost)
		}
		t.Errorf("expected nearest 'near' to outrank boosted 'far', got %s first", results[0].ID)
	}
}

// Fuse must return scores in [0,1] so they are comparable with the additive
// recency/importance weights.
func TestRRFNormalisedRange(t *testing.T) {
	fused := NewRRF(60).Fuse([]string{"A", "B"}, []string{"B", "A"})
	for _, fr := range fused {
		if fr.Score < 0 || fr.Score > 1 {
			t.Errorf("memory %s: expected score in [0,1], got %.6f", fr.MemoryID, fr.Score)
		}
	}
	if fused[0].Score < 0.99 {
		t.Errorf("rank-1 in both lists should normalise to ~1.0, got %.6f", fused[0].Score)
	}
}

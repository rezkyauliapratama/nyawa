package types_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestMemoryTypeWeights(t *testing.T) {
	for _, tt := range []struct{ typ types.MemoryType; weight, decay float64 }{
		{types.TypeDecision, 1.0, 720}, {types.TypeNote, 0.4, 168},
		{types.TypeReference, 0.3, 8760}, {types.TypeFact, 0.7, 4320},
	} {
		if w := tt.typ.Weight(); w != tt.weight { t.Errorf("weight %.1f, got %.1f", tt.weight, w) }
		if d := tt.typ.DecayHours(); d != tt.decay { t.Errorf("decay %.0f, got %.0f", tt.decay, d) }
	}
}

func TestSearchConfig(t *testing.T) {
	cfg := types.DefaultConfig()
	if cfg.Search.VectorTopK != 50 { t.Errorf("expected 50, got %d", cfg.Search.VectorTopK) }
	if cfg.Search.RRFK != 5 { t.Errorf("expected 5, got %d", cfg.Search.RRFK) }
}

func TestMemoryResultReset(t *testing.T) {
	vectorDistance, fts5RankScore, crossEncoderScore := 0.0, -0.7, 1.25
	r := &types.MemoryResult{
		Memory: types.Memory{ID: "test", Content: "content", Type: types.TypeNote, Importance: 0.5, EdgeCount: 3, Vector: []float32{1, 2, 3}},
		Score: 0.9, RRFScore: 0.5, Rank: 1, VectorRank: 2, FTS5Rank: 3,
		VectorDistance: &vectorDistance, FTS5RankScore: &fts5RankScore, CrossEncoderScore: &crossEncoderScore,
	}
	r.Reset()
	if r.ID != "" { t.Error("expected empty ID after reset") }
	if r.Content != "" { t.Error("expected empty content after reset") }
	if r.Score != 0 { t.Error("expected zero score after reset") }
	if r.VectorRank != 0 || r.FTS5Rank != 0 { t.Errorf("expected zero modality ranks after reset, got vector=%d fts5=%d", r.VectorRank, r.FTS5Rank) }
	if r.VectorDistance != nil || r.FTS5RankScore != nil || r.CrossEncoderScore != nil { t.Error("expected confidence scores to reset to nil") }
	if len(r.Vector) != 0 { t.Error("expected empty vector after reset") }
}

func TestMemoryResultRawScoresMarshalPresentZeroAndOmitMissing(t *testing.T) {
	zero := 0.0
	r := types.MemoryResult{VectorDistance: &zero, CrossEncoderScore: &zero}
	data, err := json.Marshal(r)
	if err != nil { t.Fatal(err) }
	encoded := string(data)
	if !strings.Contains(encoded, `"vector_distance":0`) { t.Fatalf("zero vector distance missing from %s", encoded) }
	if !strings.Contains(encoded, `"cross_encoder_score":0`) { t.Fatalf("zero cross-encoder score missing from %s", encoded) }
	if strings.Contains(encoded, `"fts5_rank_score"`) { t.Fatalf("missing FTS score should be omitted: %s", encoded) }
}

func TestTypeWeightSum(t *testing.T) {
	// All types should have non-zero weight
	types := []types.MemoryType{
		types.TypeDecision, types.TypeInsight, types.TypeProcedure,
		types.TypeFact, types.TypePreference, types.TypeContext,
		types.TypeNote, types.TypeEvent, types.TypeReference,
	}
	for _, typ := range types {
		if w := typ.Weight(); w <= 0 { t.Errorf("type %s should have positive weight, got %.2f", typ, w) }
		if d := typ.DecayHours(); d <= 0 { t.Errorf("type %s should have positive decay, got %.0f", typ, d) }
	}
}

func TestResultPool(t *testing.T) {
	rp := pool.NewResultPool(4)
	r := rp.Get()
	if r == nil { t.Fatal("expected non-nil result") }
	r.ID = "test"
	rp.Put(r)
	r2 := rp.Get()
	if r2.ID != "" { t.Error("expected reset after put") }
}

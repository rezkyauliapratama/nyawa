package types

import (
	"testing"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
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
	cfg := DefaultConfig()
	if cfg.Search.VectorTopK != 50 { t.Errorf("expected 50, got %d", cfg.Search.VectorTopK) }
	if cfg.Search.RRFK != 60 { t.Errorf("expected 60, got %d", cfg.Search.RRFK) }
}

func TestMemoryResultReset(t *testing.T) {
	r := &types.MemoryResult{
		Memory: types.Memory{ID: "test", Content: "content", Type: types.TypeNote, Importance: 0.5, EdgeCount: 3, Vector: []float32{1, 2, 3}},
		Score: 0.9, RRFScore: 0.5, Rank: 1,
	}
	r.Reset()
	if r.ID != "" { t.Error("expected empty ID after reset") }
	if r.Content != "" { t.Error("expected empty content after reset") }
	if r.Score != 0 { t.Error("expected zero score after reset") }
	if len(r.Vector) != 0 { t.Error("expected empty vector after reset") }
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

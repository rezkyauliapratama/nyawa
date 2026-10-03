package search

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// --- RRF fusion edge cases -------------------------------------------------

func TestRRFFusionVectorOnly(t *testing.T) {
	fused := NewRRF(60).Fuse([]string{"A", "B", "C"}, nil)
	if len(fused) != 3 {
		t.Fatalf("want 3 results, got %d", len(fused))
	}
	if fused[0].MemoryID != "A" || fused[1].MemoryID != "B" || fused[2].MemoryID != "C" {
		t.Fatalf("vector-only order wrong: %+v", fused)
	}
	for _, fr := range fused {
		if fr.VectorRank == math.MaxInt32 {
			t.Errorf("%s should have a vector rank", fr.MemoryID)
		}
		if fr.FTS5Rank != math.MaxInt32 {
			t.Errorf("%s should have no FTS rank", fr.MemoryID)
		}
	}
}

func TestRRFFusionFTSOnly(t *testing.T) {
	fused := NewRRF(60).Fuse(nil, []string{"X", "Y"})
	if len(fused) != 2 {
		t.Fatalf("want 2 results, got %d", len(fused))
	}
	if fused[0].MemoryID != "X" {
		t.Fatalf("fts-only order wrong: %+v", fused)
	}
	if fused[0].VectorRank != math.MaxInt32 {
		t.Errorf("X should have no vector rank")
	}
}

func TestRRFFusionEmpty(t *testing.T) {
	if got := NewRRF(60).Fuse(nil, nil); len(got) != 0 {
		t.Fatalf("empty inputs should fuse to no results, got %d", len(got))
	}
}

// A memory present in both lists must beat one present in only one, and the
// fused relevance must stay within [0,1].
func TestRRFFusionBothListsRewardsOverlap(t *testing.T) {
	fused := NewRRF(60).Fuse([]string{"both", "vOnly"}, []string{"both", "fOnly"})
	if fused[0].MemoryID != "both" {
		t.Fatalf("overlap should rank first, got %+v", fused)
	}
	for _, fr := range fused {
		if fr.Score < 0 || fr.Score > 1 {
			t.Errorf("%s score %.4f outside [0,1]", fr.MemoryID, fr.Score)
		}
	}
}

func TestWeightsFromEnv(t *testing.T) {
	t.Setenv("NYAWA_RECALL_V2", "1")
	t.Setenv("NYAWA_RECALL_W_TYPE", "0.5")
	t.Setenv("NYAWA_RECALL_RRF_K", "10")
	w := WeightsFromConfig(types.SearchConfig{})
	if w.Type != 0.5 {
		t.Errorf("type weight from env = %v, want 0.5", w.Type)
	}
	if w.RRFK != 10 {
		t.Errorf("rrf k from env = %d, want 10", w.RRFK)
	}
	if w.Legacy {
		t.Error("v2 should be enabled by default")
	}

	t.Setenv("NYAWA_RECALL_V2", "0")
	if w := WeightsFromConfig(types.SearchConfig{}); !w.Legacy {
		t.Error("NYAWA_RECALL_V2=0 must enable legacy mode")
	}
}

// Weighted fusion: with a higher FTS weight, the best keyword hit must outrank
// a single weaker vector hit, while a memory both legs agree on still wins.
func TestRRFWeightedFusion(t *testing.T) {
	fused := NewRRF(5).FuseWeighted([]string{"vecOnly"}, []string{"ftsOnly"}, 1, 3)
	if len(fused) != 2 {
		t.Fatalf("want 2 results, got %d", len(fused))
	}
	if fused[0].MemoryID != "ftsOnly" {
		t.Errorf("higher-weighted fts hit should lead, got %s", fused[0].MemoryID)
	}
	both := NewRRF(5).FuseWeighted([]string{"both", "vecOnly"}, []string{"both", "ftsOnly"}, 1, 3)
	if both[0].MemoryID != "both" {
		t.Errorf("overlap should still rank first, got %s", both[0].MemoryID)
	}
}

// Legacy mode must restore the pre-v2 fusion constants so an operator can
// reproduce the old ranking with NYAWA_RECALL_V2=0.
func TestLegacyWeightsRestoreOldFusion(t *testing.T) {
	t.Setenv("NYAWA_RECALL_V2", "0")
	w := WeightsFromConfig(types.SearchConfig{})
	if w.VectorWeight != 1 || w.FTSWeight != 1 {
		t.Errorf("legacy fusion weights = %v/%v, want 1/1", w.VectorWeight, w.FTSWeight)
	}
	if w.OverlapWeight != 0.5 {
		t.Errorf("legacy overlap weight = %v, want 0.5 (the old 1.5x boost)", w.OverlapWeight)
	}
	if w.RRFK != 60 {
		t.Errorf("legacy rrf k = %d, want 60", w.RRFK)
	}
}

// --- Weighting -------------------------------------------------------------

func newPP() *PostProcessor {
	w := fillWeightDefaults(Weights{})
	return NewPostProcessorWithWeights(w, pool.NewResultPool(8))
}

func memFor(id string, typ types.MemoryType, content string) *types.Memory {
	return &types.Memory{ID: id, Content: content, Type: typ, CreatedAt: time.Now()}
}

func processOrder(t *testing.T, pp *PostProcessor, mems ...*types.Memory) []*types.MemoryResult {
	t.Helper()
	fused := make([]FusionResult, len(mems))
	m := map[string]*types.Memory{}
	for i, mem := range mems {
		fused[i] = FusionResult{MemoryID: mem.ID, Score: 0.5} // equal relevance
		m[mem.ID] = mem
	}
	res := pp.Process(fused, m, float64(time.Now().Unix())/3600.0)
	if len(res) != len(mems) {
		t.Fatalf("got %d results, want %d", len(res), len(mems))
	}
	return res
}

// Content length: with identical relevance/type/importance, a concise memory
// must outrank a long one. This is the fix for "long memory wins on size alone".
func TestWeightingContentLength(t *testing.T) {
	pp := newPP()
	short := memFor("short", types.TypeInsight, strings.Repeat("a", 100))
	long := memFor("long", types.TypeInsight, strings.Repeat("a", 6000))
	res := processOrder(t, pp, long, short)
	if res[0].ID != "short" {
		t.Errorf("short memory should outrank long at equal relevance, got %s first", res[0].ID)
	}
	var ls, ll float64
	for _, r := range res {
		if r.ID == "short" {
			ls = r.LengthBoost
		}
		if r.ID == "long" {
			ll = r.LengthBoost
		}
	}
	if !(ls > ll) {
		t.Errorf("length boost short=%.4f long=%.4f, want short>long", ls, ll)
	}
}

// Type: rules/decisions/preferences get a real boost; conversation gets none.
func TestWeightingMemoryType(t *testing.T) {
	pp := newPP()
	rule := memFor("rule", types.TypeRule, "ru")
	conv := memFor("conv", types.TypeConversation, "cv")
	res := processOrder(t, pp, conv, rule)
	if res[0].ID != "rule" {
		t.Errorf("rule should outrank conversation, got %s first", res[0].ID)
	}
	for _, r := range res {
		switch r.ID {
		case "conv":
			if r.TypeBoost != 0 {
				t.Errorf("conversation must not be type-boosted, got %.4f", r.TypeBoost)
			}
		case "rule":
			if r.TypeBoost <= 0 {
				t.Errorf("rule should be type-boosted, got %.4f", r.TypeBoost)
			}
		}
	}
}

func TestWeightingImportanceAndAccess(t *testing.T) {
	pp := newPP()
	hi := memFor("hi", types.TypeInsight, "x")
	hi.Importance = 1.0
	lo := memFor("lo", types.TypeInsight, "y")
	lo.Importance = 0.0
	res := processOrder(t, pp, lo, hi)
	if res[0].ID != "hi" {
		t.Errorf("higher importance should rank first, got %s", res[0].ID)
	}

	hot := memFor("hot", types.TypeInsight, "x")
	hot.AccessCount = 10
	cold := memFor("cold", types.TypeInsight, "y")
	res = processOrder(t, pp, cold, hot)
	if res[0].ID != "hot" {
		t.Errorf("frequently accessed memory should rank first, got %s", res[0].ID)
	}
}

func TestWeightingRecency(t *testing.T) {
	pp := newPP()
	now := time.Now()
	fresh := memFor("fresh", types.TypeInsight, "x")
	fresh.CreatedAt = now
	old := memFor("old", types.TypeInsight, "y")
	old.CreatedAt = now.Add(-10000 * time.Hour)
	res := pp.Process([]FusionResult{{MemoryID: "fresh", Score: 0.5}, {MemoryID: "old", Score: 0.5}},
		map[string]*types.Memory{"fresh": fresh, "old": old}, float64(now.Unix())/3600.0)
	if res[0].ID != "fresh" {
		t.Errorf("fresh memory should rank first, got %s", res[0].ID)
	}
}

// --- Superseded filter -----------------------------------------------------

func TestProcessDropsSuperseded(t *testing.T) {
	pp := newPP()
	sup := time.Now()
	live := memFor("live", types.TypeInsight, "a")
	dead := memFor("dead", types.TypeInsight, "b")
	dead.SupersededAt = &sup
	res := pp.Process([]FusionResult{{MemoryID: "live", Score: 0.5}, {MemoryID: "dead", Score: 0.9}},
		map[string]*types.Memory{"live": live, "dead": dead}, float64(time.Now().Unix())/3600.0)
	if len(res) != 1 || res[0].ID != "live" {
		t.Fatalf("superseded memory must be dropped, got %+v", res)
	}
}

func TestProcessEmpty(t *testing.T) {
	pp := newPP()
	if res := pp.Process(nil, map[string]*types.Memory{}, 0); len(res) != 0 {
		t.Fatalf("empty fused input should yield no results, got %d", len(res))
	}
}

// --- Pipeline: both retrieval legs contribute ------------------------------

type hybridStore struct {
	fts    []string
	vector []string
	mems   map[string]*types.Memory
}

func (h *hybridStore) FTS5Search(string, int, string) ([]string, error) { return h.fts, nil }
func (h *hybridStore) FTS5SearchAt(string, store.TimeQuery) ([]string, error) {
	return h.fts, nil
}
func (h *hybridStore) VectorSearch([]float32, int, string) ([]string, error) { return h.vector, nil }
func (h *hybridStore) VectorSearchAt([]float32, store.TimeQuery) ([]string, error) {
	return h.vector, nil
}
func (h *hybridStore) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	out := make([]*types.Memory, 0, len(ids))
	for _, id := range ids {
		if m, ok := h.mems[id]; ok {
			out = append(out, m)
		}
	}
	return out, nil
}
func (h *hybridStore) IncrementAccessCount(string) error       { return nil }
func (h *hybridStore) ListNamespaces() (map[string]int, error) { return nil, nil }
func (h *hybridStore) ArchiveSuperseded(string) (int, error)   { return 0, nil }
func (h *hybridStore) TraverseGraph([]string, int, int) ([]graph.TraversalResult, error) {
	return nil, nil
}
func (h *hybridStore) ListEntityNames(int) ([]string, error) { return nil, nil }

// A memory found only by the keyword leg (ftsOnly) and one found only by the
// vector leg (vecOnly) must both appear; the one both legs agree on wins.
func TestPipelineFusesBothLegs(t *testing.T) {
	now := time.Now()
	st := &hybridStore{
		fts:    []string{"both", "ftsOnly"},
		vector: []string{"both", "vecOnly"},
		mems: map[string]*types.Memory{
			"both":    {ID: "both", Content: "both", Type: types.TypeRule, CreatedAt: now},
			"ftsOnly": {ID: "ftsOnly", Content: "fts", Type: types.TypeInsight, CreatedAt: now},
			"vecOnly": {ID: "vecOnly", Content: "vec", Type: types.TypeInsight, CreatedAt: now},
		},
	}
	p := NewPipeline(st, embedder.NewPriorityChain(fakeEmbedder{}), types.DefaultConfig().Search)
	res, err := p.Search(types.StoreQuery{QueryText: "q", Limit: 10})
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, r := range res {
		seen[r.ID] = true
	}
	if !seen["both"] || !seen["ftsOnly"] || !seen["vecOnly"] {
		t.Fatalf("expected all three legs' results, got %v", seen)
	}
	if res[0].ID != "both" {
		t.Errorf("memory both legs agree on should rank first, got %s", res[0].ID)
	}
}

func TestLengthFactorBounds(t *testing.T) {
	if got := lengthFactor(100); got != 1.0 {
		t.Errorf("short content factor = %v, want 1.0", got)
	}
	if got := lengthFactor(10000); got <= 0 || got >= 1 {
		t.Errorf("long content factor = %v, want in (0,1)", got)
	}
	if lengthFactor(1000) <= lengthFactor(5000) {
		t.Error("factor must decrease with length")
	}
}

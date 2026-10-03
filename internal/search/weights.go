package search

import (
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// Retrieval-v2 tuning. Every number lives here (or in types.SearchConfig) rather
// than being sprinkled through the ranking code, and each can be overridden at
// runtime through the documented NYAWA_RECALL_* environment variables so the
// "before vs after" comparison needs no rebuild.
const (
	DefaultRecencyWeight    = 0.05
	DefaultImportanceWeight = 0.10
	DefaultTypeWeight       = 0.12
	DefaultAccessWeight     = 0.05
	DefaultLengthWeight     = 0.05
	DefaultPinnedWeight     = 0.10
	DefaultGraphWeight      = 0.03

	// Modality weights for the RRF fusion. The keyword (FTS5/BM25) leg is trusted
	// more than the vector leg because vector similarity is computed over the
	// whole memory text — long documents dominate it and exact keywords are
	// missed — and because part of the corpus has no HNSW vector at all. The
	// vector leg still contributes at this weight, so a purely semantic match
	// (no keyword overlap) can still be retrieved.
	DefaultVectorWeight = 1.0
	DefaultFTSWeight    = 3.0

	// OverlapWeight scales the boost a memory gets when it is reachable both by
	// fused ranking and by entity-graph traversal. Kept small: the original
	// hard-coded 1.5x multiplicative boost was burying precise single-list
	// matches.
	DefaultOverlapWeight = 0.1
	// DefaultGraphInjectWeight is the weight given to graph-only memories.
	DefaultGraphInjectWeight = 0.1

	// RRF k damping. 5 (not the textbook 60) because at k=60 the gap between
	// adjacent ranks is tiny, so a memory that is mediocre in both lists beats a
	// memory that is the best hit in one list — exactly the dilution this change
	// fixes. Small k sharpens the fused ranking toward the true top matches.
	DefaultRRFK = 5

	// accessSaturation is the access_count at which the access bonus saturates.
	accessSaturation = 10.0
	// lengthRefBytes: content at or below this size gets the full length bonus;
	// longer content decays logarithmically (see lengthFactor).
	lengthRefBytes = 400.0
)

// Weights bundles every retrieval-ranking coefficient. Zero-valued fields are
// replaced by the defaults above, so a caller may set only the fields it cares
// about (this keeps existing types.SearchConfig literals working).
type Weights struct {
	Recency    float64
	Importance float64
	Type       float64
	Access     float64
	Length     float64
	Pinned     float64
	Graph      float64
	// Fusion modality weights.
	VectorWeight float64
	FTSWeight    float64
	// OverlapWeight boosts memories found by both fusion and graph traversal.
	OverlapWeight float64
	// GraphInjectWeight weights graph-only memories.
	GraphInjectWeight float64
	RRFK              int
	// Legacy disables the v2 relevance boosts and restores the pre-v2 formula.
	Legacy bool
}

func withDefault(v, def float64) float64 {
	if v <= 0 {
		return def
	}
	return v
}

// WeightsFromConfig builds the effective weights: config values first, then
// environment overrides, then the env toggle. It is the single place where the
// ranking numbers are resolved.
func WeightsFromConfig(cfg types.SearchConfig) Weights {
	w := Weights{
		Recency:    withDefault(cfg.RecencyWeight, DefaultRecencyWeight),
		Importance: withDefault(cfg.ImportanceWeight, DefaultImportanceWeight),
		Type:       withDefault(cfg.TypeWeight, DefaultTypeWeight),
		Access:     withDefault(cfg.AccessWeight, DefaultAccessWeight),
		Length:     withDefault(cfg.LengthWeight, DefaultLengthWeight),
		Pinned:     withDefault(cfg.PinnedWeight, DefaultPinnedWeight),
		Graph:      withDefault(cfg.GraphWeight, DefaultGraphWeight),
		RRFK:       cfg.RRFK,
	}
	if w.RRFK <= 0 {
		w.RRFK = cfg.RRFDefaultK
	}
	if w.RRFK <= 0 {
		w.RRFK = DefaultRRFK
	}
	w.VectorWeight = DefaultVectorWeight
	w.FTSWeight = DefaultFTSWeight
	w.OverlapWeight = DefaultOverlapWeight
	w.GraphInjectWeight = DefaultGraphInjectWeight
	w.Legacy = !RecallV2Enabled()

	// Environment overrides (documented in README). Unknown/unparseable values
	// are ignored so a typo never silently zeroes a weight.
	if v, ok := envFloat("NYAWA_RECALL_W_RECENCY"); ok {
		w.Recency = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_IMPORTANCE"); ok {
		w.Importance = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_TYPE"); ok {
		w.Type = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_ACCESS"); ok {
		w.Access = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_LENGTH"); ok {
		w.Length = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_PINNED"); ok {
		w.Pinned = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_GRAPH"); ok {
		w.Graph = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_VECTOR"); ok {
		w.VectorWeight = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_FTS"); ok {
		w.FTSWeight = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_OVERLAP"); ok {
		w.OverlapWeight = v
	}
	if v, ok := envFloat("NYAWA_RECALL_W_GRAPH_INJECT"); ok {
		w.GraphInjectWeight = v
	}
	if v, ok := envInt("NYAWA_RECALL_RRF_K"); ok && v > 0 {
		w.RRFK = v
	}
	if w.Legacy {
		// Restore the pre-v2 fusion so NYAWA_RECALL_V2=0 reproduces the old
		// ranking (symmetric RRF at k=60, hard-coded 1.5x overlap boost, old
		// d-boosts).
		w.VectorWeight, w.FTSWeight = 1, 1
		w.OverlapWeight = 0.5 // 1 + 0.5 = the original 1.5x overlap multiplier
		w.GraphInjectWeight = 0.1
		w.Graph = DefaultGraphWeight
		w.Pinned = DefaultPinnedWeight
		w.RRFK = 60
	}
	return w
}

// RecallV2Enabled reports whether the hybrid/weighted ranking is on. It defaults
// on; only an explicit 0/false/off/no turns it off, which lets operators compare
// pre- and post-change behaviour with a single env var.
func RecallV2Enabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("NYAWA_RECALL_V2"))) {
	case "0", "false", "off", "no":
		return false
	default:
		return true
	}
}

func envFloat(name string) (float64, bool) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func envInt(name string) (int, bool) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0, false
	}
	v, err := strconv.Atoi(raw)
	if err != nil {
		return 0, false
	}
	return v, true
}

// lengthFactor rewards concise memories and gently demotes very long ones: a
// memory at or below lengthRefBytes scores 1.0, longer content decays
// logarithmically. This is what stops a long document from winning purely
// because its embedding happens to look similar — previously score was computed
// over the whole memory text, so length was an unbounded advantage.
// Bounded to (0,1].
func lengthFactor(nBytes int) float64 {
	n := float64(nBytes)
	if n <= lengthRefBytes {
		return 1.0
	}
	return 1.0 / (1.0 + math.Log(n/lengthRefBytes))
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// accessFactor saturates at 1.0 once a memory has been recalled accessSaturation
// times, so a single hot memory cannot run away with the ranking.
func accessFactor(count int) float64 {
	return math.Min(float64(count)/accessSaturation, 1.0)
}

package search

import (
	"math"
	"sort"

	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// RRF implements Reciprocal Rank Fusion over the vector (HNSW) and keyword
// (FTS5/BM25) candidate lists. RRF needs no score calibration between the two
// modalities — it only uses ranks — which is what we want because HNSW cosine
// distances and BM25 scores live on incomparable scales. Both lists therefore
// take part in the final ordering; neither can be ignored.
//
// k (default 60: the value from the original RRF paper and Elasticsearch's
// default) damps the influence of the very top ranks so a single modality
// cannot dominate on one lucky hit.
type RRF struct{ k int }

func NewRRF(k int) *RRF {
	if k <= 0 {
		k = DefaultRRFK
	}
	return &RRF{k: k}
}

type FusionResult struct {
	MemoryID   string
	Score      float64
	VectorRank int
	FTS5Rank   int
}

func (r *RRF) Fuse(vectorIDs, fts5IDs []string) []FusionResult {
	return r.FuseWeighted(vectorIDs, fts5IDs, 1, 1)
}

// FuseWeighted is Fuse with per-modality weights. vectorWeight/ftsWeight let a
// deployment trust one leg more without discarding the other; both default to 1
// (symmetric RRF). Values <= 0 fall back to 1 so a misconfiguration cannot
// silently drop a whole modality.
func (r *RRF) FuseWeighted(vectorIDs, fts5IDs []string, vectorWeight, ftsWeight float64) []FusionResult {
	if vectorWeight <= 0 {
		vectorWeight = 1
	}
	if ftsWeight <= 0 {
		ftsWeight = 1
	}
	seen := make(map[string]*FusionResult, len(vectorIDs)+len(fts5IDs))
	for rank, id := range vectorIDs {
		if _, ok := seen[id]; !ok {
			seen[id] = &FusionResult{MemoryID: id, VectorRank: rank + 1, FTS5Rank: math.MaxInt32}
		} else {
			seen[id].VectorRank = rank + 1
		}
	}
	for rank, id := range fts5IDs {
		if _, ok := seen[id]; !ok {
			seen[id] = &FusionResult{MemoryID: id, VectorRank: math.MaxInt32, FTS5Rank: rank + 1}
		} else {
			seen[id].FTS5Rank = rank + 1
		}
	}
	results := make([]FusionResult, 0, len(seen))
	// Normalise the fused score to [0,1]: (vectorWeight+ftsWeight)/(k+1) is the
	// maximum possible score (rank 1 in both lists). Without normalisation the
	// raw RRF score is far smaller than any recency/importance/type boost added
	// later, so a weakly-relevant but "important" memory outranks the true
	// nearest neighbour.
	norm := (vectorWeight + ftsWeight) / float64(r.k+1)
	for _, fr := range seen {
		rrfScore := 0.0
		if fr.VectorRank < math.MaxInt32 {
			rrfScore += vectorWeight / float64(r.k+fr.VectorRank)
		}
		if fr.FTS5Rank < math.MaxInt32 {
			rrfScore += ftsWeight / float64(r.k+fr.FTS5Rank)
		}
		fr.Score = rrfScore / norm
		results = append(results, *fr)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	return results
}

type PostProcessor struct {
	RecencyWeight    float64
	ImportanceWeight float64
	W                Weights
	resultPool       *pool.ResultPool
}

// NewPostProcessor keeps the original signature (recency, importance, pool) for
// backward compatibility; the remaining v2 weights take their defaults.
func NewPostProcessor(recencyWeight, importanceWeight float64, rp *pool.ResultPool) *PostProcessor {
	if recencyWeight <= 0 {
		recencyWeight = DefaultRecencyWeight
	}
	if importanceWeight <= 0 {
		importanceWeight = DefaultImportanceWeight
	}
	w := Weights{
		Recency: recencyWeight, Importance: importanceWeight,
		Type: DefaultTypeWeight, Access: DefaultAccessWeight, Length: DefaultLengthWeight,
		Pinned: DefaultPinnedWeight, Graph: DefaultGraphWeight, RRFK: DefaultRRFK,
	}
	return &PostProcessor{RecencyWeight: recencyWeight, ImportanceWeight: importanceWeight, W: w, resultPool: rp}
}

// NewPostProcessorWithWeights is the v2 constructor used by the pipeline.
func NewPostProcessorWithWeights(w Weights, rp *pool.ResultPool) *PostProcessor {
	w = fillWeightDefaults(w)
	return &PostProcessor{RecencyWeight: w.Recency, ImportanceWeight: w.Importance, W: w, resultPool: rp}
}

func fillWeightDefaults(w Weights) Weights {
	w.Recency = withDefault(w.Recency, DefaultRecencyWeight)
	w.Importance = withDefault(w.Importance, DefaultImportanceWeight)
	w.Type = withDefault(w.Type, DefaultTypeWeight)
	w.Access = withDefault(w.Access, DefaultAccessWeight)
	w.Length = withDefault(w.Length, DefaultLengthWeight)
	w.Pinned = withDefault(w.Pinned, DefaultPinnedWeight)
	w.Graph = withDefault(w.Graph, DefaultGraphWeight)
	w.VectorWeight = withDefault(w.VectorWeight, DefaultVectorWeight)
	w.FTSWeight = withDefault(w.FTSWeight, DefaultFTSWeight)
	// Overlap/inject may legitimately be zero (to disable the graph signals), so
	// only a negative value is treated as unset.
	if w.OverlapWeight < 0 {
		w.OverlapWeight = DefaultOverlapWeight
	}
	if w.GraphInjectWeight < 0 {
		w.GraphInjectWeight = DefaultGraphInjectWeight
	}
	if w.RRFK <= 0 {
		w.RRFK = DefaultRRFK
	}
	return w
}

func (pp *PostProcessor) Process(fused []FusionResult, memories map[string]*types.Memory, now float64) []*types.MemoryResult {
	results := make([]*types.MemoryResult, 0, len(fused))
	for i, fr := range fused {
		mem, ok := memories[fr.MemoryID]
		if !ok {
			continue
		}
		// Superseded memories must never surface, even if a future caller hands
		// us an unfiltered memory map. The SQL layer already filters them; this
		// is belt-and-braces for the fused ranking.
		if mem.SupersededAt != nil {
			continue
		}
		r := pp.resultPool.Get()
		r.Memory = *mem
		r.RRFScore = fr.Score
		r.Rank = i + 1
		ageHours := now - float64(mem.CreatedAt.Unix())/3600.0
		if ageHours < 0 {
			ageHours = 0
		}
		tau := mem.Type.DecayHours()
		if tau <= 0 {
			tau = 168
		}
		r.TemporalBoost = pp.W.Recency * math.Exp(-ageHours/tau)

		pinBoost := 0.0
		if mem.Pinned {
			pinBoost = pp.W.Pinned
		}
		graphBoost := math.Log1p(float64(mem.EdgeCount)) * pp.W.Graph

		if pp.W.Legacy {
			// Pre-v2 formula, preserved for A/B comparison via NYAWA_RECALL_V2=0.
			r.ImportanceBoost = pp.W.Importance * mem.Type.Weight() * accessFactor(mem.AccessCount)
			r.Score = fr.Score + r.TemporalBoost + r.ImportanceBoost + pinBoost + graphBoost
		} else {
			r.ImportanceBoost = pp.W.Importance * clamp01(mem.Importance)
			r.TypeBoost = pp.W.Type * mem.Type.RetrievalFactor()
			r.AccessBoost = pp.W.Access * accessFactor(mem.AccessCount)
			r.LengthBoost = pp.W.Length * lengthFactor(len(mem.Content))
			r.Score = fr.Score + r.TemporalBoost + r.ImportanceBoost + r.TypeBoost +
				r.AccessBoost + r.LengthBoost + pinBoost + graphBoost
		}
		results = append(results, r)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	for i := range results {
		results[i].Rank = i + 1
	}
	return results
}

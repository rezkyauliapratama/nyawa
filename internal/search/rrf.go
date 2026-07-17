package search

import (
	"math"
	"sort"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type RRF struct{ k int }

func NewRRF(k int) *RRF {
	if k <= 0 {
		k = 60
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
	seen := make(map[string]*FusionResult)
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
	for _, fr := range seen {
		rrfScore := 0.0
		if fr.VectorRank < math.MaxInt32 {
			rrfScore += 1.0 / float64(r.k+fr.VectorRank)
		}
		if fr.FTS5Rank < math.MaxInt32 {
			rrfScore += 1.0 / float64(r.k+fr.FTS5Rank)
		}
		fr.Score = rrfScore
		results = append(results, *fr)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	return results
}

type PostProcessor struct {
	RecencyWeight    float64
	ImportanceWeight float64
	resultPool       *pool.ResultPool
}

func NewPostProcessor(rw, iw float64, rp *pool.ResultPool) *PostProcessor {
	if rw <= 0 {
		rw = 0.05
	}
	if iw <= 0 {
		iw = 0.10
	}
	return &PostProcessor{RecencyWeight: rw, ImportanceWeight: iw, resultPool: rp}
}

func (pp *PostProcessor) Process(fused []FusionResult, memories map[string]*types.Memory, now float64) []*types.MemoryResult {
	results := make([]*types.MemoryResult, 0, len(fused))
	for i, fr := range fused {
		mem, ok := memories[fr.MemoryID]
		if !ok {
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
		r.TemporalBoost = pp.RecencyWeight * math.Exp(-ageHours/mem.Type.DecayHours())

		accessFactor := math.Min(float64(mem.AccessCount)/10.0, 1.0)
		r.ImportanceBoost = pp.ImportanceWeight * mem.Type.Weight() * accessFactor

		pinBoost := 0.0
		if mem.Pinned {
			pinBoost = 0.1
		}
		graphBoost := math.Log1p(float64(mem.EdgeCount)) * 0.03

		r.Score = fr.Score + r.TemporalBoost + r.ImportanceBoost + pinBoost + graphBoost
		results = append(results, r)
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	for i := range results {
		results[i].Rank = i + 1
	}
	return results
}

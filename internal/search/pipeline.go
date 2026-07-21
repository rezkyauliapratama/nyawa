package search

import (
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Pipeline struct {
	store       StoreReader
	embedder    *embedder.PriorityChain
	rrf         *RRF
	post        *PostProcessor
	cache       *Cache
	resultPool  *pool.ResultPool
	slicePool   *pool.ResultSlicePool
	queryPool   sync.Pool
}

type StoreReader interface {
	FTS5Search(query string, topK int, namespace string) ([]string, error)
	FTS5SearchAt(query string, tq store.TimeQuery) ([]string, error)
	VectorSearch(queryVector []float32, topK int, namespace string) ([]string, error)
	VectorSearchAt(query []float32, tq store.TimeQuery) ([]string, error)
	GetMemoriesByIDs(ids []string) ([]*types.Memory, error)
	IncrementAccessCount(id string) error
	ListNamespaces() (map[string]int, error)
	ArchiveSuperseded(archivePath string) (int, error)
}

func NewPipeline(store StoreReader, emb *embedder.PriorityChain, cfg types.SearchConfig) *Pipeline {
	rp := pool.NewResultPool(64)
	return &Pipeline{
		store: store, embedder: emb, rrf: NewRRF(cfg.RRFK),
		post: NewPostProcessor(cfg.RecencyWeight, cfg.ImportanceWeight, rp),
		cache: NewCache(256, 5*time.Minute), resultPool: rp, slicePool: pool.NewResultSlicePool(),
	}
}

func (p *Pipeline) Search(q types.StoreQuery) ([]*types.MemoryResult, error) {
	if cached, ok := p.cache.Get(q.QueryText); ok { return cached, nil }
	queryVec, err := p.embedder.Embed(q.QueryText)
	haveVector := err == nil
	if !haveVector { log.Printf("embedder unavailable, falling back to FTS5-only: %v", err) }
	var vectorIDs, fts5IDs []string
	var errVec, errFTS5 error
	var wg sync.WaitGroup
	limit := q.Limit
	if limit <= 0 { limit = types.DefaultQueryLimit }
	searchTopK := forceMin(limit*3, 50)
	wg.Add(1)
	go func() { defer wg.Done(); fts5IDs, errFTS5 = p.store.FTS5Search(q.QueryText, searchTopK, q.Namespace) }()
	if haveVector { wg.Add(1); go func() { defer wg.Done(); vectorIDs, errVec = p.store.VectorSearch(queryVec, searchTopK, q.Namespace) }() }
	wg.Wait()
	if errVec != nil { return nil, fmt.Errorf("vector: %w", errVec) }
	if errFTS5 != nil { return nil, fmt.Errorf("fts5: %w", errFTS5) }
	fused := p.rrf.Fuse(vectorIDs, fts5IDs)
	allIDs := make([]string, len(fused))
	for i, fr := range fused { allIDs[i] = fr.MemoryID }
	memories, err := p.store.GetMemoriesByIDs(allIDs)
	if err != nil { return nil, fmt.Errorf("get memories: %w", err) }
	memMap := make(map[string]*types.Memory, len(memories))
	for _, m := range memories { memMap[m.ID] = m }
	now := float64(time.Now().Unix()) / 3600.0
	results := p.post.Process(fused, memMap, now)
	if q.TimeTravel != nil { results = p.filterByTime(results, *q.TimeTravel) }
	if q.MinScore > 0 {
		filtered := results[:0]
		for _, r := range results { if r.Score >= q.MinScore { filtered = append(filtered, r) } }
		results = filtered
	}
	p.cache.Set(q.QueryText, results)
	go func() { for _, r := range results { _ = p.store.IncrementAccessCount(r.ID) } }()
	return results, nil
}

func (p *Pipeline) filterByTime(results []*types.MemoryResult, targetTime time.Time) []*types.MemoryResult {
	filtered := results[:0]
	for _, r := range results {
		if r.CreatedAt.After(targetTime) { continue }
		if r.SupersededAt != nil && r.SupersededAt.Before(targetTime) { continue }
		filtered = append(filtered, r)
	}
	return filtered
}

func (p *Pipeline) ReleaseResults(results []*types.MemoryResult) {
	for _, r := range results { p.resultPool.Put(r) }
}

func forceMin(v, min int) int { if v < min { return min }; return v }

type cacheEntry struct{ results []*types.MemoryResult; expiresAt time.Time }
type Cache struct{ mu sync.RWMutex; items map[string]*cacheEntry; cap int; ttl time.Duration }
func NewCache(capacity int, ttl time.Duration) *Cache {
	if capacity <= 0 { capacity = 256 }
	if ttl <= 0 { ttl = 5 * time.Minute }
	return &Cache{items: make(map[string]*cacheEntry, capacity), cap: capacity, ttl: ttl}
}
func (c *Cache) Get(key string) ([]*types.MemoryResult, bool) {
	c.mu.RLock(); entry, ok := c.items[key]; c.mu.RUnlock()
	if !ok { return nil, false }
	if time.Now().After(entry.expiresAt) { c.mu.Lock(); delete(c.items, key); c.mu.Unlock(); return nil, false }
	return entry.results, true
}
func (c *Cache) Set(key string, results []*types.MemoryResult) {
	c.mu.Lock(); defer c.mu.Unlock()
	if len(c.items) >= c.cap { for k := range c.items { delete(c.items, k); break } }
	c.items[key] = &cacheEntry{results: results, expiresAt: time.Now().Add(c.ttl)}
}

const K = 60
func BatchCosineCompare(query []float32, candidates [][]float32) []float64 {
	scores := make([]float64, len(candidates))
	for i, vec := range candidates { scores[i] = cosineSimilarity(query, vec) }
	return scores
}
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 { return 0 }
	var dot, normA, normB float64
	for i := range a { ai, bi := float64(a[i]), float64(b[i]); dot += ai * bi; normA += ai * ai; normB += bi * bi }
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom < 1e-9 { return 0 }
	return dot / denom
}

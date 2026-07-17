package search

import (
	"fmt"
	"math"
	"sync"
	"time"
	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/pool"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type StoreReader interface {
	FTS5Search(query string, topK int, namespace string) ([]string, error)
	VectorSearch(queryVector []float32, topK int, namespace string) ([]string, error)
	GetMemoriesByIDs(ids []string) ([]*types.Memory, error)
	IncrementAccessCount(id string) error
}

type Pipeline struct {
	store      StoreReader
	embedder   *embedder.PriorityChain
	rrf        *RRF
	post       *PostProcessor
	cache      *Cache
	resultPool *pool.ResultPool
}

func NewPipeline(store StoreReader, emb *embedder.PriorityChain, cfg types.SearchConfig) *Pipeline {
	return &Pipeline{
		store: store, embedder: emb,
		rrf: NewRRF(cfg.RRFK),
		post: NewPostProcessor(cfg.RecencyWeight, cfg.ImportanceWeight, pool.NewResultPool(64)),
		cache: NewCache(256, 5*time.Minute),
		resultPool: pool.NewResultPool(64),
	}
}

func (p *Pipeline) Search(q types.StoreQuery) ([]*types.MemoryResult, error) {
	if cached, ok := p.cache.Get(q.QueryText); ok {
		return cached, nil
	}
	queryVec, err := p.embedder.Embed(q.QueryText)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}
	var (
		vectorIDs, fts5IDs []string
		wg sync.WaitGroup
		errVec, errFTS5 error
	)
	limit := q.Limit
	if limit <= 0 {
		limit = types.DefaultQueryLimit
	}
	wg.Add(2)
	go func() {
		defer wg.Done()
		vectorIDs, errVec = p.store.VectorSearch(queryVec, forceMin(limit*3, 50), q.Namespace)
	}()
	go func() {
		defer wg.Done()
		fts5IDs, errFTS5 = p.store.FTS5Search(q.QueryText, forceMin(limit*3, 50), q.Namespace)
	}()
	wg.Wait()
	if errFTS5 != nil {
		return nil, errFTS5
	}
	fused := p.rrf.Fuse(vectorIDs, fts5IDs)
	allIDs := make([]string, 0, len(fused))
	for _, fr := range fused {
		allIDs = append(allIDs, fr.MemoryID)
	}
	memories, err := p.store.GetMemoriesByIDs(allIDs)
	if err != nil {
		return nil, err
	}
	memMap := make(map[string]*types.Memory, len(memories))
	for _, m := range memories {
		memMap[m.ID] = m
	}
	results := p.post.Process(fused, memMap, float64(time.Now().Unix())/3600.0)
	p.cache.Set(q.QueryText, results)
	return results, nil
}

func (p *Pipeline) ReleaseResults(results []*types.MemoryResult) {
	for _, r := range results {
		p.resultPool.Put(r)
	}
}

func forceMin(v, min int) int {
	if v < min {
		return min
	}
	return v
}

type cacheEntry struct {
	results   []*types.MemoryResult
	expiresAt time.Time
}

type Cache struct {
	mu    sync.RWMutex
	items map[string]*cacheEntry
}

func NewCache(capacity int, ttl time.Duration) *Cache {
	return &Cache{items: make(map[string]*cacheEntry, capacity)}
}

func (c *Cache) Get(key string) ([]*types.MemoryResult, bool) {
	c.mu.RLock()
	entry, ok := c.items[key]
	c.mu.RUnlock()
	if !ok || time.Now().After(entry.expiresAt) {
		return nil, false
	}
	return entry.results, true
}

func (c *Cache) Set(key string, results []*types.MemoryResult) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[key] = &cacheEntry{results: results, expiresAt: time.Now().Add(5 * time.Minute)}
}

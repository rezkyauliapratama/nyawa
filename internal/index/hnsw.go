package index

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
)

type HNSWConfig struct{ M, Mmax, EfConstruction, EfSearch int; ML float64; Dim int }

func DefaultHNSWConfig(dim int) HNSWConfig {
	m := 16
	return HNSWConfig{M: m, Mmax: m, EfConstruction: 200, EfSearch: 50, ML: 1.0 / math.Log(float64(m)), Dim: dim}
}

type Node struct{ ID string; Vec []float32; Level int }

type HNSW struct {
	mu          sync.RWMutex
	entryPoint  string
	maxLevel    int
	config      HNSWConfig
	nodes       map[string]*Node
	graph       []map[string]map[string]float64
	rng         *rand.Rand
}

func NewHNSW(config HNSWConfig) *HNSW {
	return &HNSW{
		config: config,
		nodes:  make(map[string]*Node),
		graph:  []map[string]map[string]float64{{}},
		rng:    rand.New(rand.NewSource(42)),
	}
}

func (h *HNSW) distance(a, b []float32) float64 {
	dot, n1, n2 := float64(0), float64(0), float64(0)
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		n1 += float64(a[i]) * float64(a[i])
		n2 += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(n1) * math.Sqrt(n2)
	if denom == 0 { return 1 }
	return 1 - dot/denom
}

func (h *HNSW) Insert(id string, vec []float32) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, ok := h.nodes[id]; ok { h.nodes[id].Vec = vec; return }
	level := int(math.Floor(-math.Log(h.rng.Float64()) * h.config.ML))
	h.nodes[id] = &Node{ID: id, Vec: vec, Level: level}
	for len(h.graph) <= level { h.graph = append(h.graph, make(map[string]map[string]float64)) }
	if h.entryPoint == "" { h.entryPoint = id; h.maxLevel = level; return }
	curr := h.entryPoint
	for l := h.maxLevel; l > level; l-- { curr = h.searchLayer(vec, curr, 1, l)[0] }
	for l := level; l >= 0; l-- {
		candidates := h.searchLayer(vec, curr, h.config.EfConstruction, l)
		neighbors := candidates
		if len(neighbors) > h.config.M { neighbors = neighbors[:h.config.M] }
		for _, nID := range neighbors {
			if nID == id { continue }
			if h.graph[l][nID] == nil { h.graph[l][nID] = make(map[string]float64) }
			h.graph[l][nID][id] = h.distance(vec, h.nodes[nID].Vec)
			if h.graph[l][id] == nil { h.graph[l][id] = make(map[string]float64) }
			h.graph[l][id][nID] = h.distance(vec, h.nodes[nID].Vec)
		}
		if len(neighbors) > 0 {
			h.pruneNeighbors(l, id)
			for _, nID := range neighbors { h.pruneNeighbors(l, nID) }
		}
		curr = candidates[0]
	}
	if level > h.maxLevel { h.entryPoint = id; h.maxLevel = level }
}

func (h *HNSW) Search(query []float32, topK int) []SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entryPoint == "" || len(h.nodes) == 0 { return nil }
	ef := h.config.EfSearch
	if topK > ef { ef = topK * 2 }
	curr := h.entryPoint
	for l := h.maxLevel; l > 0; l-- { curr = h.searchLayer(query, curr, 1, l)[0] }
	candidates := h.searchLayer(query, curr, ef, 0)
	results := make([]SearchResult, 0, len(candidates))
	for _, id := range candidates { results = append(results, SearchResult{ID: id, Distance: h.distance(query, h.nodes[id].Vec)}) }
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Distance < results[i].Distance { results[i], results[j] = results[j], results[i] }
		}
	}
	if len(results) > topK { results = results[:topK] }
	return results
}

type candidate struct{ id string; dist float64 }
type minHeap []candidate
func (h minHeap) len() int                                    { return len(h) }
func (h minHeap) isEmpty() bool                                { return len(h) == 0 }
func (h minHeap) peek() candidate                              { return h[0] }
func (h *minHeap) push(c candidate)                           { *h = append(*h, c) }
func (h *minHeap) pop() candidate {
	smallest := 0
	for i := 1; i < len(*h); i++ { if (*h)[i].dist < (*h)[smallest].dist { smallest = i } }
	c := (*h)[smallest]
	*h = append((*h)[:smallest], (*h)[smallest+1:]...)
	return c
}
type maxHeap []candidate
func (h maxHeap) len() int                                    { return len(h) }
func (h maxHeap) isEmpty() bool                                { return len(h) == 0 }
func (h maxHeap) peek() candidate                              { return h[0] }
func (h *maxHeap) push(c candidate)                           { *h = append(*h, c) }
func (h *maxHeap) pop() candidate {
	largest := 0
	for i := 1; i < len(*h); i++ { if (*h)[i].dist > (*h)[largest].dist { largest = i } }
	c := (*h)[largest]
	*h = append((*h)[:largest], (*h)[largest+1:]...)
	return c
}

func (h *HNSW) searchLayer(q []float32, entry string, ef, layer int) []string {
	visited := make(map[string]bool)
	candidates, results := newMinHeap(), newMaxHeap()
	dist := h.distance(q, h.nodes[entry].Vec)
	candidates.push(candidate{id: entry, dist: dist})
	results.push(candidate{id: entry, dist: dist})
	visited[entry] = true
	for !candidates.isEmpty() {
		closest := candidates.pop()
		if results.len() >= ef && closest.dist > results.peek().dist { break }
		for neighbor := range h.graph[layer][closest.id] {
			if visited[neighbor] { continue }
			visited[neighbor] = true
			ndist := h.distance(q, h.nodes[neighbor].Vec)
			candidates.push(candidate{id: neighbor, dist: ndist})
			if results.len() < ef || ndist < results.peek().dist {
				results.push(candidate{id: neighbor, dist: ndist})
				if results.len() > ef { results.pop() }
			}
		}
	}
	resultIDs := make([]string, results.len())
	for i := len(resultIDs) - 1; i >= 0; i-- { resultIDs[i] = results.pop().id }
	return resultIDs
}

func (h *HNSW) pruneNeighbors(layer int, nodeID string) {
	neighbors := h.graph[layer][nodeID]
	if len(neighbors) <= h.config.Mmax { return }
	type pair struct{ id string; dist float64 }
	sorted := make([]pair, 0, len(neighbors))
	for id, dist := range neighbors { sorted = append(sorted, pair{id, dist}) }
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].dist < sorted[i].dist { sorted[i], sorted[j] = sorted[j], sorted[i] }
		}
	}
	h.graph[layer][nodeID] = make(map[string]float64)
	for i := 0; i < h.config.Mmax && i < len(sorted); i++ {
		h.graph[layer][nodeID][sorted[i].id] = sorted[i].dist
	}
}

type SearchResult struct{ ID string; Distance float64 }

func (h *HNSW) Size() int { h.mu.RLock(); defer h.mu.RUnlock(); return len(h.nodes) }

func (h *HNSW) Save(path string) error {
	h.mu.RLock(); defer h.mu.RUnlock()
	b, err := json.Marshal(struct {
		EntryPoint string                         `json:"ep"`
		MaxLevel   int                            `json:"ml"`
		Config     HNSWConfig                     `json:"c"`
		Nodes      map[string]*Node               `json:"ns"`
		Graph      []map[string]map[string]float64 `json:"g"`
	}{h.entryPoint, h.maxLevel, h.config, h.nodes, h.graph})
	if err != nil { return err }
	return os.WriteFile(path, b, 0644)
}

func (h *HNSW) Load(path string) error {
	h.mu.Lock(); defer h.mu.Unlock()
	b, err := os.ReadFile(path)
	if err != nil { return err }
	var data struct {
		EntryPoint string                         `json:"ep"`
		MaxLevel   int                            `json:"ml"`
		Config     HNSWConfig                     `json:"c"`
		Nodes      map[string]*Node               `json:"ns"`
		Graph      []map[string]map[string]float64 `json:"g"`
	}
	if err := json.Unmarshal(b, &data); err != nil { return err }
	h.entryPoint = data.EntryPoint; h.maxLevel = data.MaxLevel
	// Only restore config from file if it has valid values (not all zeros)
	if data.Config.M > 0 && data.Config.EfConstruction > 0 && data.Config.EfSearch > 0 {
		h.config = data.Config
	} else {
		// Keep current (DefaultHNSWConfig) config — don't overwrite with zeros
		h.config.M = max(h.config.M, 16)
		h.config.Mmax = max(h.config.Mmax, h.config.M)
		if h.config.EfConstruction == 0 { h.config.EfConstruction = 200 }
		if h.config.EfSearch == 0 { h.config.EfSearch = 50 }
		if h.config.ML == 0 { h.config.ML = 1.0 / math.Log(float64(h.config.M)) }
	}
	h.nodes = data.Nodes; h.graph = data.Graph
	h.rng = rand.New(rand.NewSource(42))
	return nil
}

func (h *HNSW) GetDB() *sql.DB { return nil }
func (h *HNSW) GetHNSW() *HNSW { return h }
func (h *HNSW) GetHNSWPath() string { return "" }

func newMinHeap() *minHeap { return &minHeap{} }
func newMaxHeap() *maxHeap { return &maxHeap{} }

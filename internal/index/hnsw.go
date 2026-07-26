package index

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sync"
)

type HNSWConfig struct {
	M, Mmax, EfConstruction, EfSearch int
	ML                                float64
	Dim                               int
}

func DefaultHNSWConfig(dim int) HNSWConfig {
	m := 16
	return HNSWConfig{M: m, Mmax: m, EfConstruction: 200, EfSearch: 50, ML: 1.0 / math.Log(float64(m)), Dim: dim}
}

type Node struct{ ID string; Vec []float32; Level int }
type HNSW struct {
	mu         sync.RWMutex
	config     HNSWConfig
	nodes      map[string]*Node
	graph      []map[string]map[string]float64
	entryPoint string
	maxLevel   int
	rng        *rand.Rand
}

func NewHNSW(cfg HNSWConfig) *HNSW {
	graph := make([]map[string]map[string]float64, 1)
	graph[0] = make(map[string]map[string]float64)
	return &HNSW{config: cfg, nodes: make(map[string]*Node), graph: graph, entryPoint: "", maxLevel: -1, rng: rand.New(rand.NewSource(42))}
}

func (h *HNSW) randomLevel() int {
	level := int(math.Floor(-math.Log(h.rng.Float64()) * h.config.ML))
	if level > h.maxLevel+1 { level = h.maxLevel + 1 }
	return level
}

func (h *HNSW) Insert(id string, vec []float32) {
	h.mu.Lock(); defer h.mu.Unlock()
	if _, exists := h.nodes[id]; exists { return }
	level := h.randomLevel()
	h.nodes[id] = &Node{ID: id, Vec: vec, Level: level}
	for len(h.graph) <= level { h.graph = append(h.graph, make(map[string]map[string]float64)) }
	for l := 0; l <= level; l++ { if h.graph[l][id] == nil { h.graph[l][id] = make(map[string]float64) } }
	if h.entryPoint == "" { h.entryPoint = id; h.maxLevel = level; return }
	curr := h.entryPoint
	for l := h.maxLevel; l > level; l-- { curr = h.searchLayer(vec, curr, 1, l)[0] }
	for l := level; l >= 0; l-- {
		ef := h.config.EfConstruction
		if l == 0 { ef = h.config.EfConstruction }
		candidates := h.searchLayer(vec, curr, ef, l)
		neighbors := candidates[:min(len(candidates), h.config.M)]
		for _, nID := range neighbors {
			if nID == id { continue }
			if h.graph[l][nID] == nil { h.graph[l][nID] = make(map[string]float64) }
			h.graph[l][id][nID] = h.distance(vec, h.nodes[nID].Vec)
			h.graph[l][nID][id] = h.distance(h.nodes[nID].Vec, vec)
			if len(h.graph[l][nID]) > h.config.Mmax { h.pruneNeighbors(l, nID) }
		}
		if len(h.graph[l][id]) > h.config.Mmax { h.pruneNeighbors(l, id) }
		curr = candidates[0]
	}
	if level > h.maxLevel { h.entryPoint = id; h.maxLevel = level }
}

type SearchResult struct{ ID string; Distance float64 }

func (h *HNSW) Search(query []float32, topK int) []SearchResult {
	h.mu.RLock(); defer h.mu.RUnlock()
	if h.entryPoint == "" || len(h.nodes) == 0 { return nil }
	ef := h.config.EfSearch
	if topK > ef { ef = topK * 2 }
	curr := h.entryPoint
	for l := h.maxLevel; l > 0; l-- { curr = h.searchLayer(query, curr, 1, l)[0] }
	candidates := h.searchLayer(query, curr, ef, 0)
	results := make([]SearchResult, len(candidates))
	for i, id := range candidates { results[i] = SearchResult{ID: id, Distance: h.distance(query, h.nodes[id].Vec)} }
	h.sortByDistance(results)
	if len(results) > topK { results = results[:topK] }
	return results
}

func (h *HNSW) searchLayer(queryVec []float32, entry string, ef int, layer int) []string {
	visited := make(map[string]bool)
	candidates := &minHeap{}
	results := &maxHeap{}
	dist := h.distance(queryVec, h.nodes[entry].Vec)
	candidates.push(candidate{entry, dist})
	results.push(candidate{entry, dist})
	visited[entry] = true
	for !candidates.isEmpty() {
		closest := candidates.pop()
		if results.len() >= ef && closest.dist > results.peek().dist { break }
		for neighbor := range h.graph[layer][closest.id] {
			if visited[neighbor] { continue }
			visited[neighbor] = true
			ndist := h.distance(queryVec, h.nodes[neighbor].Vec)
			candidates.push(candidate{neighbor, ndist})
			if results.len() < ef || ndist < results.peek().dist {
				results.push(candidate{neighbor, ndist})
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
	pairs := make([]struct{ id string; dist float64 }, 0, len(neighbors))
	for nID, d := range neighbors { pairs = append(pairs, struct{ id string; dist float64 }{nID, d}) }
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ { if pairs[j].dist < pairs[i].dist { pairs[i], pairs[j] = pairs[j], pairs[i] } }
	}
	h.graph[layer][nodeID] = make(map[string]float64)
	for i := 0; i < h.config.Mmax && i < len(pairs); i++ { h.graph[layer][nodeID][pairs[i].id] = pairs[i].dist }
}

func (h *HNSW) distance(a, b []float32) float64 {
	var sum float64; minLen := len(a); if len(b) < minLen { minLen = len(b) }
	for i := 0; i < minLen; i++ { d := float64(a[i]) - float64(b[i]); sum += d * d }
	return math.Sqrt(sum)
}

func (h *HNSW) sortByDistance(results []SearchResult) {
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ { if results[j].Distance < results[i].Distance { results[i], results[j] = results[j], results[i] } }
	}
}

func (h *HNSW) Delete(id string) {
	h.mu.Lock(); defer h.mu.Unlock()
	if _, exists := h.nodes[id]; !exists { return }
	level := h.nodes[id].Level
	delete(h.nodes, id)
	for l := 0; l <= level && l < len(h.graph); l++ {
		delete(h.graph[l], id)
		for nID := range h.graph[l] { delete(h.graph[l][nID], id) }
	}
	if h.entryPoint == id {
		h.entryPoint = ""; h.maxLevel = -1
		for id := range h.nodes { h.entryPoint = id; h.maxLevel = h.nodes[id].Level; break }
	}
}

func (h *HNSW) Size() int { h.mu.RLock(); defer h.mu.RUnlock(); return len(h.nodes) }

type hnswData struct {
	EntryPoint string                             `json:"ep"`
	MaxLevel   int                                `json:"ml"`
	Config     HNSWConfig                         `json:"cfg"`
	Nodes      map[string]*Node                   `json:"ns"`
	Graph      []map[string]map[string]float64     `json:"g"`
}

func (h *HNSW) Save(path string) error {
	h.mu.RLock(); defer h.mu.RUnlock()
	data := hnswData{EntryPoint: h.entryPoint, MaxLevel: h.maxLevel, Config: h.config, Nodes: h.nodes, Graph: h.graph}
	b, err := json.Marshal(data)
	if err != nil { return err }
	return os.WriteFile(path, b, 0644)
}

func (h *HNSW) Load(path string) error {
	h.mu.Lock(); defer h.mu.Unlock()
	b, err := os.ReadFile(path)
	if err != nil { return err }
	var data hnswData
	if err := json.Unmarshal(b, &data); err != nil { return err }
	h.entryPoint, h.maxLevel, h.config, h.nodes, h.graph = data.EntryPoint, data.MaxLevel, data.Config, data.Nodes, data.Graph
	h.rng = rand.New(rand.NewSource(42))
	return nil
}

type candidate struct{ id string; dist float64 }
type minHeap struct{ items []candidate }
func newMinHeap() *minHeap { return &minHeap{} }
func (h *minHeap) push(c candidate) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		p := (i - 1) / 2
		if h.items[i].dist >= h.items[p].dist { break }
		h.items[i], h.items[p] = h.items[p], h.items[i]; i = p
	}
}
func (h *minHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0); return top
}
func (h *minHeap) sink(i int) {
	for {
		smallest := i; l, r := 2*i+1, 2*i+2
		if l < len(h.items) && h.items[l].dist < h.items[smallest].dist { smallest = l }
		if r < len(h.items) && h.items[r].dist < h.items[smallest].dist { smallest = r }
		if smallest == i { break }
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]; i = smallest
	}
}
func (h *minHeap) isEmpty() bool { return len(h.items) == 0 }
func (h *minHeap) len() int      { return len(h.items) }

type maxHeap struct{ items []candidate }
func newMaxHeap() *maxHeap { return &maxHeap{} }
func (h *maxHeap) push(c candidate) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		p := (i - 1) / 2
		if h.items[i].dist <= h.items[p].dist { break }
		h.items[i], h.items[p] = h.items[p], h.items[i]; i = p
	}
}
func (h *maxHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0); return top
}
func (h *maxHeap) peek() candidate { return h.items[0] }
func (h *maxHeap) sink(i int) {
	for {
		largest := i; l, r := 2*i+1, 2*i+2
		if l < len(h.items) && h.items[l].dist > h.items[largest].dist { largest = l }
		if r < len(h.items) && h.items[r].dist > h.items[largest].dist { largest = r }
		if largest == i { break }
		h.items[i], h.items[largest] = h.items[largest], h.items[i]; i = largest
	}
}
func (h *maxHeap) isEmpty() bool { return len(h.items) == 0 }
func (h *maxHeap) len() int      { return len(h.items) }

package index

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
)

var ErrDimensionMismatch = errors.New("hnsw dimension mismatch")

// tmpSeq makes temp file names unique across concurrent saves in one process
// (pid alone is not enough when two goroutines save the same path at once).
var tmpSeq uint64

type HNSWConfig struct {
	M, Mmax, EfConstruction, EfSearch int
	ML                                float64
	Dim                               int
}

func DefaultHNSWConfig(dim int) HNSWConfig {
	m := 16
	return HNSWConfig{M: m, Mmax: m, EfConstruction: 200, EfSearch: 50, ML: 1.0 / math.Log(float64(m)), Dim: dim}
}

type Node struct {
	ID    string
	Vec   []float32
	Level int
}
type HNSW struct {
	mu               sync.RWMutex
	config           HNSWConfig
	nodes            map[string]*Node
	graph            []map[string]map[string]float64
	entryPoint       string
	maxLevel         int
	rng              *rand.Rand
	ragBoundedDegree bool
	ragGraphVersion  int
}

const currentRAGGraphVersion = 1

func NewHNSW(cfg HNSWConfig) *HNSW {
	graph := make([]map[string]map[string]float64, 1)
	graph[0] = make(map[string]map[string]float64)
	return &HNSW{config: cfg, nodes: make(map[string]*Node), graph: graph, entryPoint: "", maxLevel: -1, rng: rand.New(rand.NewSource(42))}
}

// EnableRAGBoundedDegree enables strict Mmax adjacency bounds for the RAG
// index. Memory indexes retain their historical insertion and SearchChecked
// behavior. Legacy normalization is explicitly deferred until the RAG profile
// has been validated by its manifest.
func (h *HNSW) EnableRAGBoundedDegree() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.ragBoundedDegree = true
}

func (h *HNSW) randomLevel() int {
	level := int(math.Floor(-math.Log(h.rng.Float64()) * h.config.ML))
	if level > h.maxLevel+1 {
		level = h.maxLevel + 1
	}
	return level
}

// Insert adds a vector only when its dimensionality matches the index. The
// returned error is deliberate: callers must not mix vector widths in one HNSW
// graph, because older versions silently compared only the shortest prefix.
func (h *HNSW) Insert(id string, vec []float32) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.insertLocked(id, vec)
}

// insertLocked adds a node without taking the write mutex; callers must hold
// h.mu for writing. It validates the vector before changing the graph.
func (h *HNSW) insertLocked(id string, vec []float32) error {
	if err := h.validateVectorLocked(vec); err != nil {
		return err
	}
	if h.config.Dim == 0 {
		h.config.Dim = len(vec)
	}
	if _, exists := h.nodes[id]; exists {
		return nil
	}
	level := h.randomLevel()
	h.nodes[id] = &Node{ID: id, Vec: vec, Level: level}
	for len(h.graph) <= level {
		h.graph = append(h.graph, make(map[string]map[string]float64))
	}
	for l := 0; l <= level; l++ {
		if h.graph[l][id] == nil {
			h.graph[l][id] = make(map[string]float64)
		}
	}
	if h.entryPoint == "" {
		h.entryPoint = id
		h.maxLevel = level
		if h.ragBoundedDegree {
			h.ragGraphVersion = currentRAGGraphVersion
		}
		return nil
	}
	curr := h.greedyDescent(vec, h.entryPoint, h.maxLevel, level)
	for l := level; l >= 0; l-- {
		ef := h.config.EfConstruction
		var candidates []string
		if h.ragBoundedDegree {
			stable := h.searchLayerStable(vec, curr, ef, l)
			candidates = make([]string, len(stable))
			for i, candidate := range stable {
				candidates[i] = candidate.id
			}
		} else {
			candidates = h.searchLayer(vec, curr, ef, l)
		}
		neighborLimit := h.config.M
		if h.ragBoundedDegree && neighborLimit > h.config.Mmax {
			neighborLimit = h.config.Mmax
		}
		if h.ragBoundedDegree {
			// Algorithm 1 delegates both the new-node set and a full neighbor's
			// merged adjacency to SELECT-NEIGHBORS. Keep the RAG path stable and
			// bounded; the legacy memory insertion path below remains unchanged.
			neighbors := h.selectRAGNeighbors(l, id, candidates, neighborLimit)
			h.connectRAGLayer(l, id, neighbors, candidates)
		} else {
			neighbors := candidates[:min(len(candidates), neighborLimit)]
			for _, nID := range neighbors {
				if nID == id {
					continue
				}
				if h.graph[l][nID] == nil {
					h.graph[l][nID] = make(map[string]float64)
				}
				distance := h.distance(vec, h.nodes[nID].Vec)
				h.graph[l][id][nID] = distance
				h.graph[l][nID][id] = distance
				if len(h.graph[l][nID]) > h.config.Mmax {
					h.pruneNeighbors(l, nID)
				}
			}
			nearest := candidates[0]
			if nearest != id {
				if h.graph[l][nearest] == nil {
					h.graph[l][nearest] = make(map[string]float64)
				}
				h.graph[l][nearest][id] = h.distance(h.nodes[nearest].Vec, vec)
			}
			if len(h.graph[l][id]) > h.config.Mmax {
				h.pruneNeighbors(l, id)
			}
		}
		curr = candidates[0]
	}
	if h.ragBoundedDegree {
		h.ragGraphVersion = currentRAGGraphVersion
	}
	if level > h.maxLevel {
		h.entryPoint = id
		h.maxLevel = level
	}
	return nil
}

// greedyDescent chooses one entry point per upper layer. RAG uses the stable
// distance/ID ordering so equal-distance candidates cannot depend on map order;
// the generic memory index retains its historical traversal semantics.
func (h *HNSW) greedyDescent(vec []float32, curr string, fromLevel, toLevel int) string {
	for layer := fromLevel; layer > toLevel; layer-- {
		if h.ragBoundedDegree {
			candidates := h.searchLayerStable(vec, curr, 1, layer)
			if len(candidates) == 0 {
				break
			}
			curr = candidates[0].id
			continue
		}
		candidates := h.searchLayer(vec, curr, 1, layer)
		if len(candidates) == 0 {
			break
		}
		curr = candidates[0]
	}
	return curr
}

// EnsureDimension checks the selected embedder before any memory operation
// writes vectors. Dimensions are a structural check only: changing model or
// provider still requires rebuilding all vectors, even if its output width is
// the same.
func (h *HNSW) EnsureDimension(dim int) error {
	if dim <= 0 {
		return fmt.Errorf("%w: embedder reports %d dimensions", ErrDimensionMismatch, dim)
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	indexDim, err := h.dimensionLocked()
	if err != nil {
		return err
	}
	if len(h.nodes) == 0 {
		// An empty persisted index has no vectors to preserve, so its configured
		// width may safely follow the explicitly selected embedder.
		h.config.Dim = dim
		return nil
	}
	if indexDim != dim {
		return fmt.Errorf("%w: index uses %d dimensions, selected embedder uses %d", ErrDimensionMismatch, indexDim, dim)
	}
	if h.config.Dim == 0 {
		h.config.Dim = indexDim
	}
	return nil
}

// ValidateVector checks a vector without mutating the index. Store uses this
// before inserting a memory row so a bad embedding does not leave a partial
// database write behind.
func (h *HNSW) ValidateVector(vec []float32) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.validateVectorLocked(vec)
}

// Dimension returns the persisted (or inferred legacy) vector width.
func (h *HNSW) Dimension() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	dim, _ := h.dimensionLocked()
	return dim
}

func (h *HNSW) validateVectorLocked(vec []float32) error {
	if len(vec) == 0 {
		return fmt.Errorf("%w: vector is empty", ErrDimensionMismatch)
	}
	if h.config.Dim < 0 {
		return fmt.Errorf("%w: index has invalid dimension %d", ErrDimensionMismatch, h.config.Dim)
	}

	// Old in-memory indexes can have vectors while Dim is unset. Derive the
	// width from those vectors before accepting another one; otherwise a first
	// post-upgrade insert could overwrite the implicit 384-width contract.
	dim := h.config.Dim
	if dim == 0 && len(h.nodes) > 0 {
		var err error
		dim, err = h.dimensionLocked()
		if err != nil {
			return err
		}
	}
	if dim != 0 && len(vec) != dim {
		return fmt.Errorf("%w: index uses %d dimensions, vector has %d", ErrDimensionMismatch, dim, len(vec))
	}
	return nil
}

func (h *HNSW) dimensionLocked() (int, error) {
	dim := h.config.Dim
	if dim < 0 {
		return 0, fmt.Errorf("%w: index has invalid dimension %d", ErrDimensionMismatch, dim)
	}
	for id, node := range h.nodes {
		if node == nil || len(node.Vec) == 0 {
			return 0, fmt.Errorf("%w: node %q has an empty vector", ErrDimensionMismatch, id)
		}
		if dim == 0 {
			dim = len(node.Vec)
		}
		if len(node.Vec) != dim {
			return 0, fmt.Errorf("%w: node %q has %d dimensions, index uses %d", ErrDimensionMismatch, id, len(node.Vec), dim)
		}
	}
	return dim, nil
}

type SearchResult struct {
	ID       string
	Distance float64
}

// Search keeps the legacy no-error API. New code should use SearchChecked so
// a caller can surface an incompatible query vector instead of treating it as
// an empty result set.
func (h *HNSW) Search(query []float32, topK int) []SearchResult {
	results, _ := h.SearchChecked(query, topK)
	return results
}

// SearchChecked returns an error when the query shape does not match this
// index. It is used by memory and RAG paths to make dimensional drift visible.
func (h *HNSW) SearchChecked(query []float32, topK int) ([]SearchResult, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entryPoint == "" || len(h.nodes) == 0 || topK <= 0 {
		return nil, nil
	}
	if err := h.validateVectorLocked(query); err != nil {
		return nil, err
	}
	ef := h.config.EfSearch
	if topK > ef {
		ef = topK * 2
	}
	curr := h.entryPoint
	for l := h.maxLevel; l > 0; l-- {
		curr = h.searchLayer(query, curr, 1, l)[0]
	}
	candidates := h.searchLayer(query, curr, ef, 0)
	results := make([]SearchResult, len(candidates))
	for i, id := range candidates {
		results[i] = SearchResult{ID: id, Distance: h.distance(query, h.nodes[id].Vec)}
	}
	h.sortByDistance(results)
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// SearchCheckedStable returns an approximate topK result using HNSW traversal,
// with deterministic (distance, ID) ordering for traversal and result heaps.
// Neighbor IDs are visited in sorted order so equal-distance cutoff membership
// is stable for a fixed graph. Memory search continues to use SearchChecked.
func (h *HNSW) SearchCheckedStable(query []float32, topK int) ([]SearchResult, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.entryPoint == "" || len(h.nodes) == 0 || topK <= 0 {
		return nil, nil
	}
	if err := h.validateVectorLocked(query); err != nil {
		return nil, err
	}

	curr := h.entryPoint
	for layer := h.maxLevel; layer > 0; layer-- {
		curr = h.searchLayerStable(query, curr, 1, layer)[0].id
	}
	ef := h.config.EfSearch
	if topK > ef {
		ef = topK * 2
	}
	if ef < 1 {
		ef = 1
	}
	candidates := h.searchLayerStable(query, curr, ef, 0)
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}
	results := make([]SearchResult, len(candidates))
	for i, candidate := range candidates {
		results[i] = SearchResult{ID: candidate.id, Distance: candidate.dist}
	}
	return results, nil
}

// searchLayerStable follows the same bounded graph traversal as searchLayer,
// but uses (distance, ID) as the total ordering and sorts each visited node's
// adjacency list. Work is proportional to visited graph edges and ef, rather
// than the total number of indexed vectors.
func (h *HNSW) searchLayerStable(queryVec []float32, entry string, ef int, layer int) []candidate {
	visited := make(map[string]bool)
	candidates := &stableMinHeap{}
	results := &stableMaxHeap{}
	entryCandidate := candidate{id: entry, dist: h.distance(queryVec, h.nodes[entry].Vec)}
	candidates.push(entryCandidate)
	results.push(entryCandidate)
	visited[entry] = true

	for !candidates.isEmpty() {
		closest := candidates.pop()
		if results.len() >= ef && stableCandidateLess(results.peek(), closest) {
			break
		}
		neighbors := make([]string, 0, len(h.graph[layer][closest.id]))
		for neighbor := range h.graph[layer][closest.id] {
			neighbors = append(neighbors, neighbor)
		}
		sort.Strings(neighbors)
		for _, neighbor := range neighbors {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			current := candidate{id: neighbor, dist: h.distance(queryVec, h.nodes[neighbor].Vec)}
			candidates.push(current)
			if results.len() < ef || stableCandidateLess(current, results.peek()) {
				results.push(current)
				if results.len() > ef {
					results.pop()
				}
			}
		}
	}

	ordered := make([]candidate, results.len())
	for i := len(ordered) - 1; i >= 0; i-- {
		ordered[i] = results.pop()
	}
	return ordered
}

func stableCandidateLess(a, b candidate) bool {
	if a.dist != b.dist {
		return a.dist < b.dist
	}
	return a.id < b.id
}

type stableMinHeap struct{ items []candidate }

func (h *stableMinHeap) push(c candidate) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		parent := (i - 1) / 2
		if !stableCandidateLess(h.items[i], h.items[parent]) {
			break
		}
		h.items[i], h.items[parent] = h.items[parent], h.items[i]
		i = parent
	}
}

func (h *stableMinHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0)
	return top
}

func (h *stableMinHeap) sink(i int) {
	for {
		smallest := i
		left, right := 2*i+1, 2*i+2
		if left < len(h.items) && stableCandidateLess(h.items[left], h.items[smallest]) {
			smallest = left
		}
		if right < len(h.items) && stableCandidateLess(h.items[right], h.items[smallest]) {
			smallest = right
		}
		if smallest == i {
			return
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
	}
}

func (h *stableMinHeap) isEmpty() bool { return len(h.items) == 0 }

type stableMaxHeap struct{ items []candidate }

func (h *stableMaxHeap) push(c candidate) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		parent := (i - 1) / 2
		if !stableCandidateLess(h.items[parent], h.items[i]) {
			break
		}
		h.items[i], h.items[parent] = h.items[parent], h.items[i]
		i = parent
	}
}

func (h *stableMaxHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0)
	return top
}

func (h *stableMaxHeap) peek() candidate { return h.items[0] }

func (h *stableMaxHeap) sink(i int) {
	for {
		largest := i
		left, right := 2*i+1, 2*i+2
		if left < len(h.items) && stableCandidateLess(h.items[largest], h.items[left]) {
			largest = left
		}
		if right < len(h.items) && stableCandidateLess(h.items[largest], h.items[right]) {
			largest = right
		}
		if largest == i {
			return
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}

func (h *stableMaxHeap) len() int { return len(h.items) }

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
		if results.len() >= ef && closest.dist > results.peek().dist {
			break
		}
		for neighbor := range h.graph[layer][closest.id] {
			if visited[neighbor] {
				continue
			}
			visited[neighbor] = true
			ndist := h.distance(queryVec, h.nodes[neighbor].Vec)
			candidates.push(candidate{neighbor, ndist})
			if results.len() < ef || ndist < results.peek().dist {
				results.push(candidate{neighbor, ndist})
				if results.len() > ef {
					results.pop()
				}
			}
		}
	}
	resultIDs := make([]string, results.len())
	for i := len(resultIDs) - 1; i >= 0; i-- {
		resultIDs[i] = results.pop().id
	}
	return resultIDs
}

func (h *HNSW) pruneNeighbors(layer int, nodeID string) {
	neighbors := h.graph[layer][nodeID]
	if len(neighbors) <= h.config.Mmax {
		return
	}
	pairs := make([]struct {
		id   string
		dist float64
	}, 0, len(neighbors))
	for nID, d := range neighbors {
		pairs = append(pairs, struct {
			id   string
			dist float64
		}{nID, d})
	}
	for i := 0; i < len(pairs); i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].dist < pairs[i].dist {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}
	h.graph[layer][nodeID] = make(map[string]float64)
	for i := 0; i < h.config.Mmax && i < len(pairs); i++ {
		h.graph[layer][nodeID][pairs[i].id] = pairs[i].dist
	}
}

// selectRAGNeighbors implements the diversity heuristic used by HNSW
// SELECT-NEIGHBORS (Algorithm 4 / hnswlib getNeighborsByHeuristic2). Candidates
// are considered in deterministic (distance to query, ID) order. A candidate is
// retained only when it is closer to the query than to every selected neighbor.
func (h *HNSW) selectRAGNeighbors(layer int, queryID string, candidates []string, limit int) []string {
	selected := h.selectRAGNeighborSet(layer, queryID, candidates, limit)
	ids := make([]string, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		di := h.distance(h.nodes[queryID].Vec, h.nodes[ids[i]].Vec)
		dj := h.distance(h.nodes[queryID].Vec, h.nodes[ids[j]].Vec)
		if di != dj {
			return di < dj
		}
		return ids[i] < ids[j]
	})
	return ids
}

// selectRAGNeighborSet returns the bounded selected adjacency and recomputes
// cached edge distances. The map iteration order cannot affect tie resolution.
func (h *HNSW) selectRAGNeighborSet(_ int, queryID string, candidates []string, limit int) map[string]float64 {
	if limit <= 0 || h.nodes[queryID] == nil {
		return map[string]float64{}
	}
	seen := make(map[string]bool, len(candidates))
	ordered := make([]string, 0, len(candidates))
	for _, id := range candidates {
		if id == queryID || seen[id] || h.nodes[id] == nil {
			continue
		}
		seen[id] = true
		ordered = append(ordered, id)
	}
	sort.Slice(ordered, func(i, j int) bool {
		di := h.distance(h.nodes[queryID].Vec, h.nodes[ordered[i]].Vec)
		dj := h.distance(h.nodes[queryID].Vec, h.nodes[ordered[j]].Vec)
		if di != dj {
			return di < dj
		}
		return ordered[i] < ordered[j]
	})
	selected := make(map[string]float64, min(limit, len(ordered)))
	for _, candidate := range ordered {
		diverse := true
		for neighbor := range selected {
			if h.distance(h.nodes[candidate].Vec, h.nodes[neighbor].Vec) < h.distance(h.nodes[candidate].Vec, h.nodes[queryID].Vec) {
				diverse = false
				break
			}
		}
		if diverse {
			selected[candidate] = h.distance(h.nodes[queryID].Vec, h.nodes[candidate].Vec)
			if len(selected) == limit {
				break
			}
		}
	}
	return selected
}

// connectRAGLayer applies the selected outgoing links and their reciprocal
// SELECT-NEIGHBORS updates. Existing adjacency is preserved when it rejects the
// new ID; if it accepts, unused slots are refilled from its old neighbors so at
// most one prior edge per candidate is displaced. Every displaced destination
// is reserved in the new node's bounded adjacency. If all updates reject the
// new ID, the nearest selected candidate supplies one bounded inbound edge.
// Callers hold h.mu for write.
func (h *HNSW) connectRAGLayer(layer int, id string, neighbors, candidates []string) {
	if layer < 0 || layer >= len(h.graph) || h.nodes[id] == nil {
		return
	}
	retainedIncoming := false
	displacedSet := make(map[string]bool)
	for _, neighborID := range neighbors {
		if neighborID == id || h.nodes[neighborID] == nil {
			continue
		}
		if h.graph[layer][neighborID] == nil {
			h.graph[layer][neighborID] = make(map[string]float64)
		}
		previous := h.graph[layer][neighborID]
		merged := make([]string, 0, len(previous)+1)
		for existing := range previous {
			merged = append(merged, existing)
		}
		merged = append(merged, id)
		selected := h.selectRAGReciprocalNeighborSet(layer, neighborID, previous, merged, id, h.config.Mmax)
		for existing := range previous {
			if _, retained := selected[existing]; !retained {
				displacedSet[existing] = true
			}
		}
		h.graph[layer][neighborID] = selected
		if _, retained := selected[id]; retained {
			retainedIncoming = true
		}
	}

	// If every reciprocal update rejected the new node, force one bounded
	// inbound edge. This can displace only one old route; route it through the
	// new node along with any prior destinations displaced above.
	if !retainedIncoming {
		if nearest := h.nearestRAGCandidate(id, neighbors); nearest != "" {
			if displaced := h.forceRAGAdjacencyEdge(layer, nearest, id); displaced != "" {
				displacedSet[displaced] = true
			}
		}
	}

	displaced := make([]string, 0, len(displacedSet))
	for oldNeighbor := range displacedSet {
		displaced = append(displaced, oldNeighbor)
	}
	sort.Strings(displaced)
	// Each selected candidate can displace at most one old route because the
	// reciprocal helper fills spare slots from its previous adjacency. The
	// all-rejected fallback runs only when that set is empty, so len(displaced)
	// cannot exceed Mmax under the RAG insertion invariant.
	h.graph[layer][id] = h.selectRAGNeighborsWithRequired(layer, id, neighbors, displaced, h.config.Mmax)
}

// selectRAGReciprocalNeighborSet applies diversity selection to an existing
// candidate list plus the new ID. If the new ID is rejected, leave the old
// adjacency unchanged. If accepted, refill spare slots with old neighbors so a
// reciprocal update can displace at most one old destination from a capped list.
func (h *HNSW) selectRAGReciprocalNeighborSet(layer int, ownerID string, previous map[string]float64, candidates []string, newID string, limit int) map[string]float64 {
	selected := h.selectRAGNeighborSet(layer, ownerID, candidates, limit)
	if _, accepted := selected[newID]; !accepted {
		return previous
	}
	oldIDs := make([]string, 0, len(previous))
	for oldID := range previous {
		if oldID != ownerID && oldID != newID && h.nodes[oldID] != nil {
			oldIDs = append(oldIDs, oldID)
		}
	}
	sort.Slice(oldIDs, func(i, j int) bool {
		di := h.distance(h.nodes[ownerID].Vec, h.nodes[oldIDs[i]].Vec)
		dj := h.distance(h.nodes[ownerID].Vec, h.nodes[oldIDs[j]].Vec)
		if di != dj {
			return di < dj
		}
		return oldIDs[i] < oldIDs[j]
	})
	for _, oldID := range oldIDs {
		if len(selected) >= limit {
			break
		}
		if _, exists := selected[oldID]; !exists {
			selected[oldID] = h.distance(h.nodes[ownerID].Vec, h.nodes[oldID].Vec)
		}
	}
	return selected
}

// selectRAGNeighborsWithRequired keeps every previously displaced destination
// as a mandatory outgoing edge, then fills remaining slots with deterministic
// diversity-aware candidates. Required edges are bounded by Mmax by construction.
func (h *HNSW) selectRAGNeighborsWithRequired(layer int, queryID string, preferred, required []string, limit int) map[string]float64 {
	if limit <= 0 || h.nodes[queryID] == nil {
		return map[string]float64{}
	}
	selected := make(map[string]float64, limit)
	ordered := make([]string, 0, len(preferred)+len(required))
	seen := make(map[string]bool, len(preferred)+len(required))
	for _, id := range required {
		if id == queryID || h.nodes[id] == nil || seen[id] {
			continue
		}
		seen[id] = true
		selected[id] = h.distance(h.nodes[queryID].Vec, h.nodes[id].Vec)
		ordered = append(ordered, id)
	}
	for _, id := range preferred {
		if id == queryID || h.nodes[id] == nil || seen[id] {
			continue
		}
		seen[id] = true
		ordered = append(ordered, id)
	}
	sort.Slice(ordered, func(i, j int) bool {
		di := h.distance(h.nodes[queryID].Vec, h.nodes[ordered[i]].Vec)
		dj := h.distance(h.nodes[queryID].Vec, h.nodes[ordered[j]].Vec)
		if di != dj {
			return di < dj
		}
		return ordered[i] < ordered[j]
	})
	for _, candidate := range ordered {
		if len(selected) >= limit {
			break
		}
		if _, exists := selected[candidate]; exists {
			continue
		}
		diverse := true
		for neighbor := range selected {
			if h.distance(h.nodes[candidate].Vec, h.nodes[neighbor].Vec) < h.distance(h.nodes[candidate].Vec, h.nodes[queryID].Vec) {
				diverse = false
				break
			}
		}
		if diverse {
			selected[candidate] = h.distance(h.nodes[queryID].Vec, h.nodes[candidate].Vec)
		}
	}
	for _, candidate := range ordered {
		if len(selected) >= limit {
			break
		}
		if _, exists := selected[candidate]; !exists {
			selected[candidate] = h.distance(h.nodes[queryID].Vec, h.nodes[candidate].Vec)
		}
	}
	return selected
}

// nearestRAGCandidate picks the nearest valid prior node with a stable ID tie
// break. Candidate lists returned by layer search are already ordered, but keep
// this helper explicit so the fallback remains deterministic independently.
func (h *HNSW) nearestRAGCandidate(queryID string, candidates []string) string {
	if h.nodes[queryID] == nil {
		return ""
	}
	nearest := ""
	nearestDistance := math.Inf(1)
	for _, id := range candidates {
		if id == queryID || h.nodes[id] == nil {
			continue
		}
		distance := h.distance(h.nodes[queryID].Vec, h.nodes[id].Vec)
		if distance < nearestDistance || (distance == nearestDistance && (nearest == "" || id < nearest)) {
			nearest, nearestDistance = id, distance
		}
	}
	return nearest
}

// forceRAGAdjacencyEdge inserts one local edge without exceeding Mmax. If the
// owner is full, it evicts the farthest edge (stable ID tie-break) and returns
// its destination so the caller can preserve that local path through the new
// node. A non-positive Mmax disables insertion rather than violating the cap.
func (h *HNSW) forceRAGAdjacencyEdge(layer int, ownerID, neighborID string) string {
	if layer < 0 || layer >= len(h.graph) || h.config.Mmax <= 0 || ownerID == neighborID ||
		h.nodes[ownerID] == nil || h.nodes[neighborID] == nil {
		return ""
	}
	if h.graph[layer][ownerID] == nil {
		h.graph[layer][ownerID] = make(map[string]float64)
	}
	adjacency := h.graph[layer][ownerID]
	if _, exists := adjacency[neighborID]; exists {
		return ""
	}

	displaced := ""
	if len(adjacency) >= h.config.Mmax {
		farthestDistance := math.Inf(-1)
		for id := range adjacency {
			distance := h.distance(h.nodes[ownerID].Vec, h.nodes[id].Vec)
			if distance > farthestDistance || (distance == farthestDistance && (displaced == "" || id > displaced)) {
				displaced, farthestDistance = id, distance
			}
		}
		if displaced == "" {
			return ""
		}
		delete(adjacency, displaced)
	}
	adjacency[neighborID] = h.distance(h.nodes[ownerID].Vec, h.nodes[neighborID].Vec)
	return displaced
}

// normalizeLegacyRAGGraphLocked removes the old artificial layer-0 backbone
// cycle, then applies diversity-aware SELECT-NEIGHBORS caps in stable node order.
// The caller validates reachability before persisting, failing closed if this
// bounded migration would disconnect any stored node.
func (h *HNSW) normalizeLegacyRAGGraphLocked(legacyOrder []string) {
	if len(h.graph) == 0 {
		h.graph = []map[string]map[string]float64{{}}
	}
	if h.graph[0] == nil {
		h.graph[0] = make(map[string]map[string]float64)
	}
	if len(legacyOrder) > 1 {
		for i, id := range legacyOrder {
			next := legacyOrder[(i+1)%len(legacyOrder)]
			if h.graph[0][id] != nil {
				delete(h.graph[0][id], next)
			}
		}
	}
	for layer := range h.graph {
		if h.graph[layer] == nil {
			h.graph[layer] = make(map[string]map[string]float64)
		}
		ids := make([]string, 0, len(h.graph[layer]))
		for id := range h.graph[layer] {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			candidates := make([]string, 0, len(h.graph[layer][id]))
			for neighbor := range h.graph[layer][id] {
				candidates = append(candidates, neighbor)
			}
			h.graph[layer][id] = h.selectRAGNeighborSet(layer, id, candidates, h.config.Mmax)
		}
	}
	h.ragGraphVersion = currentRAGGraphVersion
}

func (h *HNSW) ragGraphReachableLocked() bool {
	if len(h.nodes) == 0 {
		return true
	}
	if h.entryPoint == "" || len(h.graph) == 0 {
		return false
	}
	reachable := map[string]bool{h.entryPoint: true}
	queue := []string{h.entryPoint}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		for neighbor := range h.graph[0][id] {
			if h.nodes[neighbor] != nil && !reachable[neighbor] {
				reachable[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	return len(reachable) == len(h.nodes)
}

func (h *HNSW) distance(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return math.Sqrt(sum)
}

func (h *HNSW) sortByDistance(results []SearchResult) {
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Distance < results[i].Distance {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}

func (h *HNSW) Delete(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if _, exists := h.nodes[id]; !exists {
		return
	}
	level := h.nodes[id].Level
	delete(h.nodes, id)
	for l := 0; l <= level && l < len(h.graph); l++ {
		delete(h.graph[l], id)
		for nID := range h.graph[l] {
			delete(h.graph[l][nID], id)
		}
	}
	if h.entryPoint == id {
		h.entryPoint = ""
		h.maxLevel = -1
		for id := range h.nodes {
			h.entryPoint = id
			h.maxLevel = h.nodes[id].Level
			break
		}
	}
}

func (h *HNSW) Size() int { h.mu.RLock(); defer h.mu.RUnlock(); return len(h.nodes) }

// Contains reports whether a node with the given id is already indexed.
func (h *HNSW) Contains(id string) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	_, ok := h.nodes[id]
	return ok
}

type hnswData struct {
	EntryPoint       string                          `json:"ep"`
	MaxLevel         int                             `json:"ml"`
	Config           HNSWConfig                      `json:"cfg"`
	Nodes            map[string]*Node                `json:"ns"`
	Graph            []map[string]map[string]float64 `json:"g"`
	RAGGraphVersion  int                             `json:"rag_graph_version,omitempty"`
	RAGBackboneOrder []string                        `json:"rag_backbone_order,omitempty"` // legacy migration input
}

// Save atomically persists the index to path: the payload is marshalled in
// memory, written to a unique temp file in the same directory, fsync'd and
// renamed over path, so a concurrent reader always sees either the old or the
// new index, never a partial one. Writers and readers of the same path are
// serialised by a cross-process flock (path + ".lock").
func (h *HNSW) Save(path string) error {
	lk, err := acquireFileLock(path+lockSuffix, fileLockTimeout)
	if err != nil {
		return err
	}
	defer lk.release()

	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.saveNoLock(path)
}

// saveNoLock marshals and atomically writes the index without taking any lock.
// Callers must hold h.mu for reading and, when another process may touch the
// same path, the file lock.
func (h *HNSW) saveNoLock(path string) error {
	if _, err := h.dimensionLocked(); err != nil {
		return err
	}
	data := hnswData{EntryPoint: h.entryPoint, MaxLevel: h.maxLevel, Config: h.config, Nodes: h.nodes, Graph: h.graph}
	if h.ragBoundedDegree {
		data.RAGGraphVersion = currentRAGGraphVersion
	}
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	return atomicWriteFile(path, b)
}

// atomicWriteFile writes data to a temp file in the same directory, fsyncs it
// and renames it over path. An existing file's permissions are preserved,
// otherwise 0644 is used.
func atomicWriteFile(path string, data []byte) error {
	mode := os.FileMode(0644)
	if fi, err := os.Stat(path); err == nil {
		mode = fi.Mode().Perm()
	}

	tmp := fmt.Sprintf("%s.tmp-%d-%d", path, os.Getpid(), atomic.AddUint64(&tmpSeq, 1))
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, mode)
	if err != nil {
		return fmt.Errorf("hnsw save: create temp %s: %w", tmp, err)
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("hnsw save: write temp %s: %w", tmp, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		os.Remove(tmp)
		return fmt.Errorf("hnsw save: sync temp %s: %w", tmp, err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("hnsw save: close temp %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		os.Remove(tmp)
		return fmt.Errorf("hnsw save: rename %s -> %s: %w", tmp, path, err)
	}
	if d, err := os.Open(filepath.Dir(path)); err == nil {
		d.Sync()
		d.Close()
	} // best-effort directory fsync
	return nil
}

// MergeAndSave merges the in-memory index with the on-disk index at path and
// atomically writes the union back, all under the cross-process lock, so a
// concurrent writer (e.g. a running gateway) cannot lose vectors. Nodes already
// on disk keep their stored vectors and graph links; in-memory nodes missing
// from disk are re-inserted through the normal insert path. It returns the
// number of nodes recovered from disk and the total number written.
func (h *HNSW) MergeAndSave(path string) (merged, total int, err error) {
	lk, err := acquireFileLock(path+lockSuffix, fileLockTimeout)
	if err != nil {
		return 0, 0, err
	}
	defer lk.release()

	h.mu.RLock()
	local := make([]*Node, 0, len(h.nodes))
	for _, n := range h.nodes {
		local = append(local, &Node{ID: n.ID, Vec: n.Vec, Level: n.Level})
	}
	cfg := h.config
	ragBoundedDegree := h.ragBoundedDegree
	h.mu.RUnlock()

	disk := NewHNSW(cfg)
	disk.ragBoundedDegree = ragBoundedDegree
	if e := disk.loadFromDisk(path); e != nil && !os.IsNotExist(e) {
		return 0, 0, e
	}

	diskBefore := len(disk.nodes)
	skipped := 0
	for _, n := range local {
		if _, ok := disk.nodes[n.ID]; ok {
			skipped++
			continue
		}
		if e := disk.insertLocked(n.ID, n.Vec); e != nil {
			return 0, 0, fmt.Errorf("merge hnsw vector %q: %w", n.ID, e)
		}
	}
	merged = diskBefore - skipped // nodes on disk that memory did not have
	if merged < 0 {
		merged = 0
	}
	total = len(disk.nodes)

	if e := disk.saveNoLock(path); e != nil {
		return 0, 0, e
	}
	return merged, total, nil
}

// MigrateLegacyRAGGraph applies and persists the one-time RAG graph migration.
// It must only be called after the caller has validated the index path and
// embedding profile against its manifest. The on-disk index is reread while
// holding its file lock, and an unreachable result is rejected without writing.
func (h *HNSW) MigrateLegacyRAGGraph(path string) error {
	lk, err := acquireFileLock(path+lockSuffix, fileLockTimeout)
	if err != nil {
		return err
	}
	defer lk.release()

	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.ragBoundedDegree {
		return fmt.Errorf("RAG bounded-degree mode is not enabled")
	}
	if err := h.loadFromDisk(path); err != nil {
		return err
	}
	if h.ragGraphVersion >= currentRAGGraphVersion {
		return nil
	}

	originalGraph := cloneGraph(h.graph)
	originalVersion := h.ragGraphVersion
	var data hnswData
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}
	h.normalizeLegacyRAGGraphLocked(data.RAGBackboneOrder)
	if !h.ragGraphReachableLocked() {
		h.graph = originalGraph
		h.ragGraphVersion = originalVersion
		return fmt.Errorf("legacy RAG HNSW normalization would disconnect stored nodes")
	}
	if err := h.saveNoLock(path); err != nil {
		h.graph = originalGraph
		h.ragGraphVersion = originalVersion
		return fmt.Errorf("persist normalized RAG HNSW %s: %w", path, err)
	}
	return nil
}

func cloneGraph(graph []map[string]map[string]float64) []map[string]map[string]float64 {
	copyGraph := make([]map[string]map[string]float64, len(graph))
	for layer, nodes := range graph {
		copyGraph[layer] = make(map[string]map[string]float64, len(nodes))
		for id, neighbors := range nodes {
			copyNeighbors := make(map[string]float64, len(neighbors))
			for neighbor, distance := range neighbors {
				copyNeighbors[neighbor] = distance
			}
			copyGraph[layer][id] = copyNeighbors
		}
	}
	return copyGraph
}

// Load reads the index from path under the same cross-process lock as Save, so
// it never observes a partial write. RAG legacy migration is deliberately
// deferred to MigrateLegacyRAGGraph after profile validation.
func (h *HNSW) Load(path string) error {
	lk, err := acquireFileLock(path+lockSuffix, fileLockTimeout)
	if err != nil {
		return err
	}
	defer lk.release()

	h.mu.Lock()
	defer h.mu.Unlock()
	return h.loadFromDisk(path)
}

// loadFromDisk reads and installs the on-disk index without taking any lock.
func (h *HNSW) loadFromDisk(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var data hnswData
	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}
	if err := validateLoadedDimension(&data); err != nil {
		return fmt.Errorf("hnsw load: %w", err)
	}
	if h.ragBoundedDegree && data.RAGGraphVersion > currentRAGGraphVersion {
		return fmt.Errorf("hnsw load: unsupported RAG graph version %d", data.RAGGraphVersion)
	}
	h.entryPoint, h.maxLevel, h.config, h.nodes, h.graph = data.EntryPoint, data.MaxLevel, data.Config, data.Nodes, data.Graph
	h.ragGraphVersion = data.RAGGraphVersion
	if h.nodes == nil {
		h.nodes = make(map[string]*Node)
	}
	if h.graph == nil {
		h.graph = make([]map[string]map[string]float64, 1)
		h.graph[0] = make(map[string]map[string]float64)
	}
	if !h.ragBoundedDegree {
		h.symmetrizeLocked()
	}
	h.rng = rand.New(rand.NewSource(42))
	return nil
}

// validateLoadedDimension rejects mixed-width vectors before Load mutates a
// live HNSW. Older index files may omit cfg.Dim; infer it from their vectors so
// their existing width is still enforced on future writes.
func validateLoadedDimension(data *hnswData) error {
	dim := data.Config.Dim
	if dim < 0 {
		return fmt.Errorf("%w: index has invalid dimension %d", ErrDimensionMismatch, dim)
	}
	for id, node := range data.Nodes {
		if node == nil || len(node.Vec) == 0 {
			return fmt.Errorf("%w: node %q has an empty vector", ErrDimensionMismatch, id)
		}
		if dim == 0 {
			dim = len(node.Vec)
		}
		if len(node.Vec) != dim {
			return fmt.Errorf("%w: node %q has %d dimensions, index uses %d", ErrDimensionMismatch, id, len(node.Vec), dim)
		}
	}
	if data.Config.Dim == 0 {
		data.Config.Dim = dim
	}
	return nil
}

// symmetrizeLocked restores reciprocity on every layer: for each edge a->b it
// adds the missing reverse edge b->a. HNSW links are conceptually undirected,
// but pruning while inserting can drop the reverse half, and an on-disk index
// that accumulated such drops contains nodes with no incoming links at layer 0.
// Search only walks edges forward from the entry point, so those nodes are
// invisible to it even when they hold the closest vector. Making the graph
// symmetric guarantees every stored node is reachable. Callers must hold h.mu.
func (h *HNSW) symmetrizeLocked() {
	type edge struct {
		src, dst string
		d        float64
	}
	for _, layer := range h.graph {
		var missing []edge
		for src, neighbours := range layer {
			for dst, d := range neighbours {
				if rev, ok := layer[dst]; !ok || rev == nil {
					missing = append(missing, edge{src, dst, d})
				} else if _, ok := rev[src]; !ok {
					missing = append(missing, edge{src, dst, d})
				}
			}
		}
		for _, e := range missing {
			if layer[e.dst] == nil {
				layer[e.dst] = make(map[string]float64)
			}
			layer[e.dst][e.src] = e.d
		}
	}
}

type candidate struct {
	id   string
	dist float64
}
type minHeap struct{ items []candidate }

func newMinHeap() *minHeap { return &minHeap{} }
func (h *minHeap) push(c candidate) {
	h.items = append(h.items, c)
	for i := len(h.items) - 1; i > 0; {
		p := (i - 1) / 2
		if h.items[i].dist >= h.items[p].dist {
			break
		}
		h.items[i], h.items[p] = h.items[p], h.items[i]
		i = p
	}
}
func (h *minHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0)
	return top
}
func (h *minHeap) sink(i int) {
	for {
		smallest := i
		l, r := 2*i+1, 2*i+2
		if l < len(h.items) && h.items[l].dist < h.items[smallest].dist {
			smallest = l
		}
		if r < len(h.items) && h.items[r].dist < h.items[smallest].dist {
			smallest = r
		}
		if smallest == i {
			break
		}
		h.items[i], h.items[smallest] = h.items[smallest], h.items[i]
		i = smallest
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
		if h.items[i].dist <= h.items[p].dist {
			break
		}
		h.items[i], h.items[p] = h.items[p], h.items[i]
		i = p
	}
}
func (h *maxHeap) pop() candidate {
	top := h.items[0]
	h.items[0] = h.items[len(h.items)-1]
	h.items = h.items[:len(h.items)-1]
	h.sink(0)
	return top
}
func (h *maxHeap) peek() candidate { return h.items[0] }
func (h *maxHeap) sink(i int) {
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < len(h.items) && h.items[l].dist > h.items[largest].dist {
			largest = l
		}
		if r < len(h.items) && h.items[r].dist > h.items[largest].dist {
			largest = r
		}
		if largest == i {
			break
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}
func (h *maxHeap) isEmpty() bool { return len(h.items) == 0 }
func (h *maxHeap) len() int      { return len(h.items) }

package index

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
)

func f(s string, n int) string {
	if n < 0 {
		return "invalid"
	}
	var rev [8]byte
	pos := 0
	if n == 0 {
		rev[pos] = '0'
		pos++
	} else {
		for n > 0 && pos < 8 {
			rev[pos] = byte('0' + n%10)
			n /= 10
			pos++
		}
	}
	b := make([]byte, 0, len(s)+pos)
	for i := 0; i < len(s); i++ {
		if s[i] == '%' && i+1 < len(s) && s[i+1] == 'd' {
			for j := pos - 1; j >= 0; j-- {
				b = append(b, rev[j])
			}
			i++
		} else {
			b = append(b, s[i])
		}
	}
	return string(b)
}

func TestHNSWInsertAndSearch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	h.Insert("zero", []float32{0, 0, 0, 0})
	h.Insert("one", []float32{1, 1, 1, 1})
	h.Insert("two", []float32{2, 2, 2, 2})
	h.Insert("three", []float32{3, 3, 3, 3})
	h.Insert("ten", []float32{10, 10, 10, 10})
	results := h.Search([]float32{0, 0, 0, 0}, 3)
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0].ID != "zero" {
		t.Errorf("expected 'zero' as top result, got %s", results[0].ID)
	}
}

func TestSearchCheckedStableDeterministicTieCutoffUsesGraphTraversal(t *testing.T) {
	cfg := DefaultHNSWConfig(1)
	cfg.EfSearch = 3
	h := NewHNSW(cfg)
	h.nodes = map[string]*Node{
		"rag_chk_a": {ID: "rag_chk_a", Vec: []float32{1}, Level: 0},
		"rag_chk_b": {ID: "rag_chk_b", Vec: []float32{1}, Level: 0},
		"rag_chk_c": {ID: "rag_chk_c", Vec: []float32{1}, Level: 0},
		"rag_chk_d": {ID: "rag_chk_d", Vec: []float32{1}, Level: 0},
		"rag_chk_z": {ID: "rag_chk_z", Vec: []float32{1}, Level: 0},
		// This globally closest node is intentionally unreachable from the graph.
		// Stable search must not return it by scanning h.nodes.
		"unreachable": {ID: "unreachable", Vec: []float32{0}, Level: 0},
	}
	h.graph = []map[string]map[string]float64{{
		"rag_chk_a": {"rag_chk_z": 1},
		"rag_chk_b": {"rag_chk_z": 1},
		"rag_chk_c": {"rag_chk_z": 1},
		"rag_chk_d": {"rag_chk_z": 1},
		"rag_chk_z": {
			"rag_chk_d": 1,
			"rag_chk_c": 1,
			"rag_chk_b": 1,
			"rag_chk_a": 1,
		},
		"unreachable": {},
	}}
	h.entryPoint = "rag_chk_z"
	h.maxLevel = 0

	want := []string{"rag_chk_a", "rag_chk_b", "rag_chk_c"}
	for repeat := 0; repeat < 50; repeat++ {
		results, err := h.SearchCheckedStable([]float32{0}, len(want))
		if err != nil {
			t.Fatalf("stable search repeat %d: %v", repeat, err)
		}
		if got := resultIDs(results); !equalStrings(got, want) {
			t.Fatalf("stable result IDs on repeat %d = %v, want %v", repeat, got, want)
		}
	}

	// The memory-facing API remains the original approximate traversal and does
	// not scan all nodes either; in particular, it cannot see the disconnected ID.
	memoryResults, err := h.SearchChecked([]float32{0}, len(want))
	if err != nil {
		t.Fatalf("SearchChecked: %v", err)
	}
	for _, result := range memoryResults {
		if result.ID == "unreachable" {
			t.Fatal("SearchChecked returned a node outside its graph traversal")
		}
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestHNSWLargeDimensions(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(768))
	for i := 0; i < 10; i++ {
		vec := make([]float32, 768)
		for j := 0; j < 768; j++ {
			vec[j] = float32(i + j)
		}
		h.Insert(f("mem_%d", i), vec)
	}
	query := make([]float32, 768)
	for j := 0; j < 768; j++ {
		query[j] = float32(j)
	}
	results := h.Search(query, 3)
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
	if results[0].ID != "mem_0" {
		t.Errorf("expected 'mem_0', got %s", results[0].ID)
	}
}

func TestHNSWDelete(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	h.Insert("a", []float32{0, 0, 0, 0})
	h.Insert("b", []float32{10, 10, 10, 10})
	if h.Size() != 2 {
		t.Fatalf("expected size 2, got %d", h.Size())
	}
	h.Delete("a")
	if h.Size() != 1 {
		t.Errorf("expected size 1, got %d", h.Size())
	}
	results := h.Search([]float32{0, 0, 0, 0}, 5)
	if len(results) == 0 {
		t.Fatal("expected results after delete")
	}
	if results[0].ID == "a" {
		t.Error("deleted element should not be returned")
	}
}

func TestHNSWEmptySearch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	results := h.Search([]float32{1, 2, 3, 4}, 5)
	if len(results) != 0 {
		t.Errorf("empty index should return 0 results, got %d", len(results))
	}
}

// A node that has outgoing edges but no incoming edge at layer 0 is invisible to
// Search, which only walks edges forward from the entry point. A persisted index
// can contain such nodes after pruning drops their reverse links. Load must
// restore reciprocity so a nearest neighbour that is stored is still returned.
func TestHNSWLoadRestoresReachability(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idx.hnsw")

	h := NewHNSW(HNSWConfig{M: 2, Mmax: 2, EfConstruction: 10, EfSearch: 10, ML: 0.5, Dim: 2})
	// Triangle a->b->c->a plus "d" pointing at "a" with no edge back to "d":
	// "d" has out-degree 1 and in-degree 0, so it is unreachable from entry "a".
	h.nodes = map[string]*Node{
		"a": {ID: "a", Vec: []float32{1, 0}, Level: 0},
		"b": {ID: "b", Vec: []float32{0, 1}, Level: 0},
		"c": {ID: "c", Vec: []float32{1, 1}, Level: 0},
		"d": {ID: "d", Vec: []float32{0.99, 0.01}, Level: 0},
	}
	h.graph = []map[string]map[string]float64{{
		"a": {"b": 1}, "b": {"c": 1}, "c": {"a": 1}, "d": {"a": 0.02},
	}}
	h.entryPoint = "a"
	h.maxLevel = 0

	if err := h.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}

	g := NewHNSW(h.config)
	if err := g.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	// "d" is the closest vector to the query, so it must come back first.
	res := g.Search([]float32{0.99, 0.01}, 4)
	if len(res) == 0 || res[0].ID != "d" {
		t.Fatalf("expected nearest node 'd' to be searched after load, got %v", resultIDs(res))
	}
	if len(res) != 4 {
		t.Errorf("expected all 4 nodes reachable after load, got %d", len(res))
	}
}

// Inserting a node whose neighbours are full must still leave it with an
// incoming edge, otherwise it is dropped from every future Search.
func TestHNSWInsertLeavesIncomingEdge(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	for i := 0; i < 200; i++ {
		h.Insert(f("c%d", i), []float32{float32(i % 7), float32(i % 5), float32(i % 3), float32(i % 11)})
	}
	h.Insert("needle", []float32{9, 9, 9, 9})

	indeg := make(map[string]int, len(h.nodes))
	for _, neighbours := range h.graph[0] {
		for dst := range neighbours {
			indeg[dst]++
		}
	}
	for id := range h.nodes {
		if indeg[id] == 0 {
			t.Errorf("node %s has no incoming layer-0 edge, it can never be returned by Search", id)
		}
	}
	res := h.Search([]float32{9, 9, 9, 9}, 3)
	if len(res) == 0 || res[0].ID != "needle" {
		t.Fatalf("expected 'needle' first, got %v", resultIDs(res))
	}
}

func TestRAGReciprocalRejectionUsesBoundedFallbackAtMmaxOne(t *testing.T) {
	cfg := HNSWConfig{M: 1, Mmax: 1, EfConstruction: 8, EfSearch: 8, ML: 0, Dim: 1}
	h := NewHNSW(cfg)
	h.EnableRAGBoundedDegree()
	h.nodes = map[string]*Node{
		"candidate": {ID: "candidate", Vec: []float32{10}, Level: 0},
		"displaced": {ID: "displaced", Vec: []float32{1}, Level: 0},
		"new":       {ID: "new", Vec: []float32{0}, Level: 0},
	}
	h.graph = []map[string]map[string]float64{{
		"candidate": {"displaced": 9},
		"displaced": {"candidate": 9},
		"new":       {},
	}}
	h.entryPoint = "candidate"
	h.maxLevel = 0

	// The reciprocal SELECT-NEIGHBORS list prefers the old node: it is closer
	// to both candidate and the new vector, so every selected candidate rejects
	// the new ID. Mmax=1 leaves no spare slot and exercises the replacement path.
	selected := h.selectRAGNeighbors(0, "new", []string{"candidate"}, cfg.Mmax)
	if !equalStrings(selected, []string{"candidate"}) {
		t.Fatalf("selected outgoing neighbors = %v, want [candidate]", selected)
	}
	reciprocal := h.selectRAGNeighborSet(0, "candidate", []string{"displaced", "new"}, cfg.Mmax)
	if _, retained := reciprocal["new"]; retained {
		t.Fatal("adversarial reciprocal SELECT-NEIGHBORS unexpectedly retained new ID")
	}

	h.connectRAGLayer(0, "new", selected, []string{"candidate"})
	if _, ok := h.graph[0]["candidate"]["new"]; !ok {
		t.Fatal("fallback did not replace candidate adjacency with inbound edge to new node")
	}
	if _, ok := h.graph[0]["new"]["displaced"]; !ok {
		t.Fatal("fallback did not route the displaced neighbor through the new node")
	}
	for id, neighbors := range h.graph[0] {
		if len(neighbors) > cfg.Mmax {
			t.Fatalf("node %q degree = %d, exceeds Mmax %d", id, len(neighbors), cfg.Mmax)
		}
	}
	if !h.ragGraphReachableLocked() {
		t.Fatal("bounded fallback left a node unreachable from the entry point")
	}
}

func TestRAGMixedReciprocalDisplacementPreservesReachability(t *testing.T) {
	cfg := HNSWConfig{M: 3, Mmax: 3, EfConstruction: 8, EfSearch: 8, ML: 0, Dim: 2}
	h := NewHNSW(cfg)
	h.EnableRAGBoundedDegree()
	h.nodes = map[string]*Node{
		"entry": {ID: "entry", Vec: []float32{-5, -5}, Level: 0},
		"a":     {ID: "a", Vec: []float32{1, 0}, Level: 0},
		"b":     {ID: "b", Vec: []float32{0, 2}, Level: 0},
		"c":     {ID: "c", Vec: []float32{0, -2}, Level: 0},
		"d":     {ID: "d", Vec: []float32{0, -3}, Level: 0},
		"e":     {ID: "e", Vec: []float32{10, 2}, Level: 0},
		"f":     {ID: "f", Vec: []float32{-10, 2}, Level: 0},
		"g":     {ID: "g", Vec: []float32{10, -2}, Level: 0},
		"h":     {ID: "h", Vec: []float32{-10, -2}, Level: 0},
		"i":     {ID: "i", Vec: []float32{0, -12}, Level: 0},
		"new":   {ID: "new", Vec: []float32{0, 0}, Level: 0},
	}
	h.graph = []map[string]map[string]float64{{
		"entry": {"a": 1, "b": 2, "c": 2},
		"a":     {},
		"b":     {"d": 5, "e": 10, "f": 10},
		"c":     {"g": 10, "h": 10, "i": 10},
		"d":     {},
		"e":     {},
		"f":     {},
		"g":     {},
		"h":     {},
		"i":     {},
		"new":   {},
	}}
	h.entryPoint = "entry"
	h.maxLevel = 0

	neighbors := h.selectRAGNeighbors(0, "new", []string{"a", "b", "c"}, cfg.Mmax)
	if !equalStrings(neighbors, []string{"a", "b", "c"}) {
		t.Fatalf("new-node SELECT-NEIGHBORS = %v, want [a b c]", neighbors)
	}
	for ownerID, oldIDs := range map[string][]string{
		"b": {"d", "e", "f"},
		"c": {"g", "h", "i"},
	} {
		selected := append(append([]string(nil), oldIDs...), "new")
		reciprocal := h.selectRAGNeighborSet(0, ownerID, selected, cfg.Mmax)
		if _, retained := reciprocal["new"]; !retained {
			t.Fatalf("candidate %s did not retain new ID", ownerID)
		}
		var displaced string
		for _, oldID := range oldIDs {
			if _, retained := reciprocal[oldID]; !retained {
				displaced = oldID
				break
			}
		}
		if displaced == "" {
			t.Fatalf("candidate %s did not displace an old node", ownerID)
		}
	}
	newNode := h.nodes["new"]
	delete(h.nodes, "new")
	if !h.ragGraphReachableLocked() {
		t.Fatal("adversarial fixture must start with every old node reachable")
	}
	h.nodes["new"] = newNode

	h.connectRAGLayer(0, "new", neighbors, []string{"a", "b", "c"})
	for _, candidateID := range []string{"a", "b", "c"} {
		if _, ok := h.graph[0][candidateID]["new"]; !ok {
			t.Fatalf("candidate %s did not retain its reciprocal edge to new ID", candidateID)
		}
	}
	for _, oldID := range []string{"d", "i"} {
		if _, ok := h.graph[0]["new"][oldID]; !ok {
			t.Fatalf("displaced old node %s was not routed through new ID", oldID)
		}
	}
	if !h.ragGraphReachableLocked() {
		t.Fatal("mixed reciprocal update left an old node or new node unreachable")
	}
	for id, adjacency := range h.graph[0] {
		if len(adjacency) > cfg.Mmax {
			t.Fatalf("node %q degree = %d, exceeds Mmax %d", id, len(adjacency), cfg.Mmax)
		}
	}
}

func TestRAGBoundedDegreeCapsEveryLayerAndPreservesStableReachability(t *testing.T) {
	cfg := HNSWConfig{M: 4, Mmax: 4, EfConstruction: 40, EfSearch: 512, ML: 0.5, Dim: 3}
	h := NewHNSW(cfg)
	h.EnableRAGBoundedDegree()
	const count = 300
	vectors := make(map[string][]float32, count)
	for i := 0; i < count; i++ {
		id := f("rag-degree-%d", i)
		v := []float32{float32(math.Sin(float64(i) * 0.17)), float32(math.Cos(float64(i) * 0.13)), float32(i % 19)}
		vectors[id] = v
		if err := h.Insert(id, v); err != nil {
			t.Fatalf("insert %s: %v", id, err)
		}
	}

	for layer, nodes := range h.graph {
		for id, neighbors := range nodes {
			if len(neighbors) > cfg.Mmax {
				t.Errorf("layer %d node %q degree = %d, want <= Mmax %d", layer, id, len(neighbors), cfg.Mmax)
			}
		}
	}

	// Bounded HNSW adjacency is directed after reciprocal lists are independently
	// pruned; search must stay inside the component reachable from its entry point
	// rather than faking global recall with a graph-wide scan or artificial ring.
	reachable := map[string]bool{h.entryPoint: true}
	queue := []string{h.entryPoint}
	for len(queue) > 0 {
		id := queue[0]
		queue = queue[1:]
		for neighbor := range h.graph[0][id] {
			if !reachable[neighbor] {
				reachable[neighbor] = true
				queue = append(queue, neighbor)
			}
		}
	}
	if len(reachable) != len(h.nodes) {
		t.Fatalf("bounded graph reaches %d of %d nodes from entry point %q", len(reachable), len(h.nodes), h.entryPoint)
	}
	neighbors := make([]string, 0, len(h.graph[0][h.entryPoint]))
	for neighbor := range h.graph[0][h.entryPoint] {
		neighbors = append(neighbors, neighbor)
	}
	sort.Strings(neighbors)
	queryID := neighbors[0]
	results, err := h.SearchCheckedStable(vectors[queryID], 1)
	if err != nil {
		t.Fatalf("stable search: %v", err)
	}
	if len(results) != 1 || results[0].ID != queryID {
		t.Fatalf("stable search for reachable vector %q = %v", queryID, resultIDs(results))
	}
	for _, result := range results {
		if !reachable[result.ID] {
			t.Fatalf("stable search returned unreachable node %q", result.ID)
		}
	}
}

func TestDefaultHNSWDoesNotEnableRAGDegreeMode(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(1))
	if h.ragBoundedDegree {
		t.Fatal("generic/memory HNSW unexpectedly enabled RAG bounded-degree mode")
	}
}

func TestRAGInsertionAndTieSelectionAreDeterministic(t *testing.T) {
	cfg := HNSWConfig{M: 3, Mmax: 3, EfConstruction: 32, EfSearch: 16, ML: 0.5, Dim: 2}
	indexes := []*HNSW{NewHNSW(cfg), NewHNSW(cfg)}
	for _, h := range indexes {
		h.EnableRAGBoundedDegree()
		for i := 0; i < 80; i++ {
			id := f("tie-%d", i)
			if err := h.Insert(id, []float32{1, 1}); err != nil {
				t.Fatalf("insert %s: %v", id, err)
			}
		}
	}
	if !reflect.DeepEqual(indexes[0].graph, indexes[1].graph) {
		t.Fatal("identical RAG insertion sequences produced different adjacency maps")
	}
}

func TestRAGUpperLayerGreedyDescentUsesStableTieBreak(t *testing.T) {
	cfg := HNSWConfig{M: 2, Mmax: 2, EfConstruction: 8, EfSearch: 8, ML: 0, Dim: 2}
	for attempt := 0; attempt < 50; attempt++ {
		h := NewHNSW(cfg)
		h.EnableRAGBoundedDegree()
		h.nodes = map[string]*Node{
			"root": {ID: "root", Vec: []float32{10, 10}, Level: 1},
			"a":    {ID: "a", Vec: []float32{1, 0}, Level: 1},
			"b":    {ID: "b", Vec: []float32{-1, 0}, Level: 1},
		}
		h.graph = []map[string]map[string]float64{
			{},
			{
				"root": {"b": 1, "a": 1},
				"a":    {},
				"b":    {},
			},
		}
		h.entryPoint = "root"
		h.maxLevel = 1

		got := h.greedyDescent([]float32{0, 0}, "root", 1, 0)
		if got != "a" {
			t.Fatalf("attempt %d: stable upper-layer descent = %q, want tie-broken ID a", attempt, got)
		}
	}
}

func TestRAGNeighborSelectionUsesDiversityAndStableIDTies(t *testing.T) {
	cfg := HNSWConfig{M: 4, Mmax: 4, EfConstruction: 8, EfSearch: 8, ML: 0.5, Dim: 2}
	h := NewHNSW(cfg)
	h.EnableRAGBoundedDegree()
	h.nodes = map[string]*Node{
		"q":     {ID: "q", Vec: []float32{0, 0}, Level: 0},
		"a":     {ID: "a", Vec: []float32{1, 0}, Level: 0},
		"b":     {ID: "b", Vec: []float32{2, 0}, Level: 0},
		"c":     {ID: "c", Vec: []float32{0, 3}, Level: 0},
		"tie-a": {ID: "tie-a", Vec: []float32{-1, 0}, Level: 0},
		"tie-b": {ID: "tie-b", Vec: []float32{1, 0}, Level: 0},
	}
	got := h.selectRAGNeighbors(0, "q", []string{"b", "c", "a"}, 3)
	if !equalStrings(got, []string{"a", "c"}) {
		t.Fatalf("diversity-selected neighbors = %v, want [a c]", got)
	}
	tied := h.selectRAGNeighbors(0, "q", []string{"tie-b", "tie-a"}, 2)
	if !equalStrings(tied, []string{"tie-a", "tie-b"}) {
		t.Fatalf("equal-distance neighbors = %v, want stable ID order", tied)
	}
}

func TestRAGLegacyGraphNormalizationIsVersionedAndOneTime(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy-rag.hnsw")
	data := hnswData{
		EntryPoint: "a",
		MaxLevel:   0,
		Config:     HNSWConfig{M: 2, Mmax: 2, EfConstruction: 16, EfSearch: 8, ML: 0.5, Dim: 1},
		Nodes: map[string]*Node{
			"a": {ID: "a", Vec: []float32{0}, Level: 0},
			"b": {ID: "b", Vec: []float32{1}, Level: 0},
			"c": {ID: "c", Vec: []float32{2}, Level: 0},
			"d": {ID: "d", Vec: []float32{3}, Level: 0},
		},
		Graph: []map[string]map[string]float64{{
			"a": {"b": 1, "c": 2, "d": 3},
			"b": {"c": 1, "a": 2, "d": 3},
			"c": {"d": 1, "b": 2, "a": 3},
			"d": {"a": 1, "c": 2, "b": 3},
		}},
		RAGBackboneOrder: []string{"a", "b", "c", "d"},
	}
	encoded, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("marshal legacy graph: %v", err)
	}
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		t.Fatalf("write legacy graph: %v", err)
	}

	first := NewHNSW(data.Config)
	first.EnableRAGBoundedDegree()
	beforeLoad, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read legacy graph before load: %v", err)
	}
	if err := first.Load(path); err != nil {
		t.Fatalf("load legacy graph without migration: %v", err)
	}
	afterLoad, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read legacy graph after load: %v", err)
	}
	if !reflect.DeepEqual(beforeLoad, afterLoad) {
		t.Fatal("RAG Load mutated the legacy file before profile validation")
	}
	if err := first.MigrateLegacyRAGGraph(path); err != nil {
		t.Fatalf("migrate legacy graph: %v", err)
	}
	if !first.ragGraphReachableLocked() {
		t.Fatal("migrated legacy graph is not reachable from its entry point")
	}
	for layer, nodes := range first.graph {
		for id, neighbors := range nodes {
			if len(neighbors) > data.Config.Mmax {
				t.Fatalf("normalized layer %d node %q degree %d exceeds Mmax %d", layer, id, len(neighbors), data.Config.Mmax)
			}
		}
	}
	persisted, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read normalized graph: %v", err)
	}
	var saved hnswData
	if err := json.Unmarshal(persisted, &saved); err != nil {
		t.Fatalf("decode normalized graph: %v", err)
	}
	if saved.RAGGraphVersion != currentRAGGraphVersion || len(saved.RAGBackboneOrder) != 0 {
		t.Fatalf("saved migration metadata version=%d legacy_order=%v", saved.RAGGraphVersion, saved.RAGBackboneOrder)
	}

	second := NewHNSW(data.Config)
	second.EnableRAGBoundedDegree()
	if err := second.Load(path); err != nil {
		t.Fatalf("reload normalized graph: %v", err)
	}
	if !reflect.DeepEqual(first.graph, second.graph) {
		t.Fatal("current-version RAG graph changed on a subsequent load")
	}
}

func TestRAGLegacyMigrationFailsClosedWithoutWritingDisconnectedGraph(t *testing.T) {
	path := filepath.Join(t.TempDir(), "disconnected-legacy-rag.hnsw")
	data := hnswData{
		EntryPoint: "a",
		MaxLevel:   0,
		Config:     HNSWConfig{M: 1, Mmax: 1, EfConstruction: 8, EfSearch: 8, ML: 0.5, Dim: 1},
		Nodes: map[string]*Node{
			"a": {ID: "a", Vec: []float32{0}, Level: 0},
			"b": {ID: "b", Vec: []float32{1}, Level: 0},
		},
		Graph:            []map[string]map[string]float64{{"a": {"b": 1}, "b": {"a": 1}}},
		RAGBackboneOrder: []string{"a", "b"},
	}
	encoded, err := json.Marshal(data)
	if err != nil {
		t.Fatalf("marshal disconnected legacy graph: %v", err)
	}
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		t.Fatalf("write disconnected legacy graph: %v", err)
	}

	h := NewHNSW(data.Config)
	h.EnableRAGBoundedDegree()
	if err := h.Load(path); err != nil {
		t.Fatalf("load disconnected legacy graph: %v", err)
	}
	if err := h.MigrateLegacyRAGGraph(path); err == nil {
		t.Fatal("migration succeeded despite losing reachability")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read disconnected graph after failed migration: %v", err)
	}
	if !reflect.DeepEqual(got, encoded) {
		t.Fatal("failed migration changed the legacy HNSW file")
	}
}

func resultIDs(res []SearchResult) []string {
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

func TestHNSWSimilarity(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(3))
	h.Insert("A", []float32{1, 0, 0})
	h.Insert("B", []float32{0.95, 0.1, 0})
	h.Insert("C", []float32{0, 1, 0})
	h.Insert("D", []float32{0, 0, 1})
	results := h.Search([]float32{1, 0, 0}, 3)
	if len(results) < 2 {
		t.Fatal("expected at least 2 results")
	}
	if results[0].ID != "A" {
		t.Errorf("expected A first, got %s", results[0].ID)
	}
	if results[1].ID != "B" {
		t.Errorf("expected B second, got %s", results[1].ID)
	}
}

func TestHNSWManyInserts(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(64))
	for i := 0; i < 100; i++ {
		vec := make([]float32, 64)
		for j := 0; j < 64; j++ {
			vec[j] = float32(float64(i) * math.Sin(float64(j)))
		}
		h.Insert(f("m%d", i), vec)
	}
	if h.Size() != 100 {
		t.Errorf("expected 100, got %d", h.Size())
	}
	q := make([]float32, 64)
	for j := 0; j < 64; j++ {
		q[j] = float32(math.Sin(float64(j)))
	}
	results := h.Search(q, 5)
	if len(results) != 5 {
		t.Errorf("expected 5, got %d", len(results))
	}
}

// --- durability / concurrency tests -----------------------------------------

func newTestIndex(n int) *HNSW {
	h := NewHNSW(DefaultHNSWConfig(4))
	for i := 0; i < n; i++ {
		h.Insert(f("n%d", i), []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)})
	}
	return h
}

func countTempFiles(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	n := 0
	for _, e := range entries {
		if strings.Contains(e.Name(), ".tmp-") {
			n++
		}
	}
	return n
}

// (a) Save then Load round-trips node count and vectors.
func TestHNSWSaveLoadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idx.hnsw")
	h := newTestIndex(50)
	if err := h.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}

	g := NewHNSW(DefaultHNSWConfig(4))
	if err := g.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	if g.Size() != h.Size() {
		t.Fatalf("node count mismatch: saved %d, loaded %d", h.Size(), g.Size())
	}
	for id, n := range h.nodes {
		m, ok := g.nodes[id]
		if !ok {
			t.Fatalf("node %s missing after load", id)
		}
		if len(m.Vec) != len(n.Vec) {
			t.Fatalf("node %s vector length mismatch: %d != %d", id, len(m.Vec), len(n.Vec))
		}
		for i := range n.Vec {
			if m.Vec[i] != n.Vec[i] {
				t.Fatalf("node %s vector[%d] mismatch: %v != %v", id, i, m.Vec[i], n.Vec[i])
			}
		}
	}
}

// (b) Repeated Save atomically overwrites the previous file: the final file is
// always valid JSON, permissions are preserved, and no temp files leak.
func TestHNSWSaveAtomicOverwrite(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "idx.hnsw")

	h := newTestIndex(5)
	if err := h.Save(path); err != nil {
		t.Fatalf("initial save: %v", err)
	}
	// Simulate a pre-existing non-default mode that must not be widened.
	if err := os.Chmod(path, 0640); err != nil {
		t.Fatalf("chmod: %v", err)
	}

	for i := 0; i < 5; i++ {
		h.Insert(f("extra-%d", i), []float32{float32(i), 0, 0, 1})
		if err := h.Save(path); err != nil {
			t.Fatalf("save %d: %v", i, err)
		}
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read after save %d: %v", i, err)
		}
		var probe map[string]any
		if err := json.Unmarshal(b, &probe); err != nil {
			t.Fatalf("file is not valid JSON after save %d: %v", i, err)
		}
	}

	fi, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if fi.Mode().Perm() != 0640 {
		t.Errorf("permissions changed: got %v, want 0640", fi.Mode().Perm())
	}
	if n := countTempFiles(t, dir); n != 0 {
		t.Errorf("expected no temp files left behind, found %d", n)
	}

	// The final file must reload to the latest state.
	g := NewHNSW(DefaultHNSWConfig(4))
	if err := g.Load(path); err != nil {
		t.Fatalf("load final: %v", err)
	}
	if g.Size() != h.Size() {
		t.Errorf("final size mismatch: got %d, want %d", g.Size(), h.Size())
	}
}

// (c) Concurrent Saves from multiple goroutines to the same path must never
// produce a corrupt file.
func TestHNSWConcurrentSaveSamePath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "idx.hnsw")

	h := newTestIndex(200)
	const writers = 8
	errs := make([]error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = h.Save(path)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("writer %d: %v", i, err)
		}
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var probe map[string]any
	if err := json.Unmarshal(b, &probe); err != nil {
		t.Fatalf("concurrent save produced invalid JSON: %v", err)
	}
	g := NewHNSW(DefaultHNSWConfig(4))
	if err := g.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	if g.Size() != h.Size() {
		t.Errorf("size mismatch after concurrent saves: got %d, want %d", g.Size(), h.Size())
	}
	if n := countTempFiles(t, dir); n != 0 {
		t.Errorf("expected no temp files left behind, found %d", n)
	}
}

// MergeAndSave must union the in-memory nodes with the on-disk nodes so a
// concurrent writer's vectors are not lost.
func TestHNSWMergeAndSave(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idx.hnsw")

	// On-disk index as if written by a running gateway.
	disk := newTestIndex(10)
	if err := disk.Save(path); err != nil {
		t.Fatalf("save disk: %v", err)
	}

	// In-memory index holds a fresh set of vectors plus a shared overlap.
	mem := NewHNSW(DefaultHNSWConfig(4))
	for i := 0; i < 10; i++ {
		mem.Insert(f("n%d", i), []float32{float32(i), float32(i + 1), float32(i + 2), float32(i + 3)})
	}
	mem.Insert("fresh-1", []float32{100, 100, 100, 100})
	mem.Insert("fresh-2", []float32{200, 200, 200, 200})

	merged, total, err := mem.MergeAndSave(path)
	if err != nil {
		t.Fatalf("merge and save: %v", err)
	}
	if merged != 0 {
		t.Errorf("expected 0 recovered (disk subset of memory), got %d", merged)
	}
	if total != 12 {
		t.Errorf("expected 12 total vectors, got %d", total)
	}

	g := NewHNSW(DefaultHNSWConfig(4))
	if err := g.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	for _, id := range []string{"fresh-1", "fresh-2", "n0", "n9"} {
		if !g.Contains(id) {
			t.Errorf("node %s missing after merge", id)
		}
	}
}

// MergeAndSave must also preserve vectors that exist only on disk (written by
// another process while this one was working).
func TestHNSWMergeAndSaveRecoversDiskOnlyNodes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "idx.hnsw")

	disk := newTestIndex(10)
	disk.Insert("gateway-1", []float32{7, 7, 7, 7})
	if err := disk.Save(path); err != nil {
		t.Fatalf("save disk: %v", err)
	}

	mem := NewHNSW(DefaultHNSWConfig(4))
	mem.Insert("local-1", []float32{1, 2, 3, 4})

	merged, total, err := mem.MergeAndSave(path)
	if err != nil {
		t.Fatalf("merge and save: %v", err)
	}
	if merged != 11 {
		t.Errorf("expected 11 nodes recovered from disk, got %d", merged)
	}
	if total != 12 {
		t.Errorf("expected 12 total vectors, got %d", total)
	}

	g := NewHNSW(DefaultHNSWConfig(4))
	if err := g.Load(path); err != nil {
		t.Fatalf("load: %v", err)
	}
	for _, id := range []string{"local-1", "gateway-1", "n0", "n9"} {
		if !g.Contains(id) {
			t.Errorf("node %s missing after merge", id)
		}
	}
}

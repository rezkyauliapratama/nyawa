package index

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func f(s string, n int) string {
	if n < 0 { return "invalid" }
	var rev [8]byte; pos := 0
	if n == 0 { rev[pos] = '0'; pos++ } else {
		for n > 0 && pos < 8 { rev[pos] = byte('0' + n%10); n /= 10; pos++ }
	}
	b := make([]byte, 0, len(s)+pos)
	for i := 0; i < len(s); i++ {
		if s[i] == '%' && i+1 < len(s) && s[i+1] == 'd' {
			for j := pos - 1; j >= 0; j-- { b = append(b, rev[j]) }
			i++
		} else { b = append(b, s[i]) }
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
	if len(results) != 3 { t.Fatalf("expected 3 results, got %d", len(results)) }
	if results[0].ID != "zero" { t.Errorf("expected 'zero' as top result, got %s", results[0].ID) }
}

func TestHNSWLargeDimensions(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(768))
	for i := 0; i < 10; i++ {
		vec := make([]float32, 768)
		for j := 0; j < 768; j++ { vec[j] = float32(i + j) }
		h.Insert(f("mem_%d", i), vec)
	}
	query := make([]float32, 768)
	for j := 0; j < 768; j++ { query[j] = float32(j) }
	results := h.Search(query, 3)
	if len(results) != 3 { t.Fatalf("expected 3 results, got %d", len(results)) }
	if results[0].ID != "mem_0" { t.Errorf("expected 'mem_0', got %s", results[0].ID) }
}

func TestHNSWDelete(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	h.Insert("a", []float32{0, 0, 0, 0})
	h.Insert("b", []float32{10, 10, 10, 10})
	if h.Size() != 2 { t.Fatalf("expected size 2, got %d", h.Size()) }
	h.Delete("a")
	if h.Size() != 1 { t.Errorf("expected size 1, got %d", h.Size()) }
	results := h.Search([]float32{0, 0, 0, 0}, 5)
	if len(results) == 0 { t.Fatal("expected results after delete") }
	if results[0].ID == "a" { t.Error("deleted element should not be returned") }
}

func TestHNSWEmptySearch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	results := h.Search([]float32{1, 2, 3, 4}, 5)
	if len(results) != 0 { t.Errorf("empty index should return 0 results, got %d", len(results)) }
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
		h.Insert(f("c%d", i), []float32{float32(i%7), float32(i%5), float32(i%3), float32(i%11)})
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

func resultIDs(res []SearchResult) []string {
	out := make([]string, len(res))
	for i, r := range res {
		out[i] = r.ID
	}
	return out
}

func TestHNSWSimilarity(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(3))
	h.Insert("A", []float32{1, 0, 0}); h.Insert("B", []float32{0.95, 0.1, 0})
	h.Insert("C", []float32{0, 1, 0}); h.Insert("D", []float32{0, 0, 1})
	results := h.Search([]float32{1, 0, 0}, 3)
	if len(results) < 2 { t.Fatal("expected at least 2 results") }
	if results[0].ID != "A" { t.Errorf("expected A first, got %s", results[0].ID) }
	if results[1].ID != "B" { t.Errorf("expected B second, got %s", results[1].ID) }
}

func TestHNSWManyInserts(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(64))
	for i := 0; i < 100; i++ {
		vec := make([]float32, 64)
		for j := 0; j < 64; j++ { vec[j] = float32(float64(i) * math.Sin(float64(j))) }
		h.Insert(f("m%d", i), vec)
	}
	if h.Size() != 100 { t.Errorf("expected 100, got %d", h.Size()) }
	q := make([]float32, 64)
	for j := 0; j < 64; j++ { q[j] = float32(math.Sin(float64(j))) }
	results := h.Search(q, 5)
	if len(results) != 5 { t.Errorf("expected 5, got %d", len(results)) }
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

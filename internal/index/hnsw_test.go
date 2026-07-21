package index

import (
	"math"
	"testing"
)

func TestHNSWInsertAndSearch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	h.Insert("zero", []float32{0, 0, 0, 0})
	h.Insert("one", []float32{1, 1, 1, 1})
	h.Insert("two", []float32{2, 2, 2, 2})
	results := h.Search([]float32{0, 0, 0, 0}, 3)
	if len(results) != 3 { t.Fatalf("expected 3 results, got %d", len(results)) }
	if results[0].ID != "zero" { t.Errorf("expected 'zero' first, got %s", results[0].ID) }
	t.Logf("Results: %+v", results)
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
	if results[0].ID != "mem_0" { t.Errorf("expected mem_0 first, got %s", results[0].ID) }
	t.Logf("Top: %s dist=%.4f", results[0].ID, results[0].Distance)
}

func TestHNSWDelete(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	h.Insert("a", []float32{0, 0, 0, 0})
	h.Insert("b", []float32{10, 10, 10, 10})
	if h.Size() != 2 { t.Fatalf("expected 2, got %d", h.Size()) }
	h.Delete("a")
	if h.Size() != 1 { t.Errorf("expected 1 after delete, got %d", h.Size()) }
	results := h.Search([]float32{0, 0, 0, 0}, 5)
	if len(results) == 0 { t.Fatal("expected results") }
}

func TestHNSWEmptySearch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(4))
	results := h.Search([]float32{1, 2, 3, 4}, 5)
	if len(results) != 0 { t.Errorf("empty should return 0, got %d", len(results)) }
}

func TestHNSWSimilarity(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(3))
	h.Insert("A", []float32{1, 0, 0})
	h.Insert("B", []float32{0.95, 0.1, 0})
	h.Insert("C", []float32{0, 1, 0})
	h.Insert("D", []float32{0, 0, 1})
	results := h.Search([]float32{1, 0, 0}, 3)
	if len(results) < 2 { t.Fatal("expected >=2 results") }
	if results[0].ID != "A" { t.Errorf("expected A first, got %s", results[0].ID) }
	if results[1].ID != "B" { t.Errorf("expected B second, got %s", results[1].ID) }
	if results[0].Distance > 0.01 { t.Errorf("A dist to self should be ~0, got %.4f", results[0].Distance) }
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
	t.Logf("Top: %s dist=%.4f", results[0].ID, results[0].Distance)
}

func f(s string, n int) string {
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

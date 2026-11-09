package store

import (
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type orderedSearchEmbedder struct {
	vectors map[string][]float32
}

func (e orderedSearchEmbedder) Embed(text string) ([]float32, error) {
	vector, ok := e.vectors[text]
	if !ok {
		return nil, fmt.Errorf("missing test vector for %q", text)
	}
	return append([]float32(nil), vector...), nil
}

func (e orderedSearchEmbedder) Available() bool { return true }
func (e orderedSearchEmbedder) Dims() int       { return 2 }

func TestVectorSearchNamespaceFilterPreservesHNSWOrder(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "memory.db")
	emb := orderedSearchEmbedder{vectors: map[string][]float32{
		"far target":    {10, 0},
		"near other":    {0, 0.1},
		"near target":   {0.01, 0},
		"middle target": {0, 1},
	}}
	st, err := NewStore(dbPath, emb)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	memories := []*types.Memory{
		{ID: "a-far", Content: "far target", Type: types.TypeFact, Namespace: "target"},
		{ID: "m-other", Content: "near other", Type: types.TypeFact, Namespace: "other"},
		{ID: "z-near", Content: "near target", Type: types.TypeFact, Namespace: "target"},
		{ID: "b-middle", Content: "middle target", Type: types.TypeFact, Namespace: "target"},
	}
	for _, memory := range memories {
		if err := st.InsertMemory(memory); err != nil {
			t.Fatalf("insert %s: %v", memory.ID, err)
		}
	}

	got, err := st.VectorSearch([]float32{0, 0}, 3, "target")
	if err != nil {
		t.Fatalf("vector search: %v", err)
	}
	want := []string{"z-near", "b-middle", "a-far"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("namespace-filtered vector order = %v, want HNSW order %v", got, want)
	}

	hits, err := st.VectorSearchHits([]float32{0, 0}, 3, "target")
	if err != nil {
		t.Fatalf("scored vector search: %v", err)
	}
	if len(hits) != len(want) {
		t.Fatalf("scored vector hit count = %d, want %d", len(hits), len(want))
	}
	wantDistances := []float64{0.01, 1, 10}
	for i, hit := range hits {
		if hit.ID != want[i] || math.Abs(hit.Distance-wantDistances[i]) > 1e-6 {
			t.Fatalf("scored hit[%d] = %+v, want ID=%s distance=%g", i, hit, want[i], wantDistances[i])
		}
	}
}

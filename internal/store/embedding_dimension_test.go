package store

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/index"
)

type dimensionTestEmbedder struct {
	dims int
}

func (e dimensionTestEmbedder) Embed(string) ([]float32, error) {
	return make([]float32, e.dims), nil
}
func (e dimensionTestEmbedder) Available() bool { return true }
func (e dimensionTestEmbedder) Dims() int       { return e.dims }

func TestNewStoreRejectsEmbedderDimensionMismatchingHNSW(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "memory.db")
	h := index.NewHNSW(index.DefaultHNSWConfig(384))
	if err := h.Insert("bge-memory", make([]float32, 384)); err != nil {
		t.Fatalf("seed HNSW: %v", err)
	}
	if err := h.Save(dbPath + ".hnsw"); err != nil {
		t.Fatalf("save HNSW: %v", err)
	}

	st, err := NewStore(dbPath, dimensionTestEmbedder{dims: 1024})
	if st != nil {
		st.Close()
		t.Fatal("NewStore() returned a store for incompatible dimensions")
	}
	if !errors.Is(err, index.ErrDimensionMismatch) {
		t.Fatalf("NewStore() error = %v, want ErrDimensionMismatch", err)
	}
	// The guard runs before migration, so it does not create or alter the DB.
	if _, statErr := os.Stat(dbPath); !os.IsNotExist(statErr) {
		t.Fatalf("database was created despite rejected configuration: %v", statErr)
	}
}

func TestNewStoreAcceptsMatchingEmbedderDimension(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "memory.db")
	h := index.NewHNSW(index.DefaultHNSWConfig(384))
	if err := h.Insert("bge-memory", make([]float32, 384)); err != nil {
		t.Fatalf("seed HNSW: %v", err)
	}
	if err := h.Save(dbPath + ".hnsw"); err != nil {
		t.Fatalf("save HNSW: %v", err)
	}

	st, err := NewStore(dbPath, dimensionTestEmbedder{dims: 384})
	if err != nil {
		t.Fatalf("NewStore() error = %v", err)
	}
	defer st.Close()
	if got, want := st.GetHNSW().Dimension(), 384; got != want {
		t.Fatalf("HNSW dimension = %d, want %d", got, want)
	}
}

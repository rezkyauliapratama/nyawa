package index

import (
	"errors"
	"path/filepath"
	"testing"
)

func TestHNSWRejectsDimensionMismatch(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(384))
	if err := h.Insert("bge", make([]float32, 384)); err != nil {
		t.Fatalf("insert matching vector: %v", err)
	}

	if err := h.EnsureDimension(1024); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("EnsureDimension(1024) error = %v, want ErrDimensionMismatch", err)
	}
	if got, want := h.Dimension(), 384; got != want {
		t.Fatalf("Dimension() after failed guard = %d, want %d", got, want)
	}
	if err := h.Insert("api", make([]float32, 1024)); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("insert mismatched vector error = %v, want ErrDimensionMismatch", err)
	}
	if got, want := h.Size(), 1; got != want {
		t.Fatalf("Size() after rejected vector = %d, want %d", got, want)
	}
	if _, err := h.SearchChecked(make([]float32, 1024), 1); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("SearchChecked mismatched query error = %v, want ErrDimensionMismatch", err)
	}
}

func TestHNSWLoadInfersLegacyDimension(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.hnsw")
	h := NewHNSW(DefaultHNSWConfig(384))
	if err := h.Insert("legacy", make([]float32, 384)); err != nil {
		t.Fatalf("insert: %v", err)
	}
	// Simulate an old index file written before Dim was persisted.
	h.config.Dim = 0
	if err := h.Save(path); err != nil {
		t.Fatalf("save legacy index: %v", err)
	}

	loaded := NewHNSW(DefaultHNSWConfig(1024))
	if err := loaded.Load(path); err != nil {
		t.Fatalf("load legacy index: %v", err)
	}
	if got, want := loaded.Dimension(), 384; got != want {
		t.Fatalf("loaded Dimension() = %d, want %d", got, want)
	}
	if err := loaded.EnsureDimension(1024); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("EnsureDimension against legacy index error = %v, want ErrDimensionMismatch", err)
	}
}

func TestHNSWRejectsMismatchWhenLegacyDimensionIsUnsetInMemory(t *testing.T) {
	h := NewHNSW(DefaultHNSWConfig(384))
	if err := h.Insert("legacy", make([]float32, 384)); err != nil {
		t.Fatalf("insert: %v", err)
	}

	// Simulate an index loaded by a pre-dimension version before the metadata
	// can be normalized. Existing vectors must still determine the contract.
	h.config.Dim = 0
	if err := h.Insert("api", make([]float32, 1024)); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("insert with implicit legacy dimension error = %v, want ErrDimensionMismatch", err)
	}
	if got, want := h.Size(), 1; got != want {
		t.Fatalf("Size() after rejected vector = %d, want %d", got, want)
	}
}

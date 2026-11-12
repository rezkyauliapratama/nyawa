package search

import (
	"path/filepath"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func newC8SREStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.NewStore(filepath.Join(t.TempDir(), "c8-sre.sqlite"), nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestC8SRESyntheticB0B1ExactIDAndAbstention(t *testing.T) {
	const token = "SYNTHC8SREID001"

	t.Run("B0 abstains when no candidate exists", func(t *testing.T) {
		s := newC8SREStore(t)
		p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

		results, err := p.Search(types.StoreQuery{QueryText: token, Namespace: "alpha", Limit: 3})
		if err != nil {
			t.Fatalf("B0 query: %v", err)
		}
		if len(results) != 0 {
			t.Fatalf("B0 returned candidates for an unanswerable query: %v", idsOf(results))
		}
	})

	t.Run("B1 returns the exact scoped identifier", func(t *testing.T) {
		s := newC8SREStore(t)
		db := s.GetDB()
		for _, fixture := range []struct {
			id, content, namespace string
		}{
			{"sre-alpha-target", "Synthetic target identifier " + token, "alpha"},
			{"sre-beta-decoy", "Synthetic decoy also mentions " + token, "beta"},
		} {
			if _, err := db.Exec(
				`INSERT INTO memories (id, content, mem_type, namespace, importance) VALUES (?, ?, ?, ?, ?)`,
				fixture.id, fixture.content, string(types.TypeNote), fixture.namespace, 0.5,
			); err != nil {
				t.Fatalf("insert synthetic fixture %q: %v", fixture.id, err)
			}
		}
		p := NewPipeline(s, embedder.NewPriorityChain(), types.SearchConfig{RRFK: 60})

		results, err := p.Search(types.StoreQuery{QueryText: token, Namespace: "alpha", Limit: 3})
		if err != nil {
			t.Fatalf("B1 scoped query: %v", err)
		}
		if len(results) != 1 || results[0].ID != "sre-alpha-target" || results[0].Namespace != "alpha" {
			t.Fatalf("B1 exact-ID results = %v; want only sre-alpha-target in alpha", idsOf(results))
		}
	})
}

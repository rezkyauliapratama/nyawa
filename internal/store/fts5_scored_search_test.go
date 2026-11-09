package store

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestFTS5SearchHitsExposeRawRankInLegacyOrder(t *testing.T) {
	t.Setenv("NYAWA_RECALL_V2", "1")
	st, err := NewStore(filepath.Join(t.TempDir(), "memory.db"), nil)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	defer st.Close()

	for _, memory := range []*types.Memory{
		{ID: "exact", Content: "quartz packet protocol detail", Type: types.TypeFact, Namespace: "scores"},
		{ID: "quartz-only", Content: "quartz archive", Type: types.TypeFact, Namespace: "scores"},
		{ID: "packet-only", Content: "packet routing", Type: types.TypeFact, Namespace: "scores"},
	} {
		if err := st.InsertMemory(memory); err != nil {
			t.Fatalf("insert %s: %v", memory.ID, err)
		}
	}

	hits, err := st.FTS5SearchHits("quartz packet", 10, "scores")
	if err != nil {
		t.Fatalf("scored FTS5 search: %v", err)
	}
	if len(hits) != 3 {
		t.Fatalf("scored FTS5 hits = %d, want 3: %+v", len(hits), hits)
	}
	if hits[0].ID != "exact" {
		t.Fatalf("best FTS5 hit = %q, want exact-match document", hits[0].ID)
	}
	for i, hit := range hits {
		if math.IsNaN(hit.RankScore) || math.IsInf(hit.RankScore, 0) {
			t.Fatalf("hit %q has non-finite FTS5 rank score %v", hit.ID, hit.RankScore)
		}
		if i > 0 && hits[i-1].RankScore > hit.RankScore {
			t.Fatalf("raw FTS5 rank scores are not in ORDER BY rank order: %+v", hits)
		}
	}

	legacyIDs, err := st.FTS5Search("quartz packet", 10, "scores")
	if err != nil {
		t.Fatalf("legacy FTS5 search: %v", err)
	}
	scoredIDs := make([]string, len(hits))
	for i, hit := range hits {
		scoredIDs[i] = hit.ID
	}
	if !reflect.DeepEqual(scoredIDs, legacyIDs) {
		t.Fatalf("scored API order %v differs from legacy order %v", scoredIDs, legacyIDs)
	}
}

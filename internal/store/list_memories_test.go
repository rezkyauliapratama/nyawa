package store

import (
	"path/filepath"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// newTestStore creates a Store backed by a throwaway SQLite file.
func newTestStore(t *testing.T) *Store {
	t.Helper()
	s, err := NewStore(filepath.Join(t.TempDir(), "test.db"), nil)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// insertAt inserts a memory and forces its created_at to ts so ordering is
// deterministic (InsertMemory always stamps created_at with "now").
func insertAt(t *testing.T, s *Store, id, ns string, memType types.MemoryType, content, ts string) {
	t.Helper()
	m := &types.Memory{ID: id, Content: content, Type: memType, Namespace: ns}
	if err := s.InsertMemory(m); err != nil {
		t.Fatalf("InsertMemory(%s): %v", id, err)
	}
	if _, err := s.db.Exec(`UPDATE memories SET created_at=? WHERE id=?`, ts, id); err != nil {
		t.Fatalf("set created_at(%s): %v", id, err)
	}
}

func TestListMemories_FiltersAndOrder(t *testing.T) {
	s := newTestStore(t)

	insertAt(t, s, "m_old", "baseline", types.TypeFact, "oldest baseline fact", "2024-01-01T00:00:00Z")
	insertAt(t, s, "m_mid", "other", types.TypeFact, "mid other fact", "2024-02-01T00:00:00Z")
	insertAt(t, s, "m_new", "baseline", types.TypeDecision, "newest baseline decision", "2024-03-01T00:00:00Z")

	tests := []struct {
		name        string
		ns          string
		memType     string
		limit       int
		newestFirst bool
		wantIDs     []string
	}{
		{
			name: "namespace filter newest first",
			ns:   "baseline", limit: 10, newestFirst: true,
			wantIDs: []string{"m_new", "m_old"},
		},
		{
			name: "namespace filter oldest first",
			ns:   "baseline", limit: 10, newestFirst: false,
			wantIDs: []string{"m_old", "m_new"},
		},
		{
			name: "type filter within namespace",
			ns:   "baseline", memType: "fact", limit: 10, newestFirst: true,
			wantIDs: []string{"m_old"},
		},
		{
			name:  "no filters returns all newest first",
			limit: 10, newestFirst: true,
			wantIDs: []string{"m_new", "m_mid", "m_old"},
		},
		{
			name: "limit caps results",
			ns:   "baseline", limit: 1, newestFirst: true,
			wantIDs: []string{"m_new"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			mems, err := s.ListMemories(tc.ns, tc.memType, tc.limit, tc.newestFirst)
			if err != nil {
				t.Fatalf("ListMemories: %v", err)
			}
			if len(mems) != len(tc.wantIDs) {
				t.Fatalf("got %d memories, want %d", len(mems), len(tc.wantIDs))
			}
			for i, want := range tc.wantIDs {
				if mems[i].ID != want {
					t.Errorf("position %d: got %q, want %q", i, mems[i].ID, want)
				}
			}
		})
	}
}

func TestListMemories_ExcludesSuperseded(t *testing.T) {
	s := newTestStore(t)
	insertAt(t, s, "m_live", "baseline", types.TypeFact, "live memory", "2024-01-01T00:00:00Z")
	insertAt(t, s, "m_dead", "baseline", types.TypeFact, "superseded memory", "2024-02-01T00:00:00Z")
	if err := s.DeleteMemory("m_dead"); err != nil {
		t.Fatalf("DeleteMemory: %v", err)
	}

	mems, err := s.ListMemories("baseline", "", 10, true)
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(mems) != 1 || mems[0].ID != "m_live" {
		t.Fatalf("expected only m_live, got %+v", mems)
	}
}

func TestListMemories_DefaultLimitAndCap(t *testing.T) {
	s := newTestStore(t)
	insertAt(t, s, "m_one", "baseline", types.TypeFact, "one", "2024-01-01T00:00:00Z")

	// limit <= 0 falls back to the default (20); more rows than exist is fine.
	mems, err := s.ListMemories("baseline", "", 0, true)
	if err != nil {
		t.Fatalf("ListMemories: %v", err)
	}
	if len(mems) != 1 {
		t.Fatalf("expected 1 memory with default limit, got %d", len(mems))
	}
}

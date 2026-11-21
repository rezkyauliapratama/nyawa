package dream

import (
	"database/sql"
	"math"
	"path/filepath"
	"testing"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const prioritySchema = `CREATE TABLE memories (
	id TEXT PRIMARY KEY, content TEXT, mem_type TEXT, namespace TEXT,
	importance REAL, access_count INTEGER, pinned INTEGER,
	created_at TEXT, updated_at TEXT, superseded_at TEXT, edge_count INTEGER
)`

func newPriorityEngine(t *testing.T) (*Engine, *sql.DB) {
	t.Helper()
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "dream.db")+"?_busy_timeout=5000")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(prioritySchema); err != nil {
		t.Fatalf("schema: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return New(db, nil, ""), db
}

func seedMemory(t *testing.T, db *sql.DB, id string, importance float64, access int, age time.Duration) {
	t.Helper()
	created := time.Now().Add(-age).UTC().Format(time.RFC3339)
	if _, err := db.Exec(`INSERT INTO memories(id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at)
		VALUES(?,?,?,?,?,?,0,?,?)`, id, id+" content", "memory", "default", importance, access, created, created); err != nil {
		t.Fatalf("seed %s: %v", id, err)
	}
}

func readImportance(t *testing.T, db *sql.DB, id string) float64 {
	t.Helper()
	var v float64
	if err := db.QueryRow(`SELECT importance FROM memories WHERE id=?`, id).Scan(&v); err != nil {
		t.Fatalf("read %s: %v", id, err)
	}
	return v
}

// The regression this file exists for: phasePriority used LEAST()/GREATEST(),
// which SQLite does not provide, and discarded the error. Both counters stayed
// zero, so importance never decayed and phaseEvict never had a candidate.
func TestPhasePriorityDecaysAndBoosts(t *testing.T) {
	engine, db := newPriorityEngine(t)
	seedMemory(t, db, "stale-unused", 0.50, 0, 10*24*time.Hour)
	seedMemory(t, db, "frequently-used", 0.50, 6, time.Hour)

	if got := engine.phasePriority(); got != 2 {
		t.Fatalf("phasePriority() = %d, want 2 (a broken statement reports 0)", got)
	}
	if got := readImportance(t, db, "stale-unused"); math.Abs(got-0.48) > 1e-9 {
		t.Errorf("stale-unused importance = %v, want 0.48", got)
	}
	if got := readImportance(t, db, "frequently-used"); math.Abs(got-0.55) > 1e-9 {
		t.Errorf("frequently-used importance = %v, want 0.55", got)
	}
}

// Rows already sitting on a clamp must not be rewritten every cycle.
func TestPhasePrioritySkipsRowsAtTheClamps(t *testing.T) {
	engine, db := newPriorityEngine(t)
	seedMemory(t, db, "ceiling", 1.0, 9, time.Hour)
	seedMemory(t, db, "floor", 0.1, 0, 10*24*time.Hour)

	if got := engine.phasePriority(); got != 0 {
		t.Fatalf("phasePriority() = %d, want 0 for rows already clamped", got)
	}
	if got := readImportance(t, db, "ceiling"); got != 1.0 {
		t.Errorf("ceiling importance = %v, want 1.0", got)
	}
	if got := readImportance(t, db, "floor"); math.Abs(got-0.1) > 1e-9 {
		t.Errorf("floor importance = %v, want 0.1", got)
	}
}

// created_at is RFC3339 ("...T09:00:00Z"); the old cutoff came from
// datetime('now','-7 days') ("... 09:00:00"). On the boundary day 'T' > ' ', so
// a memory just past seven days old was skipped instead of decayed.
func TestPhasePriorityComparesCutoffInRfc3339(t *testing.T) {
	engine, db := newPriorityEngine(t)
	seedMemory(t, db, "just-past-seven-days", 0.50, 0, 7*24*time.Hour+time.Hour)
	seedMemory(t, db, "just-under-seven-days", 0.50, 0, 6*24*time.Hour+23*time.Hour)

	engine.phasePriority()

	if got := readImportance(t, db, "just-past-seven-days"); math.Abs(got-0.48) > 1e-9 {
		t.Errorf("just-past-seven-days = %v, want 0.48 (decay must apply)", got)
	}
	if got := readImportance(t, db, "just-under-seven-days"); math.Abs(got-0.50) > 1e-9 {
		t.Errorf("just-under-seven-days = %v, want 0.50 (too young to decay)", got)
	}
}

// phaseEvict is the phase that was starved by the broken decay: it needs
// importance below the threshold, which never happened while priority was dead.
func TestPhaseEvictSupersedesStaleUnusedMemories(t *testing.T) {
	engine, db := newPriorityEngine(t)
	seedMemory(t, db, "stale-low", 0.20, 0, 120*24*time.Hour)
	seedMemory(t, db, "stale-but-used", 0.20, 5, 120*24*time.Hour)
	seedMemory(t, db, "recent-zero", 0.00, 0, 5*24*time.Hour)

	if got := engine.phaseEvict(DefaultConfig()); got != 1 {
		t.Fatalf("phaseEvict() = %d, want 1", got)
	}
	for id, wantSuperseded := range map[string]bool{
		"stale-low": true, "stale-but-used": false, "recent-zero": false,
	} {
		var superseded sql.NullString
		if err := db.QueryRow(`SELECT superseded_at FROM memories WHERE id=?`, id).Scan(&superseded); err != nil {
			t.Fatalf("read superseded %s: %v", id, err)
		}
		if superseded.Valid != wantSuperseded {
			t.Errorf("%s superseded = %v, want %v", id, superseded.Valid, wantSuperseded)
		}
	}
}

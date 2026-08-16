package store

import (
	"database/sql"
	"fmt"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/index"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Embedder interface {
	Embed(string) ([]float32, error)
	Available() bool
	Dims() int
}

type Store struct {
	db       *sql.DB
	hnsw     *index.HNSW
	hnswPath string
	graph    *graph.Store
	classify *extract.Classifier
	embedder Embedder
	ready    bool
}

func NewStore(dbPath string, emb Embedder) (*Store, error) {
	// _busy_timeout=20000: write contention (e.g. Dream Cycle holding the
	// write lock) used to fail fast at 5s. 20s gives concurrent stores a real
	// chance to wait out the lock instead of returning "database is locked".
	db, err := sql.Open("sqlite3", fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=20000&_cache_size=-8000", dbPath))
	if err != nil { return nil, fmt.Errorf("sqlite: %w", err) }
	db.SetMaxOpenConns(2); db.SetMaxIdleConns(2)
	dim := 768; if emb != nil { dim = emb.Dims() }
	s := &Store{db: db, hnsw: index.NewHNSW(index.DefaultHNSWConfig(dim)), hnswPath: dbPath + ".hnsw", embedder: emb, classify: extract.NewClassifier()}
	if gs, err := graph.NewStore(db); err == nil { s.graph = gs }
	s.hnsw.Load(s.hnswPath)
	if err := s.migrate(); err != nil { return nil, fmt.Errorf("migrate: %w", err) }
	s.ready = true; return s, nil
}

// isLocked reports whether err is a SQLite transient "database is locked"
// (SQLITE_BUSY) error that is safe to retry after a short backoff.
func isLocked(err error) bool {
	if err == nil { return false }
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "database is locked") || strings.Contains(msg, "database is busy")
}

// execRetry runs Exec, retrying transient SQLITE_BUSY errors with a short
// linear backoff (1s, 2s, 3s). The 20s busy_timeout handles lock waits at the
// driver level; this catches the residual races where the lock is re-acquired
// by another writer immediately after the timeout expires.
func (s *Store) execRetry(query string, args ...any) (sql.Result, error) {
	var res sql.Result
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		res, err = s.db.Exec(query, args...)
		if err == nil || !isLocked(err) { return res, err }
		time.Sleep(time.Duration(attempt) * time.Second)
	}
	return res, err
}

// CheckpointWAL truncates the SQLite WAL file back to zero bytes so it can
// never balloon (a 200MB WAL once slowed every write and caused "database
// locked" errors). TRUNCATE requires no active readers/writers beyond this
// connection; when the DB is busy it falls back to a PASSIVE checkpoint.
// Returns true when the WAL was fully truncated.
func (s *Store) CheckpointWAL() (bool, error) {
	var busy, logFrames, checkpointed int
	err := s.db.QueryRow(`PRAGMA wal_checkpoint(TRUNCATE)`).Scan(&busy, &logFrames, &checkpointed)
	if err == nil && busy == 0 {
		return true, nil
	}
	if err == nil && busy > 0 {
		if _, e2 := s.db.Exec(`PRAGMA wal_checkpoint(PASSIVE)`); e2 != nil {
			return false, fmt.Errorf("wal checkpoint passive: %w", e2)
		}
		return false, nil
	}
	if err != nil && isLocked(err) {
		// TRUNCATE hit a busy DB; PASSIVE never blocks, so try it once more.
		if _, e2 := s.db.Exec(`PRAGMA wal_checkpoint(PASSIVE)`); e2 != nil {
			return false, fmt.Errorf("wal checkpoint passive after busy: %w", e2)
		}
		return false, nil
	}
	return false, fmt.Errorf("wal checkpoint: %w", err)
}

func (s *Store) persistHNSW() { s.hnsw.Save(s.hnswPath) }

func (s *Store) migrate() error {
	_, err := s.db.Exec(`CREATE TABLE IF NOT EXISTS memories (
		id TEXT PRIMARY KEY, content TEXT NOT NULL, mem_type TEXT NOT NULL DEFAULT 'note',
		namespace TEXT NOT NULL DEFAULT 'default', importance REAL NOT NULL DEFAULT 0.4,
		access_count INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT (datetime('now')),
		updated_at TEXT NOT NULL DEFAULT (datetime('now')),
		superseded_at TEXT, edge_count INTEGER NOT NULL DEFAULT 0);
	CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
	CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(mem_type);
	CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
	CREATE INDEX IF NOT EXISTS idx_memories_ns_type ON memories(namespace, mem_type);
	CREATE INDEX IF NOT EXISTS idx_memories_ns_created ON memories(namespace, created_at);
	CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content,tokenize='porter unicode61', content='memories', content_rowid='rowid');
	CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid,content) VALUES(new.rowid,new.content); END;
	CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.rowid,old.content); END;
	CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.rowid,old.content); INSERT INTO memories_fts(rowid,content) VALUES(new.rowid,new.content); END;
	CREATE TABLE IF NOT EXISTS compaction_log(session_key TEXT, old_session_id TEXT, new_session_id TEXT, summary_id TEXT, created_at TEXT DEFAULT (datetime('now')));`)
	return err
}

func (s *Store) InsertMemory(m *types.Memory) error {
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.execRetry(`INSERT INTO memories(id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)`,
		m.ID, m.Content, string(m.Type), m.Namespace, m.Importance, m.AccessCount, boolToInt(m.Pinned), now, now)
	if err != nil { return fmt.Errorf("insert: %w", err) }
	if s.classify != nil {
		if m.Type == "" || m.Type == types.TypeNote { m.Type = s.classify.InferType(m.Content) }
		if s.graph != nil {
			if n, e := s.graph.InsertMemoryEntities(m.ID, s.classify.ExtractEntities(m.Content)); e == nil && n > 0 {
				s.db.Exec(`UPDATE memories SET edge_count=? WHERE id=?`, n, m.ID)
			}
			s.graph.InferTypedEdges(m.ID, m.Content)
		}
	}
	if s.embedder != nil && s.embedder.Available() {
		if v, e := s.embedder.Embed(m.Content); e == nil && len(v) > 0 {
			s.hnsw.Insert(m.ID, v); s.persistHNSW()
		}
	}
	return nil
}

func (s *Store) GetMemory(id string) (*types.Memory, error) {
	m := &types.Memory{}; var mt, cs, us string; var pi, ei int; var ss *string
	err := s.db.QueryRow(`SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id=?`, id).
		Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &m.AccessCount, &pi, &cs, &us, &ss, &ei)
	if err != nil { return nil, err }
	m.Type = types.MemoryType(mt); m.Pinned = pi != 0; m.EdgeCount = ei; m.CreatedAt, _ = parseTime(cs); m.UpdatedAt, _ = parseTime(us)
	if ss != nil { if t, e := parseTime(*ss); e == nil { m.SupersededAt = &t } }
	return m, nil
}

func (s *Store) DeleteMemory(id string) error {
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(`UPDATE memories SET superseded_at=? WHERE id=?`, now, id)
	if err == nil { s.hnsw.Delete(id); s.persistHNSW() }
	return err
}

type TimeQuery struct {
	Time  time.Time
	NS    string
	Limit int
}

func (s *Store) FTS5SearchAt(query string, tq TimeQuery) ([]string, error) {
	if tq.Limit <= 0 { tq.Limit = 10 }
	var q string
	var args []any
	if tq.Time.IsZero() {
		if tq.NS != "" {
			q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.namespace=? AND m.superseded_at IS NULL ORDER BY rank LIMIT ?`
			args = []any{query, tq.NS, tq.Limit}
		} else {
			q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.superseded_at IS NULL ORDER BY rank LIMIT ?`
			args = []any{query, tq.Limit}
		}
	} else {
		ts := tq.Time.UTC().Format(time.RFC3339)
		if tq.NS != "" {
			q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.namespace=? AND m.created_at <= ? AND (m.superseded_at IS NULL OR m.superseded_at > ?) ORDER BY rank LIMIT ?`
			args = []any{query, tq.NS, ts, ts, tq.Limit}
		} else {
			q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.created_at <= ? AND (m.superseded_at IS NULL OR m.superseded_at > ?) ORDER BY rank LIMIT ?`
			args = []any{query, ts, ts, tq.Limit}
		}
	}
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close()
	var ids []string
	for rows.Next() { var id string; rows.Scan(&id); ids = append(ids, id) }
	return ids, nil
}

func (s *Store) VectorSearchAt(q []float32, tq TimeQuery) ([]string, error) {
	if len(q) == 0 { return nil, nil }
	r := s.hnsw.Search(q, tq.Limit*3)
	if len(r) == 0 { return nil, nil }
	ids := make([]string, 0, len(r))
	for _, v := range r { ids = append(ids, v.ID) }
	if tq.NS != "" || !tq.Time.IsZero() {
		return s.filterActiveIDs(ids, tq)
	}
	return ids, nil
}

func (s *Store) filterActiveIDs(ids []string, tq TimeQuery) ([]string, error) {
	if len(ids) == 0 { return nil, nil }
	q := `SELECT id FROM memories WHERE id IN (?` + strings.Repeat(",?", len(ids)-1) + `) AND superseded_at IS NULL`
	args := make([]any, len(ids))
	for i, id := range ids { args[i] = id }
	if tq.NS != "" {
		q += ` AND namespace=?`
		args = append(args, tq.NS)
	}
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close()
	var filtered []string
	for rows.Next() { var id string; rows.Scan(&id); filtered = append(filtered, id) }
	return filtered, nil
}

func (s *Store) FTS5Search(query string, k int, ns string) ([]string, error) {
	return s.FTS5SearchAt(query, TimeQuery{NS: ns, Limit: k})
}

func (s *Store) VectorSearch(q []float32, k int, ns string) ([]string, error) {
	return s.VectorSearchAt(q, TimeQuery{NS: ns, Limit: k})
}

func (s *Store) ListNamespaces() (map[string]int, error) {
	rows, err := s.db.Query(`SELECT namespace, COUNT(*) FROM memories WHERE superseded_at IS NULL GROUP BY namespace ORDER BY namespace`)
	if err != nil { return nil, err }
	defer rows.Close()
	ns := make(map[string]int)
	for rows.Next() { var n string; var c int; rows.Scan(&n, &c); ns[n] = c }
	return ns, nil
}

func (s *Store) ArchiveSuperseded(archivePath string) (int, error) {
	rows, err := s.db.Query(`SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE superseded_at IS NOT NULL`)
	if err != nil { return 0, err }
	defer rows.Close()
	archDB, err := sql.Open("sqlite3", archivePath)
	if err != nil { return 0, fmt.Errorf("archive open: %w", err) }
	defer archDB.Close()
	archDB.Exec(`CREATE TABLE IF NOT EXISTS memories (
		id TEXT PRIMARY KEY, content TEXT, mem_type TEXT, namespace TEXT,
		importance REAL, access_count INTEGER, pinned INTEGER,
		created_at TEXT, updated_at TEXT, superseded_at TEXT, edge_count INTEGER,
		archived_at TEXT DEFAULT (datetime('now')))`)
	archDB.Exec(`CREATE TABLE IF NOT EXISTS entity_nodes (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE COLLATE NOCASE, category TEXT, created_at TEXT, access_count INTEGER)`)
	archDB.Exec(`CREATE TABLE IF NOT EXISTS entity_edges (id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT, entity_id INTEGER, weight REAL, created_at TEXT)`)
	count := 0
	tx, _ := archDB.Begin()
	stmt, _ := tx.Prepare(`INSERT OR IGNORE INTO memories(id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
	for rows.Next() {
		var id, content, mt, ns, cs, us string; var pi, ei, ac int; var ss *string; var imp float64
		rows.Scan(&id, &content, &mt, &ns, &imp, &ac, &pi, &cs, &us, &ss, &ei)
		sup := ""
		if ss != nil { sup = *ss }
		stmt.Exec(id, content, mt, ns, imp, ac, pi, cs, us, sup, ei)
		count++
	}
	stmt.Close()
	tx.Commit()
	s.db.Exec(`DELETE FROM memories WHERE superseded_at IS NOT NULL`)
	s.db.Exec(`DELETE FROM memories_fts`)
	return count, nil
}

func (s *Store) SearchByEntity(name string, limit int) ([]string, error) {
	if s.graph == nil { return nil, nil }
	return s.graph.SearchByEntityName(name, limit)
}
func (s *Store) GetRelated(memoryID string, limit int) ([]graph.RelatedMemory, error) {
	if s.graph == nil { return nil, nil }
	return s.graph.FindRelatedMemories(memoryID, limit)
}
func (s *Store) IncrementAccessCount(id string) error {
	_, err := s.db.Exec(`UPDATE memories SET access_count=access_count+1 WHERE id=?`, id)
	return err
}
func (s *Store) TraverseGraph(seedNames []string, depth, limit int) ([]graph.TraversalResult, error) {
	if s.graph == nil { return nil, nil }
	return s.graph.Traverse(seedNames, depth, limit)
}
func (s *Store) ListEntityNames(limit int) ([]string, error) {
	if limit <= 0 { limit = 10000 }
	rows, err := s.db.Query(`SELECT name FROM entity_nodes ORDER BY access_count DESC LIMIT ?`, limit)
	if err != nil { return nil, err }
	defer rows.Close()
	var names []string
	for rows.Next() { var n string; rows.Scan(&n); names = append(names, n) }
	return names, nil
}
func (s *Store) ListEntities(name, category string, limit int) ([]graph.Entity, error) {
	if s.graph == nil { return nil, nil }
	return s.graph.ListEntities(name, category, limit)
}
func (s *Store) FindGraphPath(source, target string, maxDepth int) ([]graph.PathHop, error) {
	if s.graph == nil { return nil, nil }
	return s.graph.FindPath(source, target, maxDepth)
}
func (s *Store) RebuildGraph() (graph.RebuildStats, error) {
	if s.graph == nil { return graph.RebuildStats{}, nil }
	return s.graph.RebuildGraph()
}
func (s *Store) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	if len(ids) == 0 { return nil, nil }
	q := `SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id IN (?` + strings.Repeat(",?", len(ids)-1) + `) AND superseded_at IS NULL`
	args := make([]any, len(ids))
	for i, id := range ids { args[i] = id }
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close()
	var mems []*types.Memory
	for rows.Next() {
		m := &types.Memory{}; var mt, cs, us string; var pi, ei int; var ss *string
		if err := rows.Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &m.AccessCount, &pi, &cs, &us, &ss, &ei); err != nil { return nil, err }
		m.Type = types.MemoryType(mt); m.Pinned = pi != 0; m.EdgeCount = ei; m.CreatedAt, _ = parseTime(cs); m.UpdatedAt, _ = parseTime(us)
		if ss != nil { if t, e := parseTime(*ss); e == nil { m.SupersededAt = &t } }
		mems = append(mems, m)
	}
	return mems, nil
}

func (s *Store) Stats() (map[string]any, error) {
	var t, p int
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL`).Scan(&t)
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE pinned=1 AND superseded_at IS NULL`).Scan(&p)
	var sup int
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NOT NULL`).Scan(&sup)
	nsMap, _ := s.ListNamespaces()
	en, ee := 0, 0
	if s.graph != nil { if st, e := s.graph.Stats(); e == nil { en = st["entity_nodes"].(int); ee = st["entity_edges"].(int) } }
	return map[string]any{
		"total_memories": t, "superseded": sup, "pinned_memories": p,
		"vector_indexed": s.hnsw.Size(), "entity_nodes": en, "entity_edges": ee, "namespaces": nsMap,
	}, nil
}

// LogCompaction records a compaction event in the compaction_log table.
func (s *Store) LogCompaction(sessionKey, oldSessionID, newSessionID, summaryID string) error {
	_, err := s.db.Exec(`INSERT INTO compaction_log(session_key, old_session_id, new_session_id, summary_id) VALUES(?,?,?,?)`,
		sessionKey, oldSessionID, newSessionID, summaryID)
	return err
}

func (s *Store) Close() error { return s.db.Close() }
func (s *Store) Ready() bool  { return s.ready }
func (s *Store) GetDB() *sql.DB       { return s.db }
func (s *Store) GetHNSW() *index.HNSW { return s.hnsw }
func (s *Store) GetHNSWPath() string  { return s.hnswPath }
func (s *Store) GetGraph() *graph.Store { return s.graph }
func boolToInt(b bool) int            { if b { return 1 }; return 0 }
func parseTime(s string) (time.Time, error) {
	if t, e := time.Parse(time.RFC3339, s); e == nil { return t, nil }
	if t, e := time.Parse("2006-01-02 15:04:05", s); e == nil { return t, nil }
	return time.Time{}, fmt.Errorf("bad: %s", s)
}

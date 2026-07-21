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

type Embedder interface{ Embed(string) ([]float32, error); Available() bool; Dims() int }

type Store struct{ db *sql.DB; hnsw *index.HNSW; hnswPath string; graph *graph.Store; classify *extract.Classifier; embedder Embedder; ready bool }

type TimeQuery struct{ Time time.Time; NS string; Limit int }

func NewStore(dbPath string, emb Embedder) (*Store, error) {
	db, err := sql.Open("sqlite3", fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000&_cache_size=-8000", dbPath))
	if err != nil { return nil, fmt.Errorf("sqlite: %w", err) }
	db.SetMaxOpenConns(1); db.SetMaxIdleConns(1)
	dim := 768; if emb != nil { dim = emb.Dims() }
	s := &Store{db: db, hnsw: index.NewHNSW(index.DefaultHNSWConfig(dim)), hnswPath: dbPath + ".hnsw", embedder: emb, classify: extract.NewClassifier()}
	if gs, err := graph.NewStore(db); err == nil { s.graph = gs }
	s.hnsw.Load(s.hnswPath)
	if err := s.migrate(); err != nil { return nil, fmt.Errorf("migrate: %w", err) }
	s.ready = true; return s, nil
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
	CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content,tokenize='porter unicode61', content='memories', content_rowid='rowid');
	CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid,content) VALUES(new.rowid,new.content); END;
	CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.rowid,old.content); END;
	CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN INSERT INTO memories_fts(memories_fts,rowid,content) VALUES('delete',old.rowid,old.content); INSERT INTO memories_fts(rowid,content) VALUES(new.rowid,new.content); END;`)
	return err
}
func (s *Store) InsertMemory(m *types.Memory) error {
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(`INSERT INTO memories(id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)`,
		m.ID, m.Content, string(m.Type), m.Namespace, m.Importance, m.AccessCount, boolToInt(m.Pinned), now, now)
	if err != nil { return fmt.Errorf("insert: %w", err) }
	if s.classify != nil {
		if m.Type == "" || m.Type == types.TypeNote { m.Type = s.classify.InferType(m.Content) }
		if s.graph != nil { if n, e := s.graph.InsertMemoryEntities(m.ID, s.classify.ExtractEntities(m.Content)); e == nil && n > 0 { s.db.Exec(`UPDATE memories SET edge_count=? WHERE id=?`, n, m.ID) } }
	}
	if s.embedder != nil && s.embedder.Available() { if v, e := s.embedder.Embed(m.Content); e == nil && len(v) > 0 { s.hnsw.Insert(m.ID, v); s.persistHNSW() } }
	return nil
}
func (s *Store) GetMemory(id string) (*types.Memory, error) {
	m := &types.Memory{}; var mt, cs, us string; var pi, ei int; var ss *string
	err := s.db.QueryRow(`SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id=?`, id).Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &m.AccessCount, &pi, &cs, &us, &ss, &ei)
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
func (s *Store) FTS5SearchAt(query string, tq TimeQuery) ([]string, error) {
	if tq.Limit <= 0 { tq.Limit = 10 }
	var q string; var args []any
	if tq.Time.IsZero() {
		if tq.NS != "" { q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.namespace=? AND m.superseded_at IS NULL ORDER BY rank LIMIT ?`; args = []any{query, tq.NS, tq.Limit}
		} else { q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.superseded_at IS NULL ORDER BY rank LIMIT ?`; args = []any{query, tq.Limit} }
	} else {
		ts := tq.Time.UTC().Format(time.RFC3339)
		if tq.NS != "" { q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.namespace=? AND m.created_at<=? AND (m.superseded_at IS NULL OR m.superseded_at>?) ORDER BY rank LIMIT ?`; args = []any{query, tq.NS, ts, ts, tq.Limit}
		} else { q = `SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.created_at<=? AND (m.superseded_at IS NULL OR m.superseded_at>?) ORDER BY rank LIMIT ?`; args = []any{query, ts, ts, tq.Limit} }
	}
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close(); var ids []string
	for rows.Next() { var id string; rows.Scan(&id); ids = append(ids, id) }
	return ids, nil
}
func (s *Store) VectorSearchAt(q []float32, tq TimeQuery) ([]string, error) {
	if len(q) == 0 { return nil, nil }
	r := s.hnsw.Search(q, tq.Limit*3); ids := make([]string, len(r))
	for i, v := range r { ids[i] = v.ID }
	if tq.NS != "" || !tq.Time.IsZero() { return s.filterActiveIDs(ids, tq) }
	return ids, nil
}
func (s *Store) filterActiveIDs(ids []string, tq TimeQuery) ([]string, error) {
	if len(ids) == 0 { return nil, nil }
	q := `SELECT id FROM memories WHERE id IN (?` + strings.Repeat(",?", len(ids)-1) + `) AND superseded_at IS NULL`
	args := make([]any, len(ids)); for i, id := range ids { args[i] = id }
	if tq.NS != "" { q += ` AND namespace=?`; args = append(args, tq.NS) }
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close(); var filtered []string
	for rows.Next() { var id string; rows.Scan(&id); filtered = append(filtered, id) }
	return filtered, nil
}
func (s *Store) FTS5Search(query string, k int, ns string) ([]string, error) { return s.FTS5SearchAt(query, TimeQuery{NS: ns, Limit: k}) }
func (s *Store) VectorSearch(q []float32, k int, ns string) ([]string, error) { return s.VectorSearchAt(q, TimeQuery{NS: ns, Limit: k}) }
func (s *Store) ListNamespaces() (map[string]int, error) {
	rows, err := s.db.Query(`SELECT namespace,COUNT(*) FROM memories WHERE superseded_at IS NULL GROUP BY namespace ORDER BY namespace`)
	if err != nil { return nil, err }
	defer rows.Close(); ns := map[string]int{}
	for rows.Next() { var n string; var c int; rows.Scan(&n, &c); ns[n] = c }
	return ns, nil
}
func (s *Store) ArchiveSuperseded(archivePath string) (int, error) {
	rows, err := s.db.Query(`SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE superseded_at IS NOT NULL`)
	if err != nil { return 0, err }
	defer rows.Close()
	a, _ := sql.Open("sqlite3", archivePath); defer a.Close()
	a.Exec(`CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY,content TEXT,mem_type TEXT,namespace TEXT,importance REAL,access_count INTEGER,pinned INTEGER,created_at TEXT,updated_at TEXT,superseded_at TEXT,edge_count INTEGER,archived_at TEXT DEFAULT(datetime('now')))`)
	tx, _ := a.Begin(); st, _ := tx.Prepare(`INSERT OR IGNORE INTO memories(id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count) VALUES(?,?,?,?,?,?,?,?,?,?,?)`)
	c := 0
	for rows.Next() {
		var id, ct, mt, ns, cs, us string; var pi, ei, ac int; var ss *string; var imp float64
		rows.Scan(&id, &ct, &mt, &ns, &imp, &ac, &pi, &cs, &us, &ss, &ei)
		sup := ""; if ss != nil { sup = *ss }
		st.Exec(id, ct, mt, ns, imp, ac, pi, cs, us, sup, ei); c++
	}
	st.Close(); tx.Commit()
	s.db.Exec(`DELETE FROM memories WHERE superseded_at IS NOT NULL`)
	return c, nil
}
func (s *Store) SearchByEntity(name string, limit int) ([]string, error) { if s.graph == nil { return nil, nil }; return s.graph.SearchByEntityName(name, limit) }
func (s *Store) GetRelated(memoryID string, limit int) ([]graph.RelatedMemory, error) { if s.graph == nil { return nil, nil }; return s.graph.FindRelatedMemories(memoryID, limit) }
func (s *Store) IncrementAccessCount(id string) error { _, err := s.db.Exec(`UPDATE memories SET access_count=access_count+1 WHERE id=?`, id); return err }
func (s *Store) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	if len(ids) == 0 { return nil, nil }
	q := `SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id IN(?` + strings.Repeat(",?", len(ids)-1) + `) AND superseded_at IS NULL`
	args := make([]any, len(ids)); for i, id := range ids { args[i] = id }
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close(); var mems []*types.Memory
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
	var t, p, sup int
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL`).Scan(&t)
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE pinned=1 AND superseded_at IS NULL`).Scan(&p)
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NOT NULL`).Scan(&sup)
	nm, _ := s.ListNamespaces()
	en, ee := 0, 0
	if s.graph != nil { if st, e := s.graph.Stats(); e == nil { en = st["entity_nodes"].(int); ee = st["entity_edges"].(int) } }
	return map[string]any{"total_memories": t, "superseded": sup, "pinned_memories": p, "vector_indexed": s.hnsw.Size(), "entity_nodes": en, "entity_edges": ee, "namespaces": nm}, nil
}
func (s *Store) Close() error { return s.db.Close() }
func (s *Store) Ready() bool { return s.ready }
func (s *Store) GetDB() *sql.DB { return s.db }
func (s *Store) GetHNSW() *index.HNSW { return s.hnsw }
func (s *Store) GetHNSWPath() string { return s.hnswPath }
func boolToInt(b bool) int { if b { return 1 }; return 0 }
func parseTime(s string) (time.Time, error) {
	if t, e := time.Parse(time.RFC3339, s); e == nil { return t, nil }
	if t, e := time.Parse("2006-01-02 15:04:05", s); e == nil { return t, nil }
	return time.Time{}, fmt.Errorf("bad: %s", s)
}

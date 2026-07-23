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
func (s *Store) UpdateMemory(m *types.Memory) error {
	now := time.Now().UTC().Format(time.RFC3339)
	result, err := s.db.Exec(`UPDATE memories SET content=?, mem_type=?, namespace=?, importance=?, updated_at=? WHERE id=? AND superseded_at IS NULL`,
		m.Content, string(m.Type), m.Namespace, m.Importance, now, m.ID)
	if err != nil { return fmt.Errorf("update: %w", err) }
	n, _ := result.RowsAffected()
	if n == 0 { return fmt.Errorf("memory %s not found or already deleted", m.ID) }
	s.hnsw.Delete(m.ID)
	if s.embedder != nil && s.embedder.Available() {
		if v, e := s.embedder.Embed(m.Content); e == nil && len(v) > 0 {
			s.hnsw.Insert(m.ID, v)
			s.persistHNSW()
		}
	}
	return nil
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

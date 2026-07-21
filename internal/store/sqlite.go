package store

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/rezkyauliapratama/nyawa/internal/index"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Embedder interface {
	Embed(text string) ([]float32, error)
	Available() bool
	Dims() int
}

type Store struct {
	db       *sql.DB
	hnsw     *index.HNSW
	hnswPath string
	embedder Embedder
	ready    bool
}

func NewStore(dbPath string, emb Embedder) (*Store, error) {
	db, err := sql.Open("sqlite3", fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000&_cache_size=-8000", dbPath))
	if err != nil { return nil, fmt.Errorf("sqlite open: %w", err) }
	db.SetMaxOpenConns(1); db.SetMaxIdleConns(1)

	dim := 768
	if emb != nil { dim = emb.Dims() }

	hnswPath := dbPath + ".hnsw"
	s := &Store{db: db, hnsw: index.NewHNSW(index.DefaultHNSWConfig(dim)), hnswPath: hnswPath, embedder: emb}

	// Load persisted HNSW index if exists
	s.hnsw.Load(hnswPath)

	if err := s.migrate(); err != nil { return nil, fmt.Errorf("migrate: %w", err) }
	s.ready = true
	return s, nil
}

func (s *Store) persistHNSW() {
	if err := s.hnsw.Save(s.hnswPath); err != nil {
		// Non-fatal: HNSW index can be rebuilt
	}
}

func (s *Store) migrate() error {
	_, err := s.db.Exec(`
	CREATE TABLE IF NOT EXISTS memories (
		id TEXT PRIMARY KEY, content TEXT NOT NULL, mem_type TEXT NOT NULL DEFAULT 'note',
		namespace TEXT NOT NULL DEFAULT 'default', importance REAL NOT NULL DEFAULT 0.4,
		access_count INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT (datetime('now')),
		updated_at TEXT NOT NULL DEFAULT (datetime('now')),
		superseded_at TEXT, edge_count INTEGER NOT NULL DEFAULT 0
	);
	CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
	CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(mem_type);
	CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
	CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, tokenize='porter unicode61');
	CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
		INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
	END;
	CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
		DELETE FROM memories_fts WHERE rowid = old.rowid;
	END;
	CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
		DELETE FROM memories_fts WHERE rowid = old.rowid;
		INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
	END;
	`)
	return err
}

func (s *Store) InsertMemory(m *types.Memory) error {
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(`INSERT INTO memories (id, content, mem_type, namespace, importance, access_count, pinned, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		m.ID, m.Content, string(m.Type), m.Namespace, m.Importance, m.AccessCount, boolToInt(m.Pinned), now, now)
	if err != nil { return fmt.Errorf("insert: %w", err) }
	if s.embedder != nil && s.embedder.Available() {
		if vec, err := s.embedder.Embed(m.Content); err == nil && len(vec) > 0 {
			s.hnsw.Insert(m.ID, vec)
			s.persistHNSW()
		}
	}
	return nil
}

func (s *Store) GetMemory(id string) (*types.Memory, error) {
	m := &types.Memory{}
	var mt, cs, us string; var pi int; var ss *string
	err := s.db.QueryRow(`SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id=?`, id).
		Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &m.AccessCount, &pi, &cs, &us, &ss, &m.EdgeCount)
	if err != nil { return nil, err }
	m.Type = types.MemoryType(mt); m.Pinned = pi != 0
	m.CreatedAt, _ = parseTime(cs); m.UpdatedAt, _ = parseTime(us)
	if ss != nil { if t, err := parseTime(*ss); err == nil { m.SupersededAt = &t } }
	return m, nil
}

func (s *Store) DeleteMemory(id string) error {
	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(`UPDATE memories SET superseded_at=? WHERE id=?`, now, id)
	if err == nil { s.hnsw.Delete(id); s.persistHNSW() }
	return err
}

func (s *Store) IncrementAccessCount(id string) error {
	_, err := s.db.Exec(`UPDATE memories SET access_count=access_count+1 WHERE id=?`, id)
	return err
}

func (s *Store) VectorSearch(queryVector []float32, topK int, ns string) ([]string, error) {
	if len(queryVector) == 0 { return nil, nil }
	r := s.hnsw.Search(queryVector, topK*3)
	if len(r) == 0 { return nil, nil }
	ids := make([]string, len(r))
	for i, v := range r { ids[i] = v.ID }
	return ids, nil
}

func (s *Store) FTS5Search(query string, topK int, ns string) ([]string, error) {
	rows, err := s.db.Query(`SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND (m.namespace=? OR ?='') AND m.superseded_at IS NULL ORDER BY rank LIMIT ?`, query, ns, ns, topK)
	if err != nil { return nil, err }
	defer rows.Close()
	var ids []string
	for rows.Next() { var id string; rows.Scan(&id); ids = append(ids, id) }
	return ids, nil
}

func (s *Store) GetMemoriesByIDs(ids []string) ([]*types.Memory, error) {
	if len(ids) == 0 { return nil, nil }
	q := `SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE id IN (?`
	args := []any{ids[0]}
	for _, id := range ids[1:] { q += ",?"; args = append(args, id) }
	q += `)`
	rows, err := s.db.Query(q, args...)
	if err != nil { return nil, err }
	defer rows.Close()
	var mems []*types.Memory
	for rows.Next() {
		m := &types.Memory{}; var mt, cs, us string; var pi int; var ss *string
		if err := rows.Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &m.AccessCount, &pi, &cs, &us, &ss, &m.EdgeCount); err != nil { return nil, err }
		m.Type = types.MemoryType(mt); m.Pinned = pi != 0
		m.CreatedAt, _ = parseTime(cs); m.UpdatedAt, _ = parseTime(us)
		if ss != nil { if t, err := parseTime(*ss); err == nil { m.SupersededAt = &t } }
		mems = append(mems, m)
	}
	return mems, nil
}

func (s *Store) Stats() (map[string]any, error) {
	var total, pinned int
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL`).Scan(&total)
	s.db.QueryRow(`SELECT COUNT(*) FROM memories WHERE pinned=1 AND superseded_at IS NULL`).Scan(&pinned)
	return map[string]any{"total_memories": total, "pinned_memories": pinned, "vector_indexed": s.hnsw.Size()}, nil
}

func (s *Store) Close() error { return s.db.Close() }
func (s *Store) Ready() bool  { return s.ready }
func boolToInt(b bool) int    { if b { return 1 }; return 0 }
func parseTime(s string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339, s); err == nil { return t, nil }
	if t, err := time.Parse("2006-01-02 15:04:05", s); err == nil { return t, nil }
	return time.Time{}, fmt.Errorf("bad: %s", s)
}

package rag

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/rezkyauliapratama/nyawa/internal/index"
)

var (
	ErrRAGEmbedderUnavailable = errors.New("RAG embedder unavailable")
	ErrRAGIndexUnavailable    = errors.New("RAG vector index unavailable")
)

type Collection struct {
	ID          int       `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	ChunkSize   int       `json:"chunk_size"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
	DocCount    int       `json:"doc_count"`
}

type Document struct {
	ID                 string          `json:"id"`
	Collection         int             `json:"collection_id"`
	Filename           string          `json:"filename"`
	SourceType         string          `json:"source_type"`
	Metadata           json.RawMessage `json:"metadata,omitempty"`
	ChunkCount         int             `json:"chunk_count"`
	VectorIndexedCount int             `json:"vector_indexed_chunks"`
	CreatedAt          time.Time       `json:"created_at"`
}

type Chunk struct {
	ID         string          `json:"id"`
	DocumentID string          `json:"document_id"`
	Content    string          `json:"content"`
	ChunkIndex int             `json:"chunk_index"`
	Metadata   json.RawMessage `json:"metadata,omitempty"`
}

type Embedder interface {
	Embed(string) ([]float32, error)
	Available() bool
	Dims() int
}

type retrievalTaskEmbedder interface {
	EmbedQuery(string) ([]float32, error)
	EmbedPassage(string) ([]float32, error)
}

func embedRAGPassage(emb Embedder, text string) ([]float32, error) {
	if taskAware, ok := emb.(retrievalTaskEmbedder); ok {
		return taskAware.EmbedPassage(text)
	}
	return emb.Embed(text)
}

func embedRAGQuery(emb Embedder, text string) ([]float32, error) {
	if taskAware, ok := emb.(retrievalTaskEmbedder); ok {
		return taskAware.EmbedQuery(text)
	}
	return emb.Embed(text)
}

type RAGStore struct {
	db       *sql.DB
	hnsw     *index.HNSW
	hnswPath string
	embedder Embedder
	initErr  error
	mu       sync.RWMutex
}

func NewRAGStore(db *sql.DB, hnsw *index.HNSW, hnswPath string, emb Embedder) *RAGStore {
	r := &RAGStore{db: db, hnsw: hnsw, hnswPath: hnswPath, embedder: emb}
	r.initErr = r.migrate()
	return r
}

// NewUnavailableRAGStore preserves memory/MCP service availability when the
// RAG-only configuration or its independent index cannot be initialized.
func NewUnavailableRAGStore(db *sql.DB, err error) *RAGStore {
	r := NewRAGStore(db, nil, "", nil)
	if r.initErr != nil {
		r.initErr = errors.Join(r.initErr, err)
	} else {
		r.initErr = err
	}
	return r
}

func (r *RAGStore) InitError() error { return r.initErr }

func (r *RAGStore) migrate() error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS rag_collections (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL UNIQUE,
			description TEXT DEFAULT '',
			chunk_size INTEGER NOT NULL DEFAULT 500,
			created_at TEXT NOT NULL DEFAULT (datetime('now')),
			updated_at TEXT NOT NULL DEFAULT (datetime('now')))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_collections_name ON rag_collections(name)`,
		`CREATE TABLE IF NOT EXISTS rag_documents (
			id TEXT PRIMARY KEY,
			collection_id INTEGER NOT NULL REFERENCES rag_collections(id),
			filename TEXT NOT NULL,
			source_type TEXT NOT NULL DEFAULT 'file',
			metadata TEXT DEFAULT '{}',
			chunk_count INTEGER NOT NULL DEFAULT 0,
			created_at TEXT NOT NULL DEFAULT (datetime('now')))`,
		`CREATE INDEX IF NOT EXISTS idx_rag_documents_collection ON rag_documents(collection_id)`,
		`CREATE TABLE IF NOT EXISTS rag_chunks (
			id TEXT PRIMARY KEY,
			document_id TEXT NOT NULL REFERENCES rag_documents(id),
			content TEXT NOT NULL,
			chunk_index INTEGER NOT NULL DEFAULT 0,
			metadata TEXT DEFAULT '{}',
			vector_indexed INTEGER NOT NULL DEFAULT 0)`,
		`CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id)`,
	}
	for _, statement := range statements {
		if _, err := r.db.Exec(statement); err != nil {
			return fmt.Errorf("migrate RAG schema: %w", err)
		}
	}
	if err := ensureRAGChunkVectorColumn(r.db); err != nil {
		return fmt.Errorf("migrate RAG vector state: %w", err)
	}
	if err := ensureRAGFTS(r.db); err != nil {
		return err
	}
	return nil
}

func ensureRAGChunkVectorColumn(db *sql.DB) error {
	rows, err := db.Query(`PRAGMA table_info(rag_chunks)`)
	if err != nil {
		return err
	}
	found := false
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			rows.Close()
			return err
		}
		if name == "vector_indexed" {
			found = true
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	if err := rows.Close(); err != nil {
		return err
	}
	if found {
		return nil
	}
	if _, err := db.Exec(`ALTER TABLE rag_chunks ADD COLUMN vector_indexed INTEGER NOT NULL DEFAULT 0`); err == nil {
		return nil
	} else {
		// Another MCP process may have completed the additive migration after
		// this process inspected table_info. Treat that race as success only if
		// a fresh read proves the column now exists.
		rows, checkErr := db.Query(`PRAGMA table_info(rag_chunks)`)
		if checkErr != nil {
			return fmt.Errorf("alter rag_chunks and recheck column: %w (alter: %v)", checkErr, err)
		}
		defer rows.Close()
		for rows.Next() {
			var cid, notNull, primaryKey int
			var name, columnType string
			var defaultValue any
			if scanErr := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); scanErr != nil {
				return scanErr
			}
			if name == "vector_indexed" {
				return nil
			}
		}
		if rowsErr := rows.Err(); rowsErr != nil {
			return rowsErr
		}
		return err
	}
}

var (
	ftsTerms                 = regexp.MustCompile(`[\pL\pN]+`)
	ragChunkIDTerm           = regexp.MustCompile(`(?:^|[^[:alnum:]_-])(rag_chk_[[:alnum:]_-]+)(?:$|[^[:alnum:]_-])`)
	structuredIdentifierTerm = regexp.MustCompile(`[[:alnum:]]+([-_][[:alnum:]]+)+`)
	shaLikeIdentifierTerm    = regexp.MustCompile(`(?i)\b[0-9a-f]{32,}\b`)
	urlIdentifierTerm        = regexp.MustCompile(`(?i)https?://[^\s<>\x22']+`)
	pathIdentifierTerm       = regexp.MustCompile(`(?:[/\\]?[[:alnum:]_.~+-]+[/\\])+[[:alnum:]_.~+-]+(?:[/\\][[:alnum:]_.~+-]+)*`)
)

func internalRAGChunkIDs(query string) []string {
	matches := ragChunkIDTerm.FindAllStringSubmatch(query, -1)
	ids := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			ids = append(ids, match[1])
		}
	}
	return ids
}

type contentIdentifier struct {
	kind  string
	value string
}

func trimIdentifierPunctuation(value string) string {
	return strings.TrimRight(value, `.,;:!?)]}`)
}

func normalizeURLIdentifier(value string) string {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Hostname() == "" {
		return value
	}
	parsed.Scheme = strings.ToLower(parsed.Scheme)
	hostname := strings.ToLower(parsed.Hostname())
	host := hostname
	if strings.Contains(hostname, ":") {
		host = "[" + hostname + "]"
	}
	if port := parsed.Port(); port != "" {
		host = net.JoinHostPort(hostname, port)
	}
	if parsed.User != nil {
		host = parsed.User.String() + "@" + host
	}
	parsed.Host = host
	return parsed.String()
}

// contentIdentifiers extracts whole identifiers with type-specific comparison:
// structured IDs and hexadecimal hashes are case-insensitive; URLs normalize
// scheme/host only, and path values retain case throughout.
func contentIdentifiers(text string) []contentIdentifier {
	var identifiers []contentIdentifier
	remaining := text

	for _, value := range urlIdentifierTerm.FindAllString(text, -1) {
		value = trimIdentifierPunctuation(value)
		if value != "" {
			identifiers = append(identifiers, contentIdentifier{kind: "url", value: normalizeURLIdentifier(value)})
		}
	}
	remaining = urlIdentifierTerm.ReplaceAllString(remaining, " ")

	for _, value := range pathIdentifierTerm.FindAllString(remaining, -1) {
		value = trimIdentifierPunctuation(value)
		if value != "" {
			identifiers = append(identifiers, contentIdentifier{kind: "path", value: value})
		}
	}
	remaining = pathIdentifierTerm.ReplaceAllString(remaining, " ")

	for _, value := range shaLikeIdentifierTerm.FindAllString(remaining, -1) {
		identifiers = append(identifiers, contentIdentifier{kind: "hash", value: strings.ToLower(value)})
	}
	remaining = shaLikeIdentifierTerm.ReplaceAllString(remaining, " ")

	for _, value := range structuredIdentifierTerm.FindAllString(remaining, -1) {
		var hasLetter, hasDigit bool
		for _, r := range value {
			if unicode.IsLetter(r) {
				hasLetter = true
			}
			if unicode.IsDigit(r) {
				hasDigit = true
			}
		}
		if hasLetter && hasDigit {
			identifiers = append(identifiers, contentIdentifier{kind: "structured", value: strings.ToLower(value)})
		}
	}
	return identifiers
}

func containsExactContentIdentifier(content string, want contentIdentifier) bool {
	for _, got := range contentIdentifiers(content) {
		if got.kind == want.kind && got.value == want.value {
			return true
		}
	}
	return false
}

// safeFTSQuery quotes every user term so FTS operators and syntax remain data.
func safeFTSQuery(query string) string {
	terms := ftsTerms.FindAllString(query, -1)
	quoted := make([]string, 0, len(terms))
	for _, term := range terms {
		quoted = append(quoted, `"`+strings.ReplaceAll(term, `"`, `""`)+`"`)
	}
	return strings.Join(quoted, " OR ")
}

const ragFTSMigrationVersion = 1

func ragFTSSchemaReady(q interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}) (bool, error) {
	ctx := context.Background()
	var migrationTable int
	if err := q.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rag_schema_migrations'`).Scan(&migrationTable); err != nil {
		return false, err
	}
	if migrationTable == 0 {
		return false, nil
	}
	var version int
	err := q.QueryRowContext(ctx, `SELECT version FROM rag_schema_migrations WHERE name='rag_chunks_fts'`).Scan(&version)
	if errors.Is(err, sql.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if version > ragFTSMigrationVersion {
		return false, fmt.Errorf("RAG FTS migration version %d is newer than supported version %d", version, ragFTSMigrationVersion)
	}
	if version != ragFTSMigrationVersion {
		return false, nil
	}
	var objectCount int
	if err := q.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE
		(type='table' AND name='rag_chunks_fts') OR
		(type='trigger' AND name IN ('rag_chunks_fts_insert','rag_chunks_fts_update','rag_chunks_fts_delete'))`).Scan(&objectCount); err != nil {
		return false, err
	}
	return objectCount == 4, nil
}

const ragFTSBusyTimeoutMillis = 5000

func ensureRAGFTS(db *sql.DB) error {
	// The common startup path is read-only: do not open a write transaction or
	// touch the FTS index when its versioned migration and triggers are intact.
	ready, err := ragFTSSchemaReady(db)
	if err != nil {
		return fmt.Errorf("inspect RAG FTS5 migration: %w", err)
	}
	if ready {
		return nil
	}

	// Acquire and retain one physical connection so the write reservation and
	// every migration statement run on the same SQLite connection. BEGIN IMMEDIATE
	// serializes competing initializers before either can inspect-and-upgrade.
	ctx := context.Background()
	conn, err := db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("acquire connection for RAG FTS5 migration: %w", err)
	}
	defer conn.Close()
	if _, err := conn.ExecContext(ctx, fmt.Sprintf(`PRAGMA busy_timeout = %d`, ragFTSBusyTimeoutMillis)); err != nil {
		return fmt.Errorf("configure RAG FTS5 migration busy timeout: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `BEGIN IMMEDIATE`); err != nil {
		return fmt.Errorf("begin immediate RAG FTS5 migration: %w", err)
	}
	inTransaction := true
	defer func() {
		if inTransaction {
			_, _ = conn.ExecContext(ctx, `ROLLBACK`)
		}
	}()

	// The marker may have been written while this initializer waited for the
	// reservation. Recheck only after acquiring it, before issuing any DDL.
	ready, err = ragFTSSchemaReady(conn)
	if err != nil {
		return fmt.Errorf("recheck RAG FTS5 migration: %w", err)
	}
	if ready {
		return nil
	}

	if _, err := conn.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS rag_schema_migrations (
		name TEXT PRIMARY KEY,
		version INTEGER NOT NULL,
		applied_at TEXT NOT NULL DEFAULT (datetime('now')))`); err != nil {
		return fmt.Errorf("create RAG migration ledger: %w", err)
	}
	statements := []string{
		`CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(content, id, tokenize='unicode61')`,
		`CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_insert AFTER INSERT ON rag_chunks BEGIN
			INSERT INTO rag_chunks_fts(rowid,content,id) VALUES(new.rowid,new.content,new.id);
		END`,
		`CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_delete AFTER DELETE ON rag_chunks BEGIN
			DELETE FROM rag_chunks_fts WHERE rowid=old.rowid;
		END`,
		`CREATE TRIGGER IF NOT EXISTS rag_chunks_fts_update AFTER UPDATE OF content,id ON rag_chunks BEGIN
			DELETE FROM rag_chunks_fts WHERE rowid=old.rowid;
			INSERT INTO rag_chunks_fts(rowid,content,id) VALUES(new.rowid,new.content,new.id);
		END`,
	}
	for _, statement := range statements {
		if _, err := conn.ExecContext(ctx, statement); err != nil {
			return fmt.Errorf("initialize RAG FTS5 index: %w", err)
		}
	}

	// Only the initial/version-upgrade or an incomplete schema performs a
	// transactional reconcile. This recovers legacy/partial initialization,
	// removes stale entries, and records completion atomically with the backfill.
	if _, err := conn.ExecContext(ctx, `DELETE FROM rag_chunks_fts`); err != nil {
		return fmt.Errorf("clear RAG FTS5 index before backfill: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `INSERT INTO rag_chunks_fts(rowid,content,id) SELECT rowid,content,id FROM rag_chunks`); err != nil {
		return fmt.Errorf("backfill RAG FTS5 index: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `INSERT INTO rag_schema_migrations(name,version) VALUES('rag_chunks_fts',?)
		ON CONFLICT(name) DO UPDATE SET version=excluded.version,applied_at=datetime('now')`, ragFTSMigrationVersion); err != nil {
		return fmt.Errorf("record RAG FTS5 migration: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `COMMIT`); err != nil {
		return fmt.Errorf("commit RAG FTS5 migration: %w", err)
	}
	inTransaction = false
	return nil
}

type retrievalCandidate struct {
	result RAGResult
	score  float64
	exact  bool
}

func (r *RAGStore) lexicalCandidates(query, collectionName string, limit int) ([]retrievalCandidate, error) {
	if limit <= 0 {
		return nil, nil
	}
	byID := make(map[string]retrievalCandidate)
	for _, id := range internalRAGChunkIDs(query) {
		var candidate retrievalCandidate
		err := r.db.QueryRow(`SELECT c.id,c.content,c.chunk_index,d.filename FROM rag_chunks c
			JOIN rag_documents d ON d.id=c.document_id JOIN rag_collections col ON col.id=d.collection_id
			WHERE c.id=? AND (?='' OR col.name=?)`, id, collectionName, collectionName).Scan(
			&candidate.result.ChunkID, &candidate.result.Content, &candidate.result.ChunkIdx, &candidate.result.Document)
		if err != nil && !errors.Is(err, sql.ErrNoRows) {
			return nil, fmt.Errorf("load exact RAG chunk %q: %w", id, err)
		}
		if err == nil {
			candidate.exact = true
			byID[id] = candidate
		}
	}
	// Content identifiers need a whole-value comparison, not token overlap.
	// FTS narrows rows, then type-aware extraction confirms exact equality.
	for _, identifier := range contentIdentifiers(query) {
		matchID := safeFTSQuery(identifier.value)
		if matchID == "" {
			continue
		}
		rows, err := r.db.Query(`SELECT c.id,c.content,c.chunk_index,d.filename FROM rag_chunks_fts f
			JOIN rag_chunks c ON c.id=f.id JOIN rag_documents d ON d.id=c.document_id
			JOIN rag_collections col ON col.id=d.collection_id
			WHERE rag_chunks_fts MATCH ? AND (?='' OR col.name=?)`,
			matchID, collectionName, collectionName)
		if err != nil {
			return nil, fmt.Errorf("RAG exact-ID FTS5 query: %w", err)
		}
		for rows.Next() {
			var candidate retrievalCandidate
			if err := rows.Scan(&candidate.result.ChunkID, &candidate.result.Content, &candidate.result.ChunkIdx, &candidate.result.Document); err != nil {
				rows.Close()
				return nil, err
			}
			if !containsExactContentIdentifier(candidate.result.Content, identifier) {
				continue
			}
			candidate.exact = true
			prior := byID[candidate.result.ChunkID]
			prior.result = candidate.result
			prior.exact = true
			byID[candidate.result.ChunkID] = prior
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, err
		}
		if err := rows.Close(); err != nil {
			return nil, err
		}
	}
	match := safeFTSQuery(query)
	if match != "" {
		rows, err := r.db.Query(`SELECT c.id,c.content,c.chunk_index,d.filename FROM rag_chunks_fts f
			JOIN rag_chunks c ON c.id=f.id JOIN rag_documents d ON d.id=c.document_id
			JOIN rag_collections col ON col.id=d.collection_id
			WHERE rag_chunks_fts MATCH ? AND (?='' OR col.name=?)
			ORDER BY bm25(rag_chunks_fts),c.id LIMIT ?`,
			match, collectionName, collectionName, limit)
		if err != nil {
			return nil, fmt.Errorf("RAG FTS5 query: %w", err)
		}
		defer rows.Close()
		rank := 0
		for rows.Next() {
			var result RAGResult
			if err := rows.Scan(&result.ChunkID, &result.Content, &result.ChunkIdx, &result.Document); err != nil {
				return nil, err
			}
			rank++
			candidate := byID[result.ChunkID]
			candidate.result = result
			candidate.score += reciprocalRank(rank)
			byID[result.ChunkID] = candidate
		}
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	out := make([]retrievalCandidate, 0, len(byID))
	for _, candidate := range byID {
		out = append(out, candidate)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].exact != out[j].exact {
			return out[i].exact
		}
		if out[i].score != out[j].score {
			return out[i].score > out[j].score
		}
		return out[i].result.ChunkID < out[j].result.ChunkID
	})
	return out, nil
}

func reciprocalRank(rank int) float64 {
	if rank < 1 {
		rank = 1
	}
	return 1 / float64(60+rank)
}

func fuseRAGCandidates(vector, lexical []retrievalCandidate) []RAGResult {
	merged := make(map[string]retrievalCandidate, len(vector)+len(lexical))
	for rank, candidate := range vector {
		candidate.score = reciprocalRank(rank + 1)
		merged[candidate.result.ChunkID] = candidate
	}
	for rank, candidate := range lexical {
		prior := merged[candidate.result.ChunkID]
		prior.result = candidate.result
		prior.score += reciprocalRank(rank + 1)
		prior.exact = prior.exact || candidate.exact
		merged[candidate.result.ChunkID] = prior
	}
	ordered := make([]retrievalCandidate, 0, len(merged))
	for _, candidate := range merged {
		candidate.result.Score = candidate.score
		ordered = append(ordered, candidate)
	}
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].exact != ordered[j].exact {
			return ordered[i].exact
		}
		if ordered[i].score != ordered[j].score {
			return ordered[i].score > ordered[j].score
		}
		return ordered[i].result.ChunkID < ordered[j].result.ChunkID
	})
	results := make([]RAGResult, len(ordered))
	for i := range ordered {
		results[i] = ordered[i].result
		results[i].exactMatch = ordered[i].exact
	}
	return results
}

func (r *RAGStore) CreateCollection(name, description string, chunkSize int) (*Collection, error) {
	if chunkSize <= 0 {
		chunkSize = 500
	}
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := r.db.Exec(`INSERT INTO rag_collections(name,description,chunk_size,created_at,updated_at) VALUES(?,?,?,?,?)`,
		name, description, chunkSize, now, now)
	if err != nil {
		return nil, fmt.Errorf("create collection: %w", err)
	}
	id, _ := res.LastInsertId()
	return &Collection{ID: int(id), Name: name, Description: description, ChunkSize: chunkSize}, nil
}

func (r *RAGStore) ListCollections() ([]Collection, error) {
	rows, err := r.db.Query(`SELECT c.id,c.name,c.description,c.chunk_size,c.created_at,c.updated_at,
		COALESCE((SELECT COUNT(*) FROM rag_documents d WHERE d.collection_id=c.id),0) AS doc_count
		FROM rag_collections c ORDER BY c.name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var cols []Collection
	for rows.Next() {
		var c Collection
		var cs, us string
		if err := rows.Scan(&c.ID, &c.Name, &c.Description, &c.ChunkSize, &cs, &us, &c.DocCount); err != nil {
			return nil, err
		}
		c.CreatedAt, _ = time.Parse(time.RFC3339, cs)
		c.UpdatedAt, _ = time.Parse(time.RFC3339, us)
		cols = append(cols, c)
	}
	return cols, nil
}

func (r *RAGStore) DeleteCollection(name string) error {
	_, err := r.db.Exec(`DELETE FROM rag_chunks WHERE document_id IN (SELECT id FROM rag_documents WHERE collection_id=(SELECT id FROM rag_collections WHERE name=?))`, name)
	if err != nil {
		return err
	}
	_, err = r.db.Exec(`DELETE FROM rag_documents WHERE collection_id=(SELECT id FROM rag_collections WHERE name=?)`, name)
	if err != nil {
		return err
	}
	_, err = r.db.Exec(`DELETE FROM rag_collections WHERE name=?`, name)
	return err
}

func (r *RAGStore) IngestFile(filePath, collectionName string, metadata map[string]interface{}) (*Document, error) {
	if r.initErr != nil {
		return nil, fmt.Errorf("RAG index unavailable: %w", r.initErr)
	}
	if r.embedder == nil || !r.embedder.Available() {
		return nil, ErrRAGEmbedderUnavailable
	}
	r.mu.RLock()
	hnswPath := r.hnswPath
	hnsw := r.hnsw
	r.mu.RUnlock()
	if hnsw == nil || hnswPath == "" {
		return nil, ErrRAGIndexUnavailable
	}

	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}
	chunkSize, err := r.chunkSizeForCollection(collectionName)
	if err != nil {
		return nil, err
	}
	chunks := chunkText(string(data), chunkSize)
	if len(chunks) == 0 {
		return nil, fmt.Errorf("RAG document contains no chunks")
	}

	docID := fmt.Sprintf("rag_doc_%d", time.Now().UnixNano())
	type indexedChunk struct {
		id      string
		content string
		vector  []float32
	}
	prepared := make([]indexedChunk, 0, len(chunks))
	for i, chunk := range chunks {
		vector, err := embedRAGPassage(r.embedder, chunk)
		if err != nil {
			return nil, fmt.Errorf("embed chunk %d: %w", i, err)
		}
		if err := hnsw.ValidateVector(vector); err != nil {
			return nil, fmt.Errorf("validate chunk %d vector: %w", i, err)
		}
		prepared = append(prepared, indexedChunk{
			id:      fmt.Sprintf("rag_chk_%s_%d", docID, i),
			content: chunk,
			vector:  vector,
		})
	}

	col, err := r.getOrCreateCollection(collectionName)
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC().Format(time.RFC3339)
	filename := filepath.Base(filePath)
	sourceType := detectSourceType(filePath)
	metaJSON, _ := json.Marshal(metadata)
	tx, err := r.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("begin RAG ingest: %w", err)
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,chunk_count,created_at) VALUES(?,?,?,?,?,?,?)`,
		docID, col.ID, filename, sourceType, string(metaJSON), len(prepared), now); err != nil {
		return nil, fmt.Errorf("insert doc: %w", err)
	}
	for i, item := range prepared {
		if _, err := tx.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata) VALUES(?,?,?,?,?)`,
			item.id, docID, item.content, i, "{}"); err != nil {
			return nil, fmt.Errorf("insert chunk %d: %w", i, err)
		}
	}

	// Build from the current on-disk RAG index, then merge and save under its
	// cross-process lock so a concurrent writer cannot be lost.
	r.mu.Lock()
	defer r.mu.Unlock()
	current := r.hnsw
	if current == nil {
		return nil, ErrRAGIndexUnavailable
	}
	candidate := index.NewHNSW(index.DefaultHNSWConfig(current.Dimension()))
	candidate.EnableRAGBoundedDegree()
	if err := candidate.Load(hnswPath); err != nil {
		return nil, fmt.Errorf("reload RAG HNSW: %w", err)
	}
	for _, item := range prepared {
		if err := candidate.Insert(item.id, item.vector); err != nil {
			return nil, fmt.Errorf("index chunk %q: %w", item.id, err)
		}
	}
	if _, _, err := candidate.MergeAndSave(hnswPath); err != nil {
		return nil, fmt.Errorf("persist RAG HNSW: %w", err)
	}
	updated := index.NewHNSW(index.DefaultHNSWConfig(current.Dimension()))
	updated.EnableRAGBoundedDegree()
	if err := updated.Load(hnswPath); err != nil {
		return nil, fmt.Errorf("reload persisted RAG HNSW: %w", err)
	}
	if _, err := tx.Exec(`UPDATE rag_chunks SET vector_indexed=1 WHERE document_id=?`, docID); err != nil {
		return nil, fmt.Errorf("mark RAG chunks indexed: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit RAG ingest after vector persistence: %w", err)
	}
	r.hnsw = updated

	return &Document{
		ID: docID, Collection: col.ID, Filename: filename,
		SourceType: sourceType, ChunkCount: len(prepared), VectorIndexedCount: len(prepared),
		CreatedAt: time.Now(),
	}, nil
}

func (r *RAGStore) IngestDir(dirPath, collectionName string) (int, error) {
	entries, err := os.ReadDir(dirPath)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if !isSupportedExt(ext) {
			continue
		}
		path := filepath.Join(dirPath, entry.Name())
		if _, err := r.IngestFile(path, collectionName, nil); err != nil {
			fmt.Fprintf(os.Stderr, "  skip %s: %v\n", entry.Name(), err)
			continue
		}
		count++
	}
	return count, nil
}

type RAGResult struct {
	ChunkID    string  `json:"chunk_id"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
	Document   string  `json:"document"`
	ChunkIdx   int     `json:"chunk_index"`
	exactMatch bool
}

// Query performs a RAG search with default reranker.
// Priority: Jina API → opt-in local Python CrossEncoder → Noop (vector score only)
func (r *RAGStore) Query(query string, topK int, collectionName string) ([]RAGResult, error) {
	if r.initErr != nil {
		return nil, fmt.Errorf("RAG index unavailable: %w", r.initErr)
	}
	if r.embedder == nil || !r.embedder.Available() {
		return nil, ErrRAGEmbedderUnavailable
	}
	r.mu.RLock()
	hasIndex := r.hnsw != nil
	r.mu.RUnlock()
	if !hasIndex {
		return nil, ErrRAGIndexUnavailable
	}
	var reranker Reranker = &NoopReranker{}
	if jina := NewOpenAIReranker(); jina.Available() {
		reranker = jina
	} else if LocalCrossEncoderEnabled() {
		if ce := NewPythonCrossEncoder(); ce.Available() {
			defer ce.Stop()
			reranker = ce
		}
	}
	return r.QueryWithRerank(query, topK, collectionName, reranker)
}

func (r *RAGStore) chunkSizeForCollection(name string) (int, error) {
	var size int
	err := r.db.QueryRow(`SELECT chunk_size FROM rag_collections WHERE name=?`, name).Scan(&size)
	if errors.Is(err, sql.ErrNoRows) {
		return 500, nil
	}
	if err != nil {
		return 0, fmt.Errorf("read collection chunk size: %w", err)
	}
	if size <= 0 {
		return 500, nil
	}
	return size, nil
}

func (r *RAGStore) refreshHNSW() (*index.HNSW, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.initErr != nil {
		return nil, fmt.Errorf("RAG index unavailable: %w", r.initErr)
	}
	if r.hnsw == nil || r.hnswPath == "" {
		return nil, ErrRAGIndexUnavailable
	}
	if err := r.hnsw.Load(r.hnswPath); err != nil {
		return nil, fmt.Errorf("refresh RAG HNSW: %w", err)
	}
	return r.hnsw, nil
}

func (r *RAGStore) getOrCreateCollection(name string) (*Collection, error) {
	row := r.db.QueryRow(`SELECT id,name,description,chunk_size FROM rag_collections WHERE name=?`, name)
	var c Collection
	err := row.Scan(&c.ID, &c.Name, &c.Description, &c.ChunkSize)
	if err == nil {
		return &c, nil
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return nil, fmt.Errorf("read collection: %w", err)
	}
	return r.CreateCollection(name, fmt.Sprintf("Auto-created for %s", name), 500)
}

func detectSourceType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".txt":
		return "text"
	case ".md":
		return "markdown"
	case ".json":
		return "json"
	case ".csv":
		return "csv"
	default:
		return "file"
	}
}

func isSupportedExt(ext string) bool {
	switch ext {
	case ".txt", ".md", ".json", ".csv":
		return true
	default:
		return false
	}
}

func chunkText(text string, targetSize int) []string {
	if len(text) <= targetSize {
		return []string{text}
	}
	var chunks []string
	for len(text) > 0 {
		if len(text) <= targetSize {
			chunks = append(chunks, text)
			break
		}
		cut := targetSize
		if idx := strings.LastIndex(text[:targetSize], "\n\n"); idx > targetSize/2 {
			cut = idx + 2
		} else if idx := strings.LastIndex(text[:targetSize], ". "); idx > targetSize/2 {
			cut = idx + 2
		} else if idx := strings.LastIndex(text[:targetSize], "\n"); idx > targetSize/2 {
			cut = idx + 1
		}
		chunks = append(chunks, text[:cut])
		text = text[cut:]
	}
	return chunks
}

func (r *RAGStore) Stats() map[string]int {
	var cols, docs, chunks, indexedChunks int
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_collections`).Scan(&cols)
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_documents`).Scan(&docs)
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks`).Scan(&chunks)
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks WHERE vector_indexed=1`).Scan(&indexedChunks)
	hnsw, refreshErr := r.refreshHNSW()
	if refreshErr != nil {
		r.mu.RLock()
		hnsw = r.hnsw
		r.mu.RUnlock()
	}
	indexVectors, indexReady := 0, 0
	if hnsw != nil {
		indexVectors = hnsw.Size()
		if r.initErr == nil && refreshErr == nil {
			indexReady = 1
		}
	}
	embedderConfigured := 0
	if r.initErr == nil && r.embedder != nil {
		embedderConfigured = 1
	}
	return map[string]int{"collections": cols, "documents": docs, "chunks": chunks, "vector_indexed_chunks": indexedChunks, "rag_index_vectors": indexVectors, "index_ready": indexReady, "embedder_configured": embedderConfigured}
}

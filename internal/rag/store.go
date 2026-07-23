package rag

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/index"
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
	ID         string          `json:"id"`
	Collection int             `json:"collection_id"`
	Filename   string          `json:"filename"`
	SourceType string          `json:"source_type"`
	Metadata   json.RawMessage `json:"metadata,omitempty"`
	ChunkCount int             `json:"chunk_count"`
	CreatedAt  time.Time       `json:"created_at"`
}

type Chunk struct {
	ID          string          `json:"id"`
	DocumentID  string          `json:"document_id"`
	Content     string          `json:"content"`
	ChunkIndex  int             `json:"chunk_index"`
	Metadata    json.RawMessage `json:"metadata,omitempty"`
}

type Embedder interface {
	Embed(string) ([]float32, error)
	Available() bool
	Dims() int
}

type RAGStore struct {
	db       *sql.DB
	hnsw     *index.HNSW
	hnswPath string
	embedder Embedder
}

func NewRAGStore(db *sql.DB, hnsw *index.HNSW, hnswPath string, emb Embedder) *RAGStore {
	r := &RAGStore{db: db, hnsw: hnsw, hnswPath: hnswPath, embedder: emb}
	r.migrate()
	return r
}

func (r *RAGStore) migrate() {
	r.db.Exec(`CREATE TABLE IF NOT EXISTS rag_collections (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		name TEXT NOT NULL UNIQUE,
		description TEXT DEFAULT '',
		chunk_size INTEGER NOT NULL DEFAULT 500,
		created_at TEXT NOT NULL DEFAULT (datetime('now')),
		updated_at TEXT NOT NULL DEFAULT (datetime('now')));
	CREATE INDEX IF NOT EXISTS idx_rag_collections_name ON rag_collections(name);`)

	r.db.Exec(`CREATE TABLE IF NOT EXISTS rag_documents (
		id TEXT PRIMARY KEY,
		collection_id INTEGER NOT NULL REFERENCES rag_collections(id),
		filename TEXT NOT NULL,
		source_type TEXT NOT NULL DEFAULT 'file',
		metadata TEXT DEFAULT '{}',
		chunk_count INTEGER NOT NULL DEFAULT 0,
		created_at TEXT NOT NULL DEFAULT (datetime('now')));
	CREATE INDEX IF NOT EXISTS idx_rag_documents_collection ON rag_documents(collection_id);`)

	r.db.Exec(`CREATE TABLE IF NOT EXISTS rag_chunks (
		id TEXT PRIMARY KEY,
		document_id TEXT NOT NULL REFERENCES rag_documents(id),
		content TEXT NOT NULL,
		chunk_index INTEGER NOT NULL DEFAULT 0,
		metadata TEXT DEFAULT '{}');
	CREATE INDEX IF NOT EXISTS idx_rag_chunks_document ON rag_chunks(document_id);`)
}

func (r *RAGStore) CreateCollection(name, description string, chunkSize int) (*Collection, error) {
	if chunkSize <= 0 { chunkSize = 500 }
	now := time.Now().UTC().Format(time.RFC3339)
	res, err := r.db.Exec(`INSERT INTO rag_collections(name,description,chunk_size,created_at,updated_at) VALUES(?,?,?,?,?)`,
		name, description, chunkSize, now, now)
	if err != nil { return nil, fmt.Errorf("create collection: %w", err) }
	id, _ := res.LastInsertId()
	return &Collection{ID: int(id), Name: name, Description: description, ChunkSize: chunkSize}, nil
}

func (r *RAGStore) ListCollections() ([]Collection, error) {
	rows, err := r.db.Query(`SELECT c.id,c.name,c.description,c.chunk_size,c.created_at,c.updated_at,
		COALESCE((SELECT COUNT(*) FROM rag_documents d WHERE d.collection_id=c.id),0) AS doc_count
		FROM rag_collections c ORDER BY c.name`)
	if err != nil { return nil, err }
	defer rows.Close()
	var cols []Collection
	for rows.Next() {
		var c Collection
		var cs, us string
		if err := rows.Scan(&c.ID, &c.Name, &c.Description, &c.ChunkSize, &cs, &us, &c.DocCount); err != nil { return nil, err }
		c.CreatedAt, _ = time.Parse(time.RFC3339, cs)
		c.UpdatedAt, _ = time.Parse(time.RFC3339, us)
		cols = append(cols, c)
	}
	return cols, nil
}

func (r *RAGStore) DeleteCollection(name string) error {
	_, err := r.db.Exec(`DELETE FROM rag_chunks WHERE document_id IN (SELECT id FROM rag_documents WHERE collection_id=(SELECT id FROM rag_collections WHERE name=?))`, name)
	if err != nil { return err }
	_, err = r.db.Exec(`DELETE FROM rag_documents WHERE collection_id=(SELECT id FROM rag_collections WHERE name=?)`, name)
	if err != nil { return err }
	_, err = r.db.Exec(`DELETE FROM rag_collections WHERE name=?`, name)
	return err
}

func (r *RAGStore) IngestFile(filePath, collectionName string, metadata map[string]interface{}) (*Document, error) {
	data, err := os.ReadFile(filePath)
	if err != nil { return nil, fmt.Errorf("read file: %w", err) }

	col, err := r.getOrCreateCollection(collectionName)
	if err != nil { return nil, err }

	docID := fmt.Sprintf("rag_doc_%d", time.Now().UnixNano())
	filename := filepath.Base(filePath)
	sourceType := detectSourceType(filePath)
	metaJSON, _ := json.Marshal(metadata)
	now := time.Now().UTC().Format(time.RFC3339)
	_, err = r.db.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,created_at) VALUES(?,?,?,?,?,?)`,
		docID, col.ID, filename, sourceType, string(metaJSON), now)
	if err != nil { return nil, fmt.Errorf("insert doc: %w", err) }

	chunks := chunkText(string(data), col.ChunkSize)
	chunkCount := 0
	for i, chunk := range chunks {
		chunkID := fmt.Sprintf("rag_chk_%s_%d", docID, i)
		_, err := r.db.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata) VALUES(?,?,?,?,?)`,
			chunkID, docID, chunk, i, "{}")
		if err != nil { continue }
		chunkCount++
		if r.embedder != nil && r.embedder.Available() {
			if vec, e := r.embedder.Embed(chunk); e == nil && len(vec) > 0 {
				r.hnsw.Insert(chunkID, vec)
			}
		}
	}
	r.db.Exec(`UPDATE rag_documents SET chunk_count=? WHERE id=?`, chunkCount, docID)
	r.hnsw.Save(r.hnswPath)

	return &Document{
		ID: docID, Collection: col.ID, Filename: filename,
		SourceType: sourceType, ChunkCount: chunkCount,
		CreatedAt: time.Now(),
	}, nil
}

func (r *RAGStore) IngestDir(dirPath, collectionName string) (int, error) {
	entries, err := os.ReadDir(dirPath)
	if err != nil { return 0, err }
	count := 0
	for _, entry := range entries {
		if entry.IsDir() { continue }
		ext := strings.ToLower(filepath.Ext(entry.Name()))
		if !isSupportedExt(ext) { continue }
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
	ChunkID  string  `json:"chunk_id"`
	Content  string  `json:"content"`
	Score    float64 `json:"score"`
	Document string  `json:"document"`
	ChunkIdx int     `json:"chunk_index"`
}

func (r *RAGStore) Query(query string, topK int, collectionName string) ([]RAGResult, error) {
	if r.embedder == nil || !r.embedder.Available() {
		return nil, fmt.Errorf("embedder not available")
	}
	if topK <= 0 { topK = 5 }

	vec, err := r.embedder.Embed(query)
	if err != nil { return nil, fmt.Errorf("embed query: %w", err) }

	results := r.hnsw.Search(vec, topK*3)
	if len(results) == 0 { return nil, nil }

	var ragResults []RAGResult
	for _, res := range results {
		if !strings.HasPrefix(res.ID, "rag_chk_") { continue }

		var content string
		var docID string
		var chunkIdx int
		err := r.db.QueryRow(`SELECT content, document_id, chunk_index FROM rag_chunks WHERE id=?`, res.ID).Scan(&content, &docID, &chunkIdx)
		if err != nil { continue }

		if collectionName != "" {
			var colID int
			r.db.QueryRow(`SELECT collection_id FROM rag_documents WHERE id=?`, docID).Scan(&colID)
			if colID == 0 { continue }
			var colName string
			r.db.QueryRow(`SELECT name FROM rag_collections WHERE id=?`, colID).Scan(&colName)
			if colName != collectionName { continue }
		}

		var filename string
		r.db.QueryRow(`SELECT filename FROM rag_documents WHERE id=?`, docID).Scan(&filename)

		ragResults = append(ragResults, RAGResult{
			ChunkID: res.ID, Content: content, Score: 1.0 / (1.0 + res.Distance),
			Document: filename, ChunkIdx: chunkIdx,
		})
	}

	for i := 0; i < len(ragResults); i++ {
		for j := i + 1; j < len(ragResults); j++ {
			if ragResults[j].Score > ragResults[i].Score {
				ragResults[i], ragResults[j] = ragResults[j], ragResults[i]
			}
		}
	}
	if len(ragResults) > topK {
		ragResults = ragResults[:topK]
	}
	return ragResults, nil
}

func (r *RAGStore) getOrCreateCollection(name string) (*Collection, error) {
	row := r.db.QueryRow(`SELECT id,name,description,chunk_size FROM rag_collections WHERE name=?`, name)
	var c Collection
	err := row.Scan(&c.ID, &c.Name, &c.Description, &c.ChunkSize)
	if err == nil { return &c, nil }
	return r.CreateCollection(name, fmt.Sprintf("Auto-created for %s", name), 500)
}

func detectSourceType(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".txt": return "text"
	case ".md": return "markdown"
	case ".json": return "json"
	case ".csv": return "csv"
	default: return "file"
	}
}

func isSupportedExt(ext string) bool {
	switch ext {
	case ".txt", ".md", ".json", ".csv": return true
	default: return false
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
	var cols, docs, chunks int
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_collections`).Scan(&cols)
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_documents`).Scan(&docs)
	r.db.QueryRow(`SELECT COUNT(*) FROM rag_chunks`).Scan(&chunks)
	return map[string]int{"collections": cols, "documents": docs, "chunks": chunks}
}

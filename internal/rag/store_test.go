package rag

import (
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

type fakeRAGEmbedder struct {
	dims         int
	available    bool
	failAt       int
	calls        int
	queryCalls   int
	passageCalls int
}

func (e *fakeRAGEmbedder) Embed(text string) ([]float32, error) {
	e.calls++
	if e.failAt > 0 && e.calls == e.failAt {
		return nil, errors.New("fake embedding failure")
	}
	v := make([]float32, e.dims)
	if len(v) > 0 {
		v[0] = float32(len(text) + 1)
	}
	return v, nil
}

func (e *fakeRAGEmbedder) EmbedQuery(text string) ([]float32, error) {
	e.queryCalls++
	return e.Embed(text)
}

func (e *fakeRAGEmbedder) EmbedPassage(text string) ([]float32, error) {
	e.passageCalls++
	return e.Embed(text)
}

func (e *fakeRAGEmbedder) Available() bool { return e.available }
func (e *fakeRAGEmbedder) Dims() int       { return e.dims }

func newRAGTestStore(t *testing.T, emb Embedder, profile EmbeddingProfile) (*RAGStore, string) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "memory.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatalf("open test DB: %v", err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = db.Close() })

	indexPath := dbPath + ".rag.hnsw"
	hnsw, err := OpenVectorIndex(db, indexPath, profile)
	if err != nil {
		t.Fatalf("open test RAG index: %v", err)
	}
	return NewRAGStore(db, hnsw, indexPath, emb), dbPath
}

func TestIngestFileFailsClosedWhenRAGEmbedderUnavailable(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: false}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, _ := newRAGTestStore(t, emb, profile)
	file := filepath.Join(t.TempDir(), "note.txt")
	if err := os.WriteFile(file, []byte("synthetic RAG text"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := rs.IngestFile(file, "test", nil); !errors.Is(err, ErrRAGEmbedderUnavailable) {
		t.Fatalf("IngestFile() error = %v, want ErrRAGEmbedderUnavailable", err)
	}
	stats := rs.Stats()
	if stats["documents"] != 0 || stats["chunks"] != 0 || stats["vector_indexed_chunks"] != 0 {
		t.Fatalf("unavailable embedder wrote RAG data: %+v", stats)
	}
}

func TestIngestFileEmbedsAllChunksBeforeWritingAndQueryUsesRAGIndex(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: true}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, _ := newRAGTestStore(t, emb, profile)
	if _, err := rs.CreateCollection("test", "synthetic", 6); err != nil {
		t.Fatalf("CreateCollection() error = %v", err)
	}
	file := filepath.Join(t.TempDir(), "note.txt")
	if err := os.WriteFile(file, []byte("alpha beta gamma delta"), 0o600); err != nil {
		t.Fatal(err)
	}

	doc, err := rs.IngestFile(file, "test", nil)
	if err != nil {
		t.Fatalf("IngestFile() error = %v", err)
	}
	if doc.ChunkCount == 0 || doc.VectorIndexedCount != doc.ChunkCount {
		t.Fatalf("ingest counts = chunks:%d indexed:%d", doc.ChunkCount, doc.VectorIndexedCount)
	}
	stats := rs.Stats()
	if stats["vector_indexed_chunks"] != doc.VectorIndexedCount {
		t.Fatalf("vector stats = %d, want %d", stats["vector_indexed_chunks"], doc.VectorIndexedCount)
	}
	if stats["index_ready"] != 1 || stats["embedder_configured"] != 1 {
		t.Fatalf("RAG readiness stats = %+v, want ready index and configured embedder", stats)
	}

	results, err := rs.QueryWithRerank("alpha", 5, "test", &NoopReranker{})
	if err != nil {
		t.Fatalf("QueryWithRerank() error = %v", err)
	}
	if len(results) == 0 {
		t.Fatal("QueryWithRerank() returned no synthetic chunks")
	}
	if emb.passageCalls == 0 || emb.queryCalls == 0 {
		t.Fatalf("RAG task calls: passage=%d query=%d", emb.passageCalls, emb.queryCalls)
	}
}

func TestHybridRetrievalPreservesExactIDsAndFindsNearIDs(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: true}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, _ := newRAGTestStore(t, emb, profile)
	if _, err := rs.CreateCollection("ids", "synthetic identifiers", 7); err != nil {
		t.Fatalf("CreateCollection() error = %v", err)
	}
	file := filepath.Join(t.TempDir(), "identifier.txt")
	if err := os.WriteFile(file, []byte("alpha beta gamma delta"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := rs.IngestFile(file, "ids", nil); err != nil {
		t.Fatalf("IngestFile() error = %v", err)
	}
	var chunkID string
	if err := rs.db.QueryRow(`SELECT id FROM rag_chunks ORDER BY chunk_index LIMIT 1`).Scan(&chunkID); err != nil {
		t.Fatal(err)
	}

	exact, err := rs.queryInternal(chunkID, 5, "ids")
	if err != nil || len(exact) == 0 || exact[0].ChunkID != chunkID || !exact[0].exactMatch {
		t.Fatalf("exact-ID retrieval = %+v, %v; want exact chunk %q first", exact, err, chunkID)
	}
	nearID := chunkID[:len(chunkID)-1] + "9"
	near, err := rs.queryInternal(nearID, 5, "ids")
	if err != nil {
		t.Fatalf("near-ID query: %v", err)
	}
	found := false
	for _, result := range near {
		if result.ChunkID == chunkID {
			found = true
			if result.exactMatch {
				t.Fatalf("near-ID %q was incorrectly marked exact", nearID)
			}
		}
	}
	if !found {
		t.Fatalf("near-ID query %q did not retrieve candidate %q: %+v", nearID, chunkID, near)
	}
}

func TestSafeFTSQueryEscapesOperatorsAndFusionDeduplicates(t *testing.T) {
	if got, want := safeFTSQuery(`alpha" OR beta AND *`), `"alpha" OR "OR" OR "beta" OR "AND"`; got != want {
		t.Fatalf("safeFTSQuery() = %q, want %q", got, want)
	}
	vector := []retrievalCandidate{
		{result: RAGResult{ChunkID: "semantic-first", Content: "vector"}},
		{result: RAGResult{ChunkID: "conflict-pair", Content: "both"}},
	}
	lexical := []retrievalCandidate{
		{result: RAGResult{ChunkID: "conflict-pair", Content: "both"}},
		{result: RAGResult{ChunkID: "lexical-only", Content: "lexical"}},
	}
	fused := fuseRAGCandidates(vector, lexical)
	if len(fused) != 3 || fused[0].ChunkID != "conflict-pair" {
		t.Fatalf("semantic/lexical fusion = %+v; want deduplicated shared candidate first", fused)
	}
	seen := map[string]bool{}
	for _, result := range fused {
		if seen[result.ChunkID] {
			t.Fatalf("duplicate fused candidate %q", result.ChunkID)
		}
		seen[result.ChunkID] = true
	}
	exact := fuseRAGCandidates(vector, append(lexical, retrievalCandidate{result: RAGResult{ChunkID: "semantic-first"}, exact: true}))
	if exact[0].ChunkID != "semantic-first" || !exact[0].exactMatch {
		t.Fatalf("exact conflict candidate was not preserved: %+v", exact)
	}
}

func TestIngestFileEmbeddingFailureWritesNoDocumentOrChunks(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: true, failAt: 2}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, _ := newRAGTestStore(t, emb, profile)
	if _, err := rs.CreateCollection("test", "synthetic", 6); err != nil {
		t.Fatalf("CreateCollection() error = %v", err)
	}
	file := filepath.Join(t.TempDir(), "note.txt")
	if err := os.WriteFile(file, []byte("alpha beta gamma delta"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := rs.IngestFile(file, "test", nil); err == nil {
		t.Fatal("IngestFile() succeeded after fake embedder failure")
	}
	stats := rs.Stats()
	if stats["documents"] != 0 || stats["chunks"] != 0 || stats["vector_indexed_chunks"] != 0 {
		t.Fatalf("failed embedding wrote RAG data: %+v", stats)
	}
}

func TestRAGMigrationMarksExistingChunksUnindexed(t *testing.T) {
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "legacy.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE rag_chunks (id TEXT PRIMARY KEY, document_id TEXT NOT NULL, content TEXT NOT NULL, chunk_index INTEGER NOT NULL DEFAULT 0, metadata TEXT DEFAULT '{}')`); err != nil {
		t.Fatalf("create legacy rag_chunks: %v", err)
	}

	rs := NewRAGStore(db, nil, "", nil)
	if rs.initErr != nil {
		t.Fatalf("RAG migration error = %v", rs.initErr)
	}
	var vectorIndexedColumn bool
	rows, err := db.Query(`PRAGMA table_info(rag_chunks)`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			t.Fatal(err)
		}
		if name == "vector_indexed" {
			vectorIndexedColumn = true
			if fmt.Sprint(defaultValue) != "0" {
				t.Fatalf("vector_indexed default = %v, want 0", defaultValue)
			}
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if !vectorIndexedColumn {
		t.Fatal("RAG migration did not add vector_indexed")
	}
}

func TestExactStructuredIDsInDocumentContentAndNearMiss(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: true}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, _ := newRAGTestStore(t, emb, profile)
	collection, err := rs.CreateCollection("record-ids", "synthetic records", 500)
	if err != nil {
		t.Fatal(err)
	}
	for _, doc := range []struct {
		id       string
		filename string
		chunkID  string
		content  string
	}{
		{"doc-exact", "target.txt", "chunk-exact", "Record REC-2024-00831 has target details"},
		{"doc-other", "other.txt", "chunk-other", "Record OTHER-2033-90210 has other details"},
	} {
		if _, err := rs.db.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,chunk_count,created_at) VALUES(?,?,?,'text','{}',1,'now')`, doc.id, collection.ID, doc.filename); err != nil {
			t.Fatalf("insert document %s: %v", doc.id, err)
		}
		if _, err := rs.db.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata) VALUES(?,?,?,0,'{}')`, doc.chunkID, doc.id, doc.content); err != nil {
			t.Fatalf("insert chunk %s: %v", doc.chunkID, err)
		}
	}

	exact, err := rs.lexicalCandidates("REC-2024-00831", "record-ids", 10)
	if err != nil {
		t.Fatalf("exact document-ID search: %v", err)
	}
	if len(exact) == 0 || exact[0].result.ChunkID != "chunk-exact" || !exact[0].exact {
		t.Fatalf("exact content-ID results = %+v, want chunk-exact flagged exact and prioritized", exact)
	}

	near, err := rs.lexicalCandidates("REC-2024-00839", "record-ids", 10)
	if err != nil {
		t.Fatalf("near-miss document-ID search: %v", err)
	}
	foundNearCandidate := false
	for _, candidate := range near {
		if candidate.result.ChunkID == "chunk-exact" {
			foundNearCandidate = true
			if candidate.exact {
				t.Fatalf("one-character near miss was marked exact: %+v", candidate)
			}
		}
	}
	if !foundNearCandidate {
		t.Fatalf("FTS did not return near-miss candidate for a non-exact comparison: %+v", near)
	}
}

func TestRAGFTSMigrationBackfillsLegacyRowsAndRepairsPartialState(t *testing.T) {
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "legacy-fts.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	legacySchema := []string{
		`CREATE TABLE rag_collections(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL UNIQUE,description TEXT DEFAULT '',chunk_size INTEGER NOT NULL DEFAULT 500,created_at TEXT NOT NULL DEFAULT 'now',updated_at TEXT NOT NULL DEFAULT 'now')`,
		`CREATE TABLE rag_documents(id TEXT PRIMARY KEY,collection_id INTEGER NOT NULL,filename TEXT NOT NULL,source_type TEXT NOT NULL DEFAULT 'file',metadata TEXT DEFAULT '{}',chunk_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL DEFAULT 'now')`,
		`CREATE TABLE rag_chunks(id TEXT PRIMARY KEY,document_id TEXT NOT NULL,content TEXT NOT NULL,chunk_index INTEGER NOT NULL DEFAULT 0,metadata TEXT DEFAULT '{}')`,
		`INSERT INTO rag_collections(name) VALUES('legacy')`,
		`INSERT INTO rag_documents(id,collection_id,filename) VALUES('legacy-doc',1,'legacy.txt')`,
		`INSERT INTO rag_chunks(id,document_id,content) VALUES('legacy-chunk','legacy-doc','legacyneedle searchable content')`,
		`CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(content,id,tokenize='unicode61')`,
		`INSERT INTO rag_chunks_fts(rowid,content,id) VALUES(999,'staleorphan stale','orphan-chunk')`,
	}
	for _, statement := range legacySchema {
		if _, err := db.Exec(statement); err != nil {
			t.Fatalf("prepare pre-migration legacy state: %v", err)
		}
	}

	rs := NewRAGStore(db, nil, "", nil)
	if rs.initErr != nil {
		t.Fatalf("migrate legacy RAG store: %v", rs.initErr)
	}
	legacy, err := rs.lexicalCandidates("legacyneedle", "legacy", 5)
	if err != nil || len(legacy) != 1 || legacy[0].result.ChunkID != "legacy-chunk" {
		t.Fatalf("legacy lexical result = %+v, %v; want backfilled legacy-chunk", legacy, err)
	}
	assertFTSMatchCount := func(term string, want int) {
		t.Helper()
		var count int
		if err := db.QueryRow(`SELECT COUNT(*) FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ?`, `"`+term+`"`).Scan(&count); err != nil {
			t.Fatalf("count FTS match %q: %v", term, err)
		}
		if count != want {
			t.Fatalf("FTS match count for %q = %d, want %d", term, count, want)
		}
	}
	assertFTSMatchCount("legacyneedle", 1)
	assertFTSMatchCount("staleorphan", 0)
	var migrationVersion int
	if err := db.QueryRow(`SELECT version FROM rag_schema_migrations WHERE name='rag_chunks_fts'`).Scan(&migrationVersion); err != nil {
		t.Fatalf("read FTS migration marker: %v", err)
	}
	if migrationVersion != ragFTSMigrationVersion {
		t.Fatalf("FTS migration version = %d, want %d", migrationVersion, ragFTSMigrationVersion)
	}

	// An FTS-only sentinel survives a healthy second initialization, proving
	// startup did not truncate and rebuild the index.
	if _, err := db.Exec(`INSERT INTO rag_chunks_fts(rowid,content,id) VALUES(9999,'sentinelmarker retained','sentinel-chunk')`); err != nil {
		t.Fatalf("insert no-rebuild sentinel: %v", err)
	}
	if repeated := NewRAGStore(db, nil, "", nil); repeated.InitError() != nil {
		t.Fatalf("second RAG store initialization: %v", repeated.InitError())
	}
	assertFTSMatchCount("legacyneedle", 1)
	assertFTSMatchCount("sentinelmarker", 1)
	var repeatedVersion int
	if err := db.QueryRow(`SELECT version FROM rag_schema_migrations WHERE name='rag_chunks_fts'`).Scan(&repeatedVersion); err != nil {
		t.Fatalf("read repeated FTS migration marker: %v", err)
	}
	if repeatedVersion != migrationVersion {
		t.Fatalf("FTS migration version changed from %d to %d on normal startup", migrationVersion, repeatedVersion)
	}

	if _, err := db.Exec(`INSERT INTO rag_chunks(id,document_id,content) VALUES('trigger-chunk','legacy-doc','insertmarker content')`); err != nil {
		t.Fatalf("insert trigger row: %v", err)
	}
	assertFTSMatchCount("insertmarker", 1)
	if _, err := db.Exec(`UPDATE rag_chunks SET content='updatemarker replacement' WHERE id='trigger-chunk'`); err != nil {
		t.Fatalf("update trigger row: %v", err)
	}
	assertFTSMatchCount("insertmarker", 0)
	assertFTSMatchCount("updatemarker", 1)
	if _, err := db.Exec(`DELETE FROM rag_chunks WHERE id='trigger-chunk'`); err != nil {
		t.Fatalf("delete trigger row: %v", err)
	}
	assertFTSMatchCount("updatemarker", 0)
}

func TestConcurrentRAGFTSInitializersSerializeLegacyBackfill(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "concurrent-legacy.db")
	setupDB, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	setupDB.SetMaxOpenConns(1)
	legacySchema := []string{
		`CREATE TABLE rag_collections(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL UNIQUE,description TEXT DEFAULT '',chunk_size INTEGER NOT NULL DEFAULT 500,created_at TEXT NOT NULL DEFAULT 'now',updated_at TEXT NOT NULL DEFAULT 'now')`,
		`CREATE INDEX idx_rag_collections_name ON rag_collections(name)`,
		`CREATE TABLE rag_documents(id TEXT PRIMARY KEY,collection_id INTEGER NOT NULL,filename TEXT NOT NULL,source_type TEXT NOT NULL DEFAULT 'file',metadata TEXT DEFAULT '{}',chunk_count INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL DEFAULT 'now')`,
		`CREATE INDEX idx_rag_documents_collection ON rag_documents(collection_id)`,
		`CREATE TABLE rag_chunks(id TEXT PRIMARY KEY,document_id TEXT NOT NULL,content TEXT NOT NULL,chunk_index INTEGER NOT NULL DEFAULT 0,metadata TEXT DEFAULT '{}',vector_indexed INTEGER NOT NULL DEFAULT 0)`,
		`CREATE INDEX idx_rag_chunks_document ON rag_chunks(document_id)`,
		`INSERT INTO rag_collections(name) VALUES('legacy')`,
		`INSERT INTO rag_documents(id,collection_id,filename,chunk_count) VALUES('legacy-doc',1,'legacy.txt',3)`,
		`INSERT INTO rag_chunks(id,document_id,content,chunk_index) VALUES('legacy-chunk-a','legacy-doc','concurrentalpha legacy content',0)`,
		`INSERT INTO rag_chunks(id,document_id,content,chunk_index) VALUES('legacy-chunk-b','legacy-doc','concurrentbeta legacy content',1)`,
		`INSERT INTO rag_chunks(id,document_id,content,chunk_index) VALUES('legacy-chunk-c','legacy-doc','concurrentgamma legacy content',2)`,
	}
	for _, statement := range legacySchema {
		if _, err := setupDB.Exec(statement); err != nil {
			_ = setupDB.Close()
			t.Fatalf("prepare legacy RAG schema: %v", err)
		}
	}
	if err := setupDB.Close(); err != nil {
		t.Fatal(err)
	}

	const initializerCount = 12
	dbs := make([]*sql.DB, initializerCount)
	for i := range dbs {
		dbs[i], err = sql.Open("sqlite3", dbPath)
		if err != nil {
			t.Fatalf("open initializer DB %d: %v", i, err)
		}
		dbs[i].SetMaxOpenConns(1)
		defer dbs[i].Close()
	}

	start := make(chan struct{})
	errs := make(chan error, initializerCount)
	var wg sync.WaitGroup
	for _, db := range dbs {
		wg.Add(1)
		go func(db *sql.DB) {
			defer wg.Done()
			<-start
			errs <- ensureRAGFTS(db)
		}(db)
	}
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Errorf("concurrent ensureRAGFTS(): %v", err)
		}
	}

	verifyDB, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer verifyDB.Close()
	var markerCount, markerVersion int
	if err := verifyDB.QueryRow(`SELECT COUNT(*),COALESCE(MAX(version),0) FROM rag_schema_migrations WHERE name='rag_chunks_fts'`).Scan(&markerCount, &markerVersion); err != nil {
		t.Fatalf("read concurrent migration marker: %v", err)
	}
	if markerCount != 1 || markerVersion != ragFTSMigrationVersion {
		t.Fatalf("migration marker count/version = %d/%d, want 1/%d", markerCount, markerVersion, ragFTSMigrationVersion)
	}
	var indexedRows int
	if err := verifyDB.QueryRow(`SELECT COUNT(*) FROM rag_chunks_fts`).Scan(&indexedRows); err != nil {
		t.Fatalf("count concurrent FTS rows: %v", err)
	}
	if indexedRows != 3 {
		t.Fatalf("indexed legacy row count = %d, want exactly 3", indexedRows)
	}
	for _, term := range []string{"concurrentalpha", "concurrentbeta", "concurrentgamma"} {
		var matches int
		if err := verifyDB.QueryRow(`SELECT COUNT(*) FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ?`, `"`+term+`"`).Scan(&matches); err != nil {
			t.Fatalf("search migrated legacy term %q: %v", term, err)
		}
		if matches != 1 {
			t.Errorf("legacy term %q match count = %d, want 1", term, matches)
		}
	}
}

func TestRAGVectorSearchOverfetchesCollectionsAndFusionTiesAreStable(t *testing.T) {
	emb := &fakeRAGEmbedder{dims: 4, available: true}
	profile := EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4}
	rs, dbPath := newRAGTestStore(t, emb, profile)
	targetCollection, err := rs.CreateCollection("requested", "requested collection", 500)
	if err != nil {
		t.Fatal(err)
	}
	otherCollection, err := rs.CreateCollection("busy-other", "distractor collection", 500)
	if err != nil {
		t.Fatal(err)
	}

	insertChunk := func(collectionID int, docID, chunkID, filename, content string, vector []float32) {
		t.Helper()
		if _, err := rs.db.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,chunk_count,created_at) VALUES(?,?,?,'text','{}',1,'now')`, docID, collectionID, filename); err != nil {
			t.Fatalf("insert vector document %s: %v", docID, err)
		}
		if _, err := rs.db.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata,vector_indexed) VALUES(?,?,?,0,'{}',1)`, chunkID, docID, content); err != nil {
			t.Fatalf("insert vector chunk %s: %v", chunkID, err)
		}
		if err := rs.hnsw.Insert(chunkID, vector); err != nil {
			t.Fatalf("insert HNSW vector %s: %v", chunkID, err)
		}
	}
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("rag_chk_other_%02d", i)
		insertChunk(otherCollection.ID, "doc-other-"+fmt.Sprint(i), id, "other.txt", "other collection content", []float32{2, 0, 0, 0})
	}
	targetID := "rag_chk_requested_0"
	insertChunk(targetCollection.ID, "doc-requested", targetID, "target.txt", "requested-only content", []float32{2, 0.1, 0, 0})
	if err := rs.hnsw.Save(dbPath + ".rag.hnsw"); err != nil {
		t.Fatalf("save test HNSW: %v", err)
	}

	results, err := rs.queryInternal("q", 1, "requested")
	if err != nil {
		t.Fatalf("collection-filtered vector query: %v", err)
	}
	if len(results) != 1 || results[0].ChunkID != targetID {
		t.Fatalf("filtered vector results = %+v; want requested chunk despite ten closer other-collection vectors", results)
	}

	// RRF assigns rank weight based on input order, so equal vector scores must
	// be sorted by chunk ID before fusion, regardless of map/HNSW iteration.
	for _, initialIDs := range [][]string{{"rag_chk_tie_c", "rag_chk_tie_a", "rag_chk_tie_b"}, {"rag_chk_tie_b", "rag_chk_tie_c", "rag_chk_tie_a"}} {
		vector := make([]retrievalCandidate, 0, len(initialIDs))
		for _, id := range initialIDs {
			vector = append(vector, retrievalCandidate{result: RAGResult{ChunkID: id, Score: 0.75}})
		}
		sortRAGVectorCandidates(vector)
		fused := fuseRAGCandidates(vector, nil)
		want := []string{"rag_chk_tie_a", "rag_chk_tie_b", "rag_chk_tie_c"}
		if len(fused) != len(want) {
			t.Fatalf("fused tie result count = %d, want %d", len(fused), len(want))
		}
		for i := range want {
			if fused[i].ChunkID != want[i] {
				t.Fatalf("fusion order from %v = %+v; want stable chunk-ID order %v", initialIDs, fused, want)
			}
		}
	}
}

func TestInternalRAGChunkIDExtractionRequiresWholeToken(t *testing.T) {
	chunkID := "rag_chk_doc_001_0"
	if got := internalRAGChunkIDs("lookup " + chunkID + " now"); len(got) != 1 || got[0] != chunkID {
		t.Fatalf("exact chunk ID extraction = %v, want [%s]", got, chunkID)
	}
	for _, nearMiss := range []string{"x" + chunkID, chunkID + "suffix"} {
		for _, got := range internalRAGChunkIDs(nearMiss) {
			if got == chunkID {
				t.Errorf("near-miss %q extracted the embedded ID %q", nearMiss, chunkID)
			}
		}
	}

	rs, _ := newRAGTestStore(t, &fakeRAGEmbedder{dims: 4, available: true}, EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4})
	if err := rs.InitError(); err != nil {
		t.Fatalf("initialize RAG store: %v", err)
	}
	collection, err := rs.CreateCollection("chunk-id-boundaries", "synthetic boundary fixture", 500)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rs.db.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,chunk_count,created_at) VALUES(?,?,?,'text','{}',1,'now')`, "boundary-doc", collection.ID, "boundary.txt"); err != nil {
		t.Fatal(err)
	}
	if _, err := rs.db.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata) VALUES(?,?,?,0,'{}')`, chunkID, "boundary-doc", "ordinary searchable content"); err != nil {
		t.Fatal(err)
	}
	candidates, err := rs.lexicalCandidates(chunkID, "chunk-id-boundaries", 10)
	if err != nil {
		t.Fatalf("lexicalCandidates exact ID: %v", err)
	}
	foundExact := false
	for _, candidate := range candidates {
		if candidate.result.ChunkID == chunkID && candidate.exact {
			foundExact = true
		}
	}
	if !foundExact {
		t.Fatalf("exact chunk ID did not produce an exact candidate: %+v", candidates)
	}
	for _, nearMiss := range []string{"x" + chunkID, chunkID + "suffix"} {
		candidates, err := rs.lexicalCandidates(nearMiss, "chunk-id-boundaries", 10)
		if err != nil {
			t.Fatalf("lexicalCandidates near-miss %q: %v", nearMiss, err)
		}
		for _, candidate := range candidates {
			if candidate.result.ChunkID == chunkID && candidate.exact {
				t.Errorf("near-miss %q incorrectly marked chunk %q exact", nearMiss, chunkID)
			}
		}
	}
}

func TestExactContentIdentifiersCoverStructuredHashURLAndPath(t *testing.T) {
	rs, _ := newRAGTestStore(t, &fakeRAGEmbedder{dims: 4, available: true}, EmbeddingProfile{Provider: "fake", Model: "fake-4", Dimensions: 4})
	collection, err := rs.CreateCollection("content-identifiers", "synthetic identifier fixtures", 500)
	if err != nil {
		t.Fatal(err)
	}
	fixtures := []struct {
		kind, value, contentValue, near string
	}{
		{"structured", "REC-2024-00831", "rec-2024-00831", "REC-2024-00839"},
		{"hash", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdee"},
		{"url", "https://docs.example.test/Guide/CaseSensitive", "HTTPS://Docs.Example.test/Guide/CaseSensitive", "https://docs.example.test/Guide/casesensitive"},
		{"path", "/srv/Docs/Guide.md", "/srv/Docs/Guide.md", "/srv/Docs/guide.md"},
	}
	for _, length := range []int{40, 64, 128} {
		hash := strings.Repeat("a", length)
		fixtures = append(fixtures, struct {
			kind, value, contentValue, near string
		}{"hash", hash, hash, hash[:len(hash)-1] + "b"})
	}
	for i, fixture := range fixtures {
		docID := fmt.Sprintf("identifier-doc-%d", i)
		chunkID := fmt.Sprintf("identifier-chunk-%d", i)
		content := fmt.Sprintf("Reference %s appears in this synthetic record.", fixture.contentValue)
		if _, err := rs.db.Exec(`INSERT INTO rag_documents(id,collection_id,filename,source_type,metadata,chunk_count,created_at) VALUES(?,?,?,'text','{}',1,'now')`, docID, collection.ID, docID+".txt"); err != nil {
			t.Fatalf("insert %s document: %v", fixture.kind, err)
		}
		if _, err := rs.db.Exec(`INSERT INTO rag_chunks(id,document_id,content,chunk_index,metadata) VALUES(?,?,?,0,'{}')`, chunkID, docID, content); err != nil {
			t.Fatalf("insert %s chunk: %v", fixture.kind, err)
		}

		exact, err := rs.lexicalCandidates(fixture.value, "content-identifiers", 10)
		if err != nil {
			t.Fatalf("%s exact lookup: %v", fixture.kind, err)
		}
		if len(exact) == 0 || exact[0].result.ChunkID != chunkID || !exact[0].exact {
			t.Errorf("%s exact results = %+v; want %s exact and prioritized", fixture.kind, exact, chunkID)
		}

		near, err := rs.lexicalCandidates(fixture.near, "content-identifiers", 10)
		if err != nil {
			t.Fatalf("%s near-miss lookup: %v", fixture.kind, err)
		}
		for _, candidate := range near {
			if candidate.result.ChunkID == chunkID && candidate.exact {
				t.Errorf("%s near miss %q incorrectly marked %s exact", fixture.kind, fixture.near, chunkID)
			}
		}
	}
}

func TestUnavailableRAGStoreFailsQueryClosed(t *testing.T) {
	db, err := sql.Open("sqlite3", filepath.Join(t.TempDir(), "memory.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()

	want := errors.New("RAG profile mismatch")
	rs := NewUnavailableRAGStore(db, want)
	_, err = rs.QueryWithRerank("synthetic query", 3, "", &NoopReranker{})
	if !errors.Is(err, want) {
		t.Fatalf("QueryWithRerank() error = %v, want wrapped init error", err)
	}
}

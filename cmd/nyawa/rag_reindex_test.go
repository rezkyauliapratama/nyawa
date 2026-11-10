package main

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestInspectRAGBackfillPlanLegacySchemaTreatsAllChunksAsPending(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "legacy.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE rag_chunks(id TEXT PRIMARY KEY, content TEXT NOT NULL)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO rag_chunks(id,content) VALUES('synthetic-a','alpha'),('synthetic-b','bravo charlie')`); err != nil {
		t.Fatal(err)
	}

	count, chars, err := inspectRAGBackfillPlan(db)
	if err != nil {
		t.Fatalf("inspectRAGBackfillPlan() error = %v", err)
	}
	if count != 2 || chars != len("alpha")+len("bravo charlie") {
		t.Fatalf("plan=(%d chunks,%d chars), want 2 chunks and exact synthetic char count", count, chars)
	}
}

func TestInspectRAGBackfillPlanOnlyCountsUnindexedChunks(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "indexed.db")
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, err := db.Exec(`CREATE TABLE rag_chunks(id TEXT PRIMARY KEY, content TEXT NOT NULL, vector_indexed INTEGER NOT NULL DEFAULT 0)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO rag_chunks(id,content,vector_indexed) VALUES('synthetic-indexed','not-counted',1),('synthetic-pending','counted',0)`); err != nil {
		t.Fatal(err)
	}

	count, chars, err := inspectRAGBackfillPlan(db)
	if err != nil {
		t.Fatalf("inspectRAGBackfillPlan() error = %v", err)
	}
	if count != 1 || chars != len("counted") {
		t.Fatalf("plan=(%d chunks,%d chars), want 1 pending chunk", count, chars)
	}
}

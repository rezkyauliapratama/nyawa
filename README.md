# Nyawa — Offline-First AI Memory Engine

Go-based, SQLite + HNSW + FTS5 + RRF hybrid search.

## Phase 1a

- SQLite store with FTS5 full-text search
- RRF fusion engine (k=60)
- Embedder interface with Ollama HTTP client
- Memory pooling for GC mitigation
- CLI: init, store, recall, search, stats

## Build

```bash
go build -tags "sqlite_fts5" -o nyawa ./cmd/nyawa/
```

## Usage

```bash
./nyawa init nyawa.db
./nyawa store nyawa.db "content here"
./nyawa recall nyawa.db "search query"
./nyawa stats nyawa.db
```

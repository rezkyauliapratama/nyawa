<p align="center">
  <img src="https://img.shields.io/badge/status-stable-brightgreen?style=flat-square" alt="Status">
  <img src="https://github.com/rezkyauliapratama/nyawa/actions/workflows/go-test.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue&style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go&style=flat-square" alt="Go">
  <img src="https://img.shields.io/badge/binary-8.1MB-green?style=flat-square" alt="Size">
  <img src="https://img.shields.io/badge/version-v1.0.0-blue?style=flat-square" alt="Version">
</p>

<h1 align="center">Nyawa</h1>

<p align="center">
  <strong>Offline-First AI Memory Engine</strong><br>
  <em>Give your AI a memory that lasts — no cloud, no Docker, no vector database required.</em>
</p>

<p align="center">
  <i>"nyawa" means "soul" or "spirit" in Indonesian — because memory is the soul of intelligence.</i>
</p>

---

## Why Nyawa?

Most AI memory tools require Docker, external vector databases (Pinecone, Qdrant), cloud APIs, or hundreds of MB of dependencies. Nyawa is different:

- **Single 8.1MB binary** — `go build` and you're done
- **Zero runtime dependencies** — just SQLite
- **100% offline** — all data stays local
- **Fast** — ~11ms search, 22 mems/sec throughput
- **Dual-mode** — Memory (semantic recall) + RAG (document retrieval)

> "Nyawa is what happens when you ask what the simplest thing that could work is — and refuse to add anything else."

---

## Features at a Glance

| Feature | What It Does | Powered By |
|---------|-------------|------------|
| **Hybrid Search** | Semantic + keyword fused via RRF | HNSW (pure Go) + SQLite FTS5 |
| **Entity Graph** | Auto-extract People, Tech, URLs, Locations | 18 regex patterns, zero LLM |
| **Dream Cycle** | 6-phase autonomous memory maintenance | Background goroutine |
| **RAG Engine** | Document-level retrieval with chunking + reranking | HNSW + Jina/Python CrossEncoder |
| **Web Dashboard** | Memory + RAG UI in one page | Go HTTP handler + inline JS |
| **Namespaces** | Isolate memories by context | SQLite namespace column |
| **Time-Travel** | Query memories as they existed at any date | Superseded_at tracking |
| **Batch Import** | Import thousands of memories from JSON | Bulk insert |
| **MCP Protocol** | Plug into any AI agent (10 tools) | Built-in MCP server stdio |

---

## Quick Start (30 seconds)

```bash
# 1. Clone and build
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
make build

# 2. Initialize database
./nyawa init /tmp/nyawa.db

# 3. Store memories
./nyawa store /tmp/nyawa.db "Go backend with PostgreSQL running on GKE"
./nyawa store /tmp/nyawa.db "Team decided to use microservices architecture"
./nyawa store /tmp/nyawa.db "Deploying to production via GitHub Actions"

# 4. Semantic search
./nyawa recall /tmp/nyawa.db "infrastructure architecture"

# 5. Launch the dashboard!
./nyawa serve /tmp/nyawa.db
# Open http://localhost:3300/dashboard
```

**Search results:**
```
#1 [0.9214] Team decided to use microservices architecture
#2 [0.8732] Go backend with PostgreSQL running on GKE
#3 [0.6541] Deploying to production via GitHub Actions
```

---

## RAG — Retrieval-Augmented Generation

Nyawa includes a built-in RAG engine for document-level retrieval:

```bash
# 1. Create a collection
./nyawa rag /tmp/nyawa.db create-collection my-docs --chunk-size 500

# 2. Ingest documents (txt, md, json, csv)
./nyawa rag /tmp/nyawa.db ingest my-docs ./document.md

# 3. Query your documents
./nyawa rag /tmp/nyawa.db query "What does the system architecture look like?"
```

**RAG Pipeline:**
```
Document → Chunking (paragraph-aware) → Embedding → HNSW Index → Reranking (Jina/Python/local)
```

Available via REST API (`/v1/rag/`) or MCP tools (`rag_query`, `rag_ingest_file`, etc.).

---

## Performance

| Metric | Nyawa | Alternative (Qdrant + Docker) |
|--------|-------|------------------------------|
| **Binary size** | **8.1 MB** | ~2 GB (Docker image) |
| **Dependencies** | **0** (SQLite built-in) | Docker, Python, grpc, ... |
| **Search latency** | **~11 ms** | ~5-20 ms (+ network overhead) |
| **Store throughput** | **22 mems/sec** | ~100 mems/sec (batched) |
| **Memory per memory** | **~1.5 KB** | ~2-10 KB |
| **Cold start** | **~2 sec** (load DB) | ~30 sec (container start) |
| **Offline support** | **Native** | Requires network |

---

## Dream Cycle

Nyawa runs a Dream Cycle — a background process that maintains memory automatically:

```
Dream Cycle running every 1h...
 [1/6] Evict      -> Soft-delete stale memories (>90d, low access)
 [2/6] Contra     -> Detect contradictions (like vs dislike)
 [3/6] Dedup      -> Merge near-duplicates (>92% overlap)
 [4/6] Link       -> Strengthen co-occurring entity connections
 [5/6] Prioritize -> Boost popular memories, decay neglected ones
 [6/6] Snapshot   -> Compress old memories into summaries
```

No LLM calls. No API bills. All algorithmic — 100% free and private.

---

## Installation

### From source

```bash
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa && make build
sudo make install   # -> /usr/local/bin/nyawa
```

Requirements: Go 1.23+, gcc (for SQLite CGO)

### Pre-built binary

Download from [Releases](https://github.com/rezkyauliapratama/nyawa/releases):

```bash
curl -L https://github.com/rezkyauliapratama/nyawa/releases/latest/download/nyawa-linux-amd64.gz | gunzip > nyawa
chmod +x ./nyawa
```

### Docker

```bash
docker pull ghcr.io/rezkyauliapratama/nyawa:latest
docker run -d --name nyawa -v ./memory.db:/data/memory.db -p 3300:3300 ghcr.io/rezkyauliapratama/nyawa:latest
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `nyawa init <db>` | Initialize a new database |
| `nyawa store <db> <content>` | Store a memory |
| `nyawa recall <db> <query>` | Semantic search |
| `nyawa import <db> <file.json>` | Batch import from JSON |
| `nyawa stats <db>` | Engine statistics |
| `nyawa ns <db>` | List namespaces |
| `nyawa serve <db>` | Start HTTP server + dashboard |
| `nyawa mcp <db>` | Start MCP server |
| `nyawa dream <db>` | Run Dream Cycle manually |
| `nyawa archive <db> <out>` | Archive old memories |
| `nyawa version` | Check version |

### REST API

**Memory:**
```
POST   /v1/memories            Store a memory
POST   /v1/memories/batch      Batch store
GET    /v1/memories            List (paginated)
GET    /v1/memories/:id        Get by ID
DELETE /v1/memories/:id        Delete
POST   /v1/recall              Search (query, namespace, time_travel)
GET    /v1/stats               Statistics
GET    /v1/health              Health check
GET    /v1/namespaces          List namespaces
DELETE /v1/forget/:id          Forget a memory
```

**RAG:**
```
GET    /v1/rag/collections      List collections
POST   /v1/rag/collections      Create collection
DELETE /v1/rag/collections/:name  Delete collection
POST   /v1/rag/ingest           Ingest file into collection
POST   /v1/rag/query            Query RAG collection
GET    /v1/rag/stats            RAG statistics
```

**Dashboard:**
```
GET    /dashboard              Web dashboard (Memory + RAG)
```

### MCP Tools (10 tools)

**Memory Tools:**
- `nyawa_store` — Store a new memory
- `nyawa_recall` — Semantic search across memories
- `nyawa_stats` — Memory statistics
- `nyawa_forget` — Soft-delete a memory by ID

**RAG Tools:**
- `rag_create_collection` — Create a RAG collection
- `rag_list_collections` — List all RAG collections
- `rag_delete_collection` — Delete a RAG collection
- `rag_ingest_file` — Ingest a file into a collection
- `rag_query` — Query RAG collections for relevant chunks
- `rag_stats` — RAG statistics

---

## Architecture

```
+----------------------------------------------------------+
|              CLI / HTTP / MCP (10 tools)                  |
+----------------------------------------------------------+
|                    Search Pipeline                        |
|   +-------------+  +-----------+  +------------------+   |
|   |   HNSW      |  |  SQLite   |  |  Entity Graph    |   |
|   |  (semantic) |  |  FTS5     |  |  (traversal)     |   |
|   +------+------+  +-----+-----+  +--------+---------+   |
|          +-----------------+------------------+            |
|                    +------+------+                        |
|                    |  RRF Fusion |                        |
|                    +-------------+                        |
+----------------------------------------------------------+
|                    RAG Pipeline                           |
|    Chunking → Embedding → HNSW → Rerank (Jina/Python)    |
+----------------------------------------------------------+
|                    Dream Cycle (background)               |
|            Evict -> Contra -> Dedup -> Link -> Prio -> Snap|
+----------------------------------------------------------+
|                    Embedder Chain                         |
|         BGE (ONNX) <-- priority --> Jina <-- Ollama      |
+----------------------------------------------------------+
|                    SQLite (single file)                   |
|         memories + fts5 + rag_collections + entities      |
+----------------------------------------------------------+
```

---

## Roadmap

| Phase | Status | Features |
|-------|--------|----------|
| Phase 1 | Done | SQLite, FTS5, RRF, CLI, HTTP API, MCP |
| Phase 2 | Done | HNSW, BGE embedder, entity extraction |
| Phase 3 | Done | Entity graph, Dream Cycle |
| Phase 4 | Done | Namespaces, time-travel, archival, dashboard |
| Phase 5 | Done | RAG engine, MCP RAG tools, dashboard RAG UI |
| Phase 6 | Coming | Prometheus metrics, auth, TLS, rate limiting |

---

## Testing

```bash
# Unit tests with race detection
make test

# E2E test suite
make test-e2e

# Build check
make build

# All checks before commit
make commit
```

---

## Contributing

Nyawa is open source and welcoming! See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
1. Fork the repository
2. Create a branch: git checkout -b feat/awesome-feature
3. Commit: git commit -m "feat: add awesome feature"
4. Push: git push origin feat/awesome-feature
5. Open a Pull Request
```

---

## License

MIT (c) [Rezky Aulia Pratama](https://github.com/rezkyauliapratama)

---

<p align="center">
  <sub>Built with love in <a href="https://go.dev/">Go</a> — 8.1MB, 11ms search, RAG + Memory, Dream Cycle, Zero LLM.</sub>
</p>

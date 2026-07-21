<p align="center">
  <img src="https://img.shields.io/badge/status-stable-brightgreen?style=flat-square" alt="Status">
  <img src="https://github.com/rezkyauliapratama/nyawa/actions/workflows/go-test.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue&style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go&style=flat-square" alt="Go">
  <img src="https://img.shields.io/badge/binary-14MB-green?style=flat-square" alt="Size">
  <img src="https://img.shields.io/badge/dependencies-zero-success?style=flat-square" alt="Zero Deps">
</p>

<h1 align="center">
  Nyawa
</h1>

<p align="center">
  <strong>Offline-First AI Memory Engine</strong><br>
  <em>Give your AI a memory that lasts no cloud, no Docker, no vector database required.</em>
</p>

<p align="center">
  <i>"nyawa" means "soul" or "spirit" in Indonesian because memory is the soul of intelligence.</i>
</p>

<br>

---

## Why Nyawa?

Most AI memory tools require:
- Docker and Kubernetes
- External vector databases (Pinecone, Qdrant, Weaviate)
- Cloud APIs / GPU
- Hundreds of MB of dependencies

**Nyawa is different:**
- **Single 14MB binary** go build and you are done
- **Zero dependencies** just SQLite
- **100% offline** all data stays local
- **Fast** 11ms search, 22 mems/sec throughput

> "Nyawa is what happens when you ask what the simplest thing that could work is and refuse to add anything else."

---

## Features at a Glance

| Feature | What It Does | Powered By |
|---------|-------------|------------|
| Hybrid Search | Semantic + keyword fused via RRF | HNSW (pure Go) + SQLite FTS5 |
| Entity Graph | Auto-extract People, Tech, URLs, Locations | 18 regex patterns, zero LLM |
| Dream Cycle | 6-phase autonomous memory maintenance | Background goroutine |
| Web Dashboard | Real-time UI for store, search, browse, delete | Go HTTP handler + Chart.js |
| Namespaces | Isolate memories by context | SQLite namespace column |
| Time-Travel | Query memories as they existed at any date | Superseded_at tracking |
| Batch Import | Import thousands of memories from JSON | Bulk insert |
| MCP Protocol | Plug into any AI agent | Built-in MCP server |

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

## Performance

| Metric | Nyawa | Alternative (Qdrant + Docker) |
|--------|-------|------------------------------|
| **Binary size** | **14 MB** | ~2 GB (Docker image) |
| **Dependencies** | **0** (SQLite built-in) | Docker, Python, grpc, ... |
| **Search latency** | **~11 ms** | ~5-20 ms (+ network overhead) |
| **Store throughput** | **22 mems/sec** | ~100 mems/sec (batched) |
| **Memory per memory** | **~1.5 KB** | ~2-10 KB |
| **Cold start** | **~2 sec** (load DB) | ~30 sec (container start) |
| **Offline support** | **Native** | Requires network |

---

## Dream Cycle

Nyawa runs a Dream Cycle a background process that maintains memory automatically:

```
Dream Cycle running every 1h...
 [1/6] Evict      -> Soft-delete stale memories (>90d, low access)
 [2/6] Contra     -> Detect contradictions (like vs dislike)
 [3/6] Dedup      -> Merge near-duplicates (>92% overlap)
 [4/6] Link       -> Strengthen co-occurring entity connections
 [5/6] Prioritize -> Boost popular memories, decay neglected ones
 [6/6] Snapshot   -> Compress old memories into summaries
```

No LLM calls. No API bills. All algorithmic 100% free and private.

---

## Installation

### From source

```bash
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa && make build
sudo make install   # -> /usr/local/bin/nyawa
```

### Pre-built binary

Download from [Releases](https://github.com/rezkyauliapratama/nyawa/releases):

```bash
curl -L https://github.com/rezkyauliapratama/nyawa/releases/latest/download/nyawa-linux-amd64.gz | gunzip > nyawa
chmod +x ./nyawa
```

> Requirements: Go 1.23+, gcc (for SQLite CGO)

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

```
POST   /v1/memories          Store a memory
POST   /v1/memories/batch    Batch store
GET    /v1/memories          List (paginated)
GET    /v1/memories/:id      Get by ID
DELETE /v1/memories/:id      Delete
POST   /v1/recall            Search (query, namespace, time_travel)
GET    /v1/stats             Statistics
GET    /v1/health            Health check
GET    /v1/namespaces        List namespaces
DELETE /v1/forget/:id        Forget a memory
GET    /dashboard            Web dashboard
```

---

## Architecture

```
+----------------------------------------------------------+
|                    CLI / HTTP / MCP                       |
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
|                    Dream Cycle (background)               |
|            Evict -> Contra -> Dedup -> Link -> Prio -> Snap|
+----------------------------------------------------------+
|                    Embedder Chain                         |
|         BGE (ONNX) <-- priority --> Ollama (fallback)    |
+----------------------------------------------------------+
|                    SQLite (single file)                   |
|         memories + fts5 index + entity_nodes + edges      |
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
| Phase 5 | Coming | Prometheus metrics, auth, TLS, rate limiting |

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

Nyawa is open source and welcoming! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
1. Fork the repository
2. Create a branch: git checkout -b feat/awesome-feature
3. Commit: git commit -m feat add awesome feature
4. Push: git push origin feat/awesome-feature
5. Open a Pull Request
```

---

## License

MIT (c) [Rezky Aulia Pratama](https://github.com/rezkyauliapratama)

---

<p align="center">
  <sub>Built with love in <a href="https://go.dev/">Go</a> because sometimes the smartest solution is the simplest one.</sub>
  <br>
  <sub>14MB | 11ms search | Dream Cycle | Zero LLM</sub>
</p>
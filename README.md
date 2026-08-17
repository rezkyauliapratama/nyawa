<p align="center">
  <img src="https://img.shields.io/badge/status-stable-brightgreen?style=flat-square" alt="Status">
  <img src="https://github.com/rezkyauliapratama/nyawa/actions/workflows/go-test.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue&style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go&style=flat-square" alt="Go">
  <img src="https://img.shields.io/badge/binary-8.1MB-green?style=flat-square" alt="Size">
  <img src="https://img.shields.io/badge/version-v1.1.9-blue?style=flat-square" alt="Version">
</p>

<h1 align="center">Nyawa</h1>

<p align="center">
  <strong>Offline-First AI Memory Engine with GraphRAG</strong><br>
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
- **GraphRAG built-in** — entity graph + multi-hop traversal merged into recall
- **Dual-mode** — Memory (semantic recall) + RAG (document retrieval)

> "Nyawa is what happens when you ask what the simplest thing that could work is — and refuse to add anything else."

---

## Features at a Glance

| Feature | What It Does | Powered By |
|---------|-------------|------------|
| **Hybrid Search** | Semantic + keyword fused via RRF | HNSW (pure Go) + SQLite FTS5 |
| **GraphRAG** | Entity graph + typed edges + multi-hop traversal merged into recall | BFS traversal + regex inference |
| **Entity Graph** | Auto-extract People, Tech, URLs, Locations + 4 typed relations (works_at, uses, located_in, part_of) | Regex patterns, zero LLM |
| **Dream Cycle** | 7-phase autonomous memory + graph maintenance | Background goroutine |
| **RAG Engine** | Document-level retrieval with chunking + reranking | HNSW + Jina/Python CrossEncoder |
| **Web Dashboard** | Memory + RAG UI in one page | Go HTTP handler + inline JS |
| **Namespaces** | Isolate memories by context | SQLite namespace column |
| **Time-Travel** | Query memories as they existed at any date | Superseded_at tracking |
| **Batch Import** | Import thousands of memories from JSON | Bulk insert |
| **MCP Protocol** | Plug into any AI agent (13 tools) | Built-in MCP server stdio |

---

## Quick Start (step by step)

### 0. Prerequisites

| Tool | Version | Why |
|------|---------|-----|
| **Go** | 1.23+ | Build from source |
| **gcc** | any recent | SQLite CGO binding |
| **make** | any | Convenience targets (optional, can use `go build` directly) |

Check your environment:

```bash
go version        # go1.23.x or newer
gcc --version     # any recent version
make --version    # optional but recommended
```

> **Windows note:** use WSL2 or Git Bash. CGO requires gcc (install via MSYS2/MinGW if building natively).

### 1. Clone the repository

```bash
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
```

### 2. Build the binary

```bash
make build
# or, without make:
go build -tags "sqlite_fts5" -ldflags="-s -w" -o nyawa ./cmd/nyawa/
```

You should see a single binary appear:

```bash
ls -lh nyawa        # ~8.1MB
./nyawa version     # nyawa v1.1.9
```

### 3. Initialize a database

```bash
./nyawa init /tmp/nyawa.db
```

This creates a SQLite database file (plus FTS5 index). Output shows empty stats — that's expected:

```json
{"total_memories":0,"entity_nodes":0,"entity_edges":0}
```

### 4. Store your first memories

```bash
./nyawa store /tmp/nyawa.db "Andi bekerja di PT Maju sebagai Data Engineer"
./nyawa store /tmp/nyawa.db "Tim backend menggunakan Kafka untuk streaming data"
./nyawa store /tmp/nyawa.db "Kafka adalah bagian dari platform MCP"
./nyawa store /tmp/nyawa.db "Go backend with PostgreSQL running on GKE"
./nyawa store /tmp/nyawa.db "Team decided to use microservices architecture"
```

Each store returns an ID:

```
Stored: mem_1785720908630614152
```

> **Embedder note:** Nyawa uses an embedder chain (BGE → Ollama → OpenAI-compatible) to generate vectors. If no embedder is available, Nyawa **still works** — it falls back to FTS5 keyword-only search (see [Embedder Setup](#embedder-setup) below).

### 5. Search your memories

```bash
./nyawa recall /tmp/nyawa.db "infrastructure architecture"
./nyawa recall /tmp/nyawa.db "siapa yang kerja di bank?" --ns default
```

Output is ranked results:

```
#1 [0.9214] Team decided to use microservices architecture
#2 [0.8732] Go backend with PostgreSQL running on GKE
#3 [0.6541] Andi bekerja di PT Maju sebagai Data Engineer
```

**Try GraphRAG in action** — query by an entity and watch related memories surface via graph traversal:

```bash
# Traverse the entity graph from query-matched entities
./nyawa graph /tmp/nyawa.db "Kafka" --depth 2 --limit 10
```

You should see memories about Kafka **and** entities connected to it (MCP, Tim backend) — even though the query text doesn't mention them. That's graph-aware recall.

### 6. Launch the web dashboard

```bash
./nyawa serve /tmp/nyawa.db
# Open http://localhost:3300/dashboard
```

The dashboard shows memories, RAG collections, and graph stats in one page.

> **Important — `serve` must keep running for background features:**
> The Dream Cycle (7-phase maintenance) and REST API only run **while `nyawa serve` is alive**.
> CLI commands (`store`, `recall`, `graph`, ...) are one-shot and do NOT start the Dream Cycle.
> For long-term use, run `nyawa serve` as a persistent service — see [Running as a Service](#running-as-a-service).

### 7. (Optional) Run the Dream Cycle manually

```bash
./nyawa dream /tmp/nyawa.db
```

This runs all 7 maintenance phases immediately (instead of waiting for the hourly background cycle):

```
[1/7] Evict      -> Soft-delete stale memories (>90d, low access)
[2/7] Contra     -> Detect contradictions (like vs dislike)
[3/7] Dedup      -> Merge near-duplicates (>92% overlap)
[4/7] Link       -> Strengthen co-occurring entity connections
[5/7] Prioritize -> Boost popular memories, decay neglected ones
[6/7] Snapshot   -> Compress old memories into summaries
[7/7] GraphBuild -> Rebuild co-occurrence edges, prune stale, preserve typed
```

---

## Embedder Setup

Nyawa's semantic search needs **embeddings**. It tries embedders in priority order and falls back gracefully:

| Priority | Embedder | How to enable |
|----------|----------|---------------|
| 1 | **BGE** (local Python/ONNX) | Place model files under `internal/embedder/model/` (model path is configurable in code — see `internal/embedder/py_embedder.go`) |
| 2 | **Ollama** | `ollama pull nomic-embed-text`, then run Ollama on `http://localhost:11434` |
| 3 | **OpenAI-compatible** | Set `EMBEDDING_API_KEY` + `EMBEDDING_BASE_URL` (+ `EMBEDDING_MODEL`) |
| 4 | **None** | Falls back to FTS5 keyword-only search — recall still works, just less semantic |

Re-ranking (RAG): set `JINA_API_KEY` (or `RERANK_API_KEY`) for Jina cross-encoder reranking.

**Recommended quickest path to full semantic search:**

```bash
# Option A: Ollama (easiest)
ollama pull nomic-embed-text

# Option B: OpenAI-compatible API
export EMBEDDING_API_KEY=sk-...
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMBEDDING_MODEL=text-embedding-3-small
```

---

## GraphRAG

Nyawa doesn't just store memories — it learns the **relationships between them** and uses that knowledge during recall.

### How it works

```
Memory store
   └─ Entity extraction (regex, zero LLM)
        └─ Entity nodes (people, tech, places, orgs)
             ├─ Co-occurrence edges (entities seen together in ≥2 memories)
             └─ Typed edges (works_at, uses, located_in, part_of — bilingual ID/EN)
                    └─ Multi-hop BFS traversal (decay 0.5/hop)
                           └─ Merged into RRF recall (boost 1.5x overlap, inject 0.1x graph-only)
```

### Typed relations (auto-inferred, zero LLM)

| Relation | Indonesian | English |
|----------|-----------|---------|
| `works_at` | "Andi **bekerja di** PT Maju" | "Andi **works at** PT Maju" |
| `uses` | "Tim **menggunakan** Kafka" | "Tim **uses** Kafka" |
| `located_in` | "Kantor **berlokasi di** Bandung" | "Office **located in** Bandung" |
| `part_of` | "Kafka **bagian dari** MCP" | "Kafka **part of** MCP" |

### Query it yourself

**CLI:**
```bash
./nyawa graph /tmp/nyawa.db "Kafka" --depth 2 --limit 10
./nyawa graph /tmp/nyawa.db "PT Maju" --depth 3 --limit 20
```

**REST:**
```bash
curl "http://localhost:3300/v1/graph/query?q=Kafka&depth=2&limit=10"
curl "http://localhost:3300/v1/graph/entities?name=Kaf&category=tech"
curl "http://localhost:3300/v1/graph/path?source=Andi&target=MCP&max_depth=4"
```

**MCP tools:** `nyawa_graph_query`, `nyawa_graph_entities`, `nyawa_graph_path`

### Maintenance

The Dream Cycle **Phase 7 (GRAPH BUILD)** rebuilds co-occurrence edges from scratch each cycle:
- Recomputes pair counts from all memories
- **Preserves** typed edges (works_at, uses, etc.)
- Prunes stale co-occurrence edges (count < 2)
- Logs stats: `nodes, edges, avg degree`

---

## RAG — Retrieval-Augmented Generation

Nyawa includes a built-in RAG engine for document-level retrieval. RAG is exposed via **REST API** and **MCP tools** (no CLI subcommand):

**Via REST (requires `nyawa serve`):**

```bash
# 1. Create a collection
curl -X POST http://localhost:3300/v1/rag/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"my-docs","chunk_size":500}'

# 2. Ingest documents (txt, md, json, csv)
curl -X POST http://localhost:3300/v1/rag/ingest \
  -F "collection=my-docs" -F "file=@./document.md"

# 3. Query your documents
curl -X POST http://localhost:3300/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"collection":"my-docs","query":"What does the system architecture look like?"}'
```

**Via MCP tools:** `rag_create_collection`, `rag_list_collections`, `rag_delete_collection`, `rag_ingest_file`, `rag_query`, `rag_stats`

**RAG Pipeline:**
```
Document → Chunking (paragraph-aware) → Embedding → HNSW Index → Reranking (Jina/Python/local)
```

Available via REST API (`/v1/rag/`) or MCP tools (`rag_query`, `rag_ingest_file`, etc.).

---

## Hybrid Search & RRF (Ranked Reciprocal Fusion)

Nyawa runs **two** search strategies on every recall and fuses them into a single final ranking:

- **Vector search** (HNSW) — finds memories by *meaning* (semantic embedding)
- **Keyword search** (SQLite FTS5) — finds memories by *literal terms*

Each has complementary blind spots: vector search excels at synonyms and paraphrases ("how do I buy BTC") but can miss exact identifiers; FTS5 nails exact matches ("mcp-trading-crypto") but ignores semantics. RRF combines both so the final ranking is stronger than either alone.

### The formula

```
score(item) = Σ 1 / (k + rank(item))

k    = constant (60 in Nyawa)
rank = position of the item in each engine's result list
```

Only *positions* matter — raw similarity scores are never compared across engines (they live on different scales). An item that ranks #1 in vector search and #3 in FTS5 gets:

```
1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
```

### Why it works

- Items appearing in **both** result lists accumulate score from both engines → strongly favored
- Items found by only one engine rank lower — they're likely one-sided matches
- No score normalization needed — rank is scale-free and robust to engine drift

### The full recall pipeline (with GraphRAG)

```
Query
 ├─ HNSW vector search ──────────────┐
 ├─ FTS5 keyword search ─────────────┤
 ├─ Entity graph traversal (multi-hop)┤
 └─ RRF fusion (k=60) ───────────────┘
      └─ Graph merge (overlap ×1.5, graph-only ×0.1)
           └─ Filter (namespace, min_score, exclude_types)
                └─ Top-K results
```

Implementation: [`internal/search/rrf.go`](internal/search/rrf.go), [`internal/search/pipeline.go`](internal/search/pipeline.go)

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

Nyawa runs a Dream Cycle — a background process that maintains memory and the entity graph automatically:

```
Dream Cycle running every 2h (random phase offset 0-15m)...
 [1/7] Evict      -> Soft-delete stale memories (>90d, low access)
 [2/7] Contra     -> Detect contradictions (like vs dislike)
 [3/7] Dedup      -> Merge near-duplicates (>92% overlap)
 [4/7] Link       -> Strengthen co-occurring entity connections
 [5/7] Prioritize -> Boost popular memories, decay neglected ones
 [6/7] Snapshot   -> Compress old memories into summaries
 [7/7] GraphBuild -> Rebuild co-occurrence edges, prune stale, preserve typed
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

The image runs `nyawa serve` as its default entrypoint (Dream Cycle active while container is up):

```bash
docker pull ghcr.io/rezkyauliapratama/nyawa:latest
docker run -d --name nyawa --restart unless-stopped \
  -v ./memory.db:/data/memory.db -p 3300:3300 \
  ghcr.io/rezkyauliapratama/nyawa:latest
```

---

## Running as a Service

`nyawa serve` is a **foreground process** — the Dream Cycle (7-phase maintenance), REST API, and dashboard only run while it stays alive. For production / long-term use, run it as a persistent service:

### Option A: systemd (Linux, recommended)

Create `/etc/systemd/system/nyawa.service`:

```ini
[Unit]
Description=Nyawa AI Memory Engine
After=network.target

[Service]
Type=simple
User=nyawa
ExecStart=/usr/local/bin/nyawa serve /var/lib/nyawa/memory.db
Restart=on-failure
RestartSec=5
# Optional: restrict the service
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nyawa
sudo systemctl status nyawa        # verify it's running
```

### Option B: Docker

The Docker image runs `serve` as its default entrypoint, so the container stays alive with the Dream Cycle active:

```bash
docker run -d --name nyawa --restart unless-stopped \
  -v ./memory.db:/data/memory.db -p 3300:3300 \
  ghcr.io/rezkyauliapratama/nyawa:latest
```

`--restart unless-stopped` keeps it running across reboots.

### Option C: tmux / screen / nohup (quick & dirty)

```bash
nohup ./nyawa serve /tmp/nyawa.db > nyawa.log 2>&1 &
```

### Verify it's working

```bash
curl http://localhost:3300/v1/health    # {"status":"ok",...}
journalctl -u nyawa -f                   # watch Dream Cycle logs (systemd)
```

**Expected log pattern** — the Dream Cycle fires every hour:

```
nyawa: Dream Cycle starting
nyawa: Dream graph build: scanned=1234 pairs=89 updated=45 pruned=3 nodes=321 edges=952 avgdeg=5.93
nyawa: Dream Cycle done in 312.4ms: ev=0 ct=0 de=2 lk=5 pr=1 sn=0 gb=45(nd=321 ed=952)
```

If you only run one-shot CLI commands (`store`, `recall`, `graph`) without a persistent `serve`, you get **no Dream Cycle, no REST API** — data is still stored and searchable, but automatic maintenance never runs.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `nyawa init <db>` | Initialize a new database |
| `nyawa store <db> <content>` | Store a memory |
| `nyawa recall <db> <query> [--ns <ns>] [--at <time>]` | Semantic search (alias: `search`) |
| `nyawa import <db> <file.json>` | Batch import from JSON |
| `nyawa stats <db>` | Engine statistics (incl. graph stats) |
| `nyawa ns <db>` | List namespaces |
| `nyawa graph <db> <query> [--depth 2] [--limit 10]` | Traverse entity graph |
| `nyawa serve <db>` | Start HTTP server + dashboard + Dream Cycle |
| `nyawa mcp <db>` | Start MCP server |
| `nyawa dream <db>` | Run Dream Cycle manually (all 7 phases) |
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

**Graph (GraphRAG):**
```
GET    /v1/graph/query?q=...&depth=2&limit=10    Traverse graph from query entities
GET    /v1/graph/entities?name=...&category=...   List/filter entity nodes
GET    /v1/graph/path?source=...&target=...       Find path between two entities
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
GET    /dashboard              Web dashboard (Memory + RAG + Graph stats)
```

### MCP Tools (13 tools)

**Memory Tools:**
- `nyawa_store` — Store a new memory
- `nyawa_recall` — Semantic search across memories
- `nyawa_stats` — Memory statistics
- `nyawa_forget` — Soft-delete a memory by ID

**Graph Tools (GraphRAG):**
- `nyawa_graph_query` — Traverse the entity graph from query-matched seeds
- `nyawa_graph_entities` — List/filter entity nodes
- `nyawa_graph_path` — Find shortest path between two entities

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
||              CLI / HTTP / MCP (13 tools)                 |
+----------------------------------------------------------+
||                    Search Pipeline                       |
||   +-------------+  +-----------+  +------------------+  |
||   |   HNSW      |  |  SQLite   |  |  Entity Graph    |  |
||   |  (semantic) |  |  FTS5     |  | (traverse+typed) |  |
||   +------+------+  +-----+-----+  +--------+---------+  |
||          +-----------------+------------------+          |
||                    +------+------+                      |
||                    |  RRF Fusion |                      |
||                    +-------------+                      |
+----------------------------------------------------------+
||                    RAG Pipeline                         |
||    Chunking → Embedding → HNSW → Rerank (Jina/Python)  |
+----------------------------------------------------------+
||                    Dream Cycle (background)             |
||      Evict→Contra→Dedup→Link→Prio→Snap→GraphBuild      |
+----------------------------------------------------------+
||                    Embedder Chain                       |
||         BGE (ONNX) <-- priority --> Ollama <-- OpenAI   |
+----------------------------------------------------------+
||                    SQLite (single file)                 |
||   memories + fts5 + rag_collections + entity_nodes      |
||   entity_edges + entity_entity_edges + pair_counts      |
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
| Phase 5.1-5.6 | Done | **GraphRAG**: co-occurrence, typed edges, traversal, recall merge, graph API (MCP/REST/CLI), Dream Cycle Phase 7 |
| Phase 6 | Coming | Prometheus metrics, auth, TLS, rate limiting |
| Phase 7 | Planned | LLM hybrid entity extraction (regex coverage ~40-60%) |

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

MIT (c) 2026 Nyawa Contributors

---

<p align="center">
  <sub>Built with love in <a href="https://go.dev/">Go</a> — 8.1MB, 11ms search, GraphRAG + RAG + Memory, Dream Cycle, Zero LLM.</sub>
</p>

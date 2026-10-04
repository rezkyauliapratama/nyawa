<p align="center">
  <img src="https://img.shields.io/badge/status-stable-brightgreen?style=flat-square" alt="Status">
  <img src="https://github.com/rezkyauliapratama/nyawa/actions/workflows/go-test.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue&style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go&style=flat-square" alt="Go">
  <img src="https://img.shields.io/badge/binary-8.1MB-green?style=flat-square" alt="Size">
  <img src="https://img.shields.io/badge/version-v1.2.1-blue?style=flat-square" alt="Version">
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
| **MCP Protocol** | Plug into any AI agent (15 tools) | Built-in MCP server stdio |

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
./nyawa version     # nyawa dev
```

A local build reports `nyawa dev` because the release version is not injected.
To build a binary that reports a real version, inject it the same way the
release workflow does:

```bash
make build VERSION=1.2.1        # -> nyawa v1.2.1
# or, without make:
go build -tags "sqlite_fts5" \
  -ldflags="-s -w -X github.com/rezkyauliapratama/nyawa/internal/version.Version=1.2.1" \
  -o nyawa ./cmd/nyawa/
```

The version literal lives only in `internal/version/version.go` and is
overwritten at link time; a leading `v` on the injected value is stripped, and
an uninjected binary always reports `dev` (see [Version injection](#version-injection)).

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
                           └─ Merged into RRF recall (overlap ×1.1, inject 0.1x graph-only)
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
  -H "Content-Type: application/json" \
  -d '{"file_path":"./document.md","collection":"my-docs"}'

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
- **Keyword search** (SQLite FTS5 with BM25 ranking) — finds memories by *literal terms*

Each has complementary blind spots: vector search excels at synonyms and paraphrases ("how do I buy BTC") but can miss exact identifiers; BM25 over FTS5 nails exact matches ("mcp-trading-crypto") but ignores semantics. RRF combines both so the final ranking is stronger than either alone.

### The two legs

**Vector leg (HNSW).** The query is embedded and the nearest neighbours are
returned. Similarity is computed over the whole memory text, so long documents
can dominate. Part of the corpus may have no vector at all until
`nyawa reindex` fills it in (see [Reindexing](#reindexing-the-vector-index)).

**Keyword leg (FTS5 / BM25).** `memories_fts` is a SQLite FTS5 virtual table;
`ORDER BY rank` uses FTS5's BM25 relevance, so documents that contain more (and
rarer) query terms rank first. The free-text query is **not** passed to `MATCH`
raw. `buildFTSMatchExpr` (`internal/store/sqlite.go`) tokenises it to lowercase
alphanumeric tokens (min length 2, duplicates removed, capped at 32 tokens) and
joins them with `OR`, each token quoted. This makes multi-word queries behave
like OR instead of FTS5's implicit AND, and stops punctuation from being read
as query syntax: a raw `crypto-data` is parsed as a column filter and fails with
`no such column: data`, while `SKILL.md` is invalid syntax. Both symptoms used
to make the keyword leg return **zero rows silently**, leaving recall to the
vector leg alone. If the rewritten `MATCH` still errors, the pipeline logs it
and degrades to vector-only rather than failing the whole query.

### The formula

Weighted Reciprocal Rank Fusion, with the score normalized to `[0,1]` so it is
comparable with the relevance boosts added afterwards:

```
score(item) = [ w_v · 1/(k + rank_v(item)) + w_f · 1/(k + rank_f(item)) ] / ((w_v + w_f)/(k+1))

w_v, w_f = modality weights (vector / FTS5)
k        = damping constant (5 in Nyawa; see below)
rank_*   = position of the item in each engine's result list, ∞ if absent
```

Only *positions* matter — raw similarity scores are never compared across engines (they live on different scales).

### Why k=5, not 60

At the textbook `k=60` the gap between adjacent ranks is tiny, so a memory that
is only *mediocre in both* lists outranks a memory that is the **best hit in
one** list — exactly the dilution that made precise keyword matches (e.g. the
two docker cleanup rules) fall out of the top results. A small `k` sharpens the
fused ranking toward the true top matches. The modality weights default to
`w_v=1`, `w_f=3`: the keyword leg is trusted more because vector similarity is
computed over the whole memory text (long documents dominate it, exact keywords
are missed) and part of the corpus has no HNSW vector. Both legs always
contribute, so a purely semantic match is still retrievable.

### Post-fusion weighting

After fusion, each candidate gets a small, bounded boost so that long or
ephemeral memories no longer win on size alone:

| Signal | Behaviour |
|--------|-----------|
| **Content length** | concise memories keep the full bonus; long ones decay logarithmically |
| **`importance`** | higher column value ranks higher |
| **`access_count`** | saturates at 10 recalls |
| **`mem_type`** | rules/decisions/preferences boosted, `conversation` gets **zero** |
| **recency** | exponential decay with a per-type half-life |
| **pinned / edge_count** | small fixed bonuses |
| **superseded** | always dropped |

Every coefficient is a named constant in
[`internal/search/weights.go`](internal/search/weights.go) and can be overridden
at runtime (no rebuild) with env vars:

| Env var | Default | Meaning |
|---------|---------|---------|
| `NYAWA_RECALL_V2` | `1` | `0`/`false` restores the pre-v2 ranking (symmetric RRF at k=60, no v2 boosts) for A/B comparison |
| `NYAWA_RECALL_RRF_K` | `5` | RRF damping constant |
| `NYAWA_RECALL_W_VECTOR` / `NYAWA_RECALL_W_FTS` | `1` / `3` | modality weights |
| `NYAWA_RECALL_W_TYPE` / `_W_IMPORTANCE` / `_W_ACCESS` / `_W_LENGTH` / `_W_RECENCY` | `0.12` / `0.10` / `0.05` / `0.05` / `0.05` | boost weights |
| `NYAWA_RECALL_W_PINNED` / `_W_GRAPH` / `_W_OVERLAP` / `_W_GRAPH_INJECT` | `0.10` / `0.03` / `0.10` / `0.10` | graph + pinned weights |

### The full recall pipeline (with GraphRAG)

```
Query
 |-- embed query
 |-- HNSW vector search
 |-- FTS5 / BM25 keyword search   (query tokenised + OR'd, so multi-word
 |                                 and hyphenated queries match)
 +-- Weighted RRF fusion (k=5)
      +-- Fetch memories, drop superseded_at
           +-- Post-fusion weighting (length, importance, access, type, recency, pinned, edge_count)
                +-- Graph merge (query-matched entity seeds; overlap x1.1, graph-only x0.1)
                     +-- Time-travel filter (if --at)
                          +-- Filter (min_score, exclude_types)
                               +-- Top-K cut (limit means "at most K after filtering")
                                    +-- access_count incremented asynchronously
```

The entity graph is merged **after** fusion, not as a third RRF leg: entity
names that appear in the query seed a multi-hop traversal, memories reachable
both ways get the overlap boost, and graph-only memories are injected at the
0.1 weight.

Implementation: [`internal/search/rrf.go`](internal/search/rrf.go), [`internal/search/pipeline.go`](internal/search/pipeline.go). For the full detail, weights, and benchmark numbers see [`docs/recall-pipeline.md`](docs/recall-pipeline.md).

### Result filters: `min_score` and `exclude_types`

`nyawa_recall`, `POST /v1/recall`, and the HTTP API accept two optional
post-ranking filters (the CLI takes a query, namespace and time only):

| Field | JSON key | Behaviour |
|-------|----------|-----------|
| Minimum score | `min_score` | Drops every result whose final `score` is strictly below the threshold. `0` (the default) disables the filter; it is not clamped, so a negative value also disables it. |
| Excluded types | `exclude_types` | Drops results whose `type` is in the list, e.g. `["note","conversation"]`. |

Both run **after** fusion, post-fusion weighting, and the graph merge, and
**before** the top-K cut, so `limit` means "at most this many results after
filtering". `min_score` compares against the final score, which is the
normalised RRF score in `[0,1]` plus the small bounded boosts (see the table
above), so a sensible threshold is well under `1` (roughly `0.1` to `0.3`).
`0` disables it rather than selecting everything.

### Reindexing the vector index

The HNSW vector index lives beside the database as `<db>.hnsw`. Memories stored
while no embedder was available, or imported/bulk-loaded, can be missing their
vector. `nyawa reindex <db>` walks every active memory, embeds the ones that are
not yet in the index (HNSW membership is checked first, so existing vectors are
never duplicated), and persists the updated index through `HNSW.MergeAndSave`,
which reloads the on-disk index under a cross-process lock and writes the union
atomically so a running gateway's concurrent writes are not lost.

```bash
./nyawa reindex /tmp/nyawa.db
```

The command needs a live embedder: without one it logs `BGE unavailable`, the
embedding calls fail, and the memories that still need a vector are counted as
failed rather than reindexed. Example output:

```
nyawa: hnsw: nothing to reindex, 8386 vectors in memory (coverage 187.5% of 4473 active memories)
Reindexed 0 memories (0 already indexed, 0 failed)
```

Coverage is `persisted / total active memories * 100`; values above 100% are
expected because the index can retain vectors for memories that are no longer
active. Reindex is safe to run against a live database.

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

## Version injection

The version string is not hardcoded in the source. `internal/version/version.go`
declares `var Version = DefaultVersion` (where `DefaultVersion = "dev"`), and the
release workflow overwrites it at link time from the pushed git tag:

```
-ldflags "-X github.com/rezkyauliapratama/nyawa/internal/version.Version=$(TAG without leading v)"
```

`.github/workflows/release.yml` runs on any `v*` tag push, strips the leading
`v` (`v1.2.1` becomes `VERSION=1.2.1`), injects it, and then **verifies** the
built binary: it runs `nyawa version` and fails the workflow unless the output
equals `v<tag>`. That check exists because a build that reported a stale version
once shipped. `version.Number()` strips a leading `v` from whatever was injected
and falls back to `dev` for a blank value, so an uninjected binary never
masquerades as a release. Everything that reports a version (`nyawa version`,
the CLI banner, the MCP `serverInfo.version`, and the HTTP `/`, `/v1/stats`,
`/v1/health` endpoints) reads from this single source.

Local builds default to `dev`. To inject a version the same way:

```bash
make build VERSION=1.2.1
# or:
go build -tags "sqlite_fts5" \
  -ldflags="-X github.com/rezkyauliapratama/nyawa/internal/version.Version=1.2.1" \
  -o nyawa ./cmd/nyawa/
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
| `nyawa reindex <db>` | Re-embed active memories missing from the HNSW index |
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
POST   /v1/rag/ingest           Ingest file into collection (JSON: {"file_path": "...", "collection": "..."})
POST   /v1/rag/query            Query RAG collection (JSON: {"query": "...", "collection": "...", "top_k": 5})
GET    /v1/rag/stats            RAG statistics
```

**Dashboard:**
```
GET    /dashboard              Web dashboard (Memory + RAG + Graph stats)
```

### MCP Tools (15 tools)

**Memory Tools:**
- `nyawa_store` — Store a new memory
- `nyawa_recall` — Semantic search across memories (`query`, `namespace`, `limit`, `min_score`, `exclude_types`)
- `nyawa_list` — Deterministically list memories filtered by namespace/type, ordered by `created_at` (no semantic search)
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

**Context Tools:**
- `compact_context` — Compress a conversation transcript into a compact summary block (stores it under `namespace=context`; LLM summarization when `NYAWA_LLM_API_KEY` is set, deterministic fallback otherwise)

---

## Architecture

```
+----------------------------------------------------------+
||              CLI / HTTP / MCP (15 tools)                 |
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

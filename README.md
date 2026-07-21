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
  <em>Beri AI-mu ingatan yang bertahan tanpa cloud, tanpa Docker, tanpa vector database.</em>
</p>

<p align="center">
  <b>Bahasa Indonesia:</b> <i>"nyawa" = soul, spirit, life karena memory adalah jiwa dari intelligence.</i>
</p>

<br>

---

## Kenapa Nyawa?

Kebanyakan AI memory tools butuh:
- Docker & Kubernetes
- Vector database terpisah (Pinecone, Qdrant, Weaviate)
- Cloud API / GPU
- Ratusan MB dependency

**Nyawa berbeda:**
- **Single 14MB binary** cukup `go build` langsung jadi
- **Zero dependencies** cukup SQLite
- **100% offline** semua data di lokal
- **Cepat** search 11ms, store 22 mem/detik

> "Nyawa is what happens when you ask 'what's the simplest thing that could work?' and refuse to add anything else."

---

## Fitur Unggulan

| Fitur | Apa yang Dilakukan | Teknologi |
|-------|-------------------|-----------|
| Hybrid Search | Semantic + keyword digabung dengan RRF fusion | HNSW (Go murni) + SQLite FTS5 |
| Entity Graph | Ekstrak otomatis People, Tech, URLs, Locations | 18 regex patterns, zero LLM |
| Dream Cycle | Maintenance 6 fase clean up, dedup, link, prioritaskan | Background goroutine |
| Web Dashboard | UI real-time untuk store, search, browse, delete | Go HTTP handler, Chart.js |
| Namespace | Isolasi memori per konteks | SQLite namespace column |
| Time-Travel | Lihat masa lalu query memori pada tanggal tertentu | Superseded_at tracking |
| Batch Import | Import ribuan memori dari JSON | Bulk insert |
| MCP Protocol | Integrasi dengan AI agent manapun | MCP server built-in |

---

## Quick Start (30 detik)

```bash
# 1. Clone and build
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
make build

# 2. Init database
./nyawa init /tmp/nyawa.db

# 3. Simpan memori
./nyawa store /tmp/nyawa.db "Go backend with PostgreSQL running on GKE"
./nyawa store /tmp/nyawa.db "Team decided to use microservices architecture"
./nyawa store /tmp/nyawa.db "Deploy ke production pake GitHub Actions"

# 4. Cari secara semantic
./nyawa recall /tmp/nyawa.db "infrastructure architecture"

# 5. Dashboard!
./nyawa serve /tmp/nyawa.db
# Buka http://localhost:3300/dashboard
```

**Hasil search:**
```
#1 [0.9214] Team decided to use microservices architecture
#2 [0.8732] Go backend with PostgreSQL running on GKE
#3 [0.6541] Deploy ke production pake GitHub Actions
```

---

## Performance

| Metric | Nyawa | Alternatif (Qdrant + Docker) |
|--------|-------|------------------------------|
| **Binary size** | **14 MB** | ~2 GB (Docker image) |
| **Dependencies** | **0** (SQLite built-in) | Docker, Python, grpc, ... |
| **Search latency** | **~11 ms** | ~5-20 ms (+ network) |
| **Store throughput** | **22 mem/detik** | ~100 mem/detik (batch) |
| **Memory per memory** | **~1.5 KB** | ~2-10 KB |
| **Cold start** | **~2 detik** (load DB) | ~30 detik (start container) |
| **Offline support** | **Native** | Butuh network |

---

## Dream Cycle

Nyawa punya Dream Cycle proses background yang merawat memori secara mandiri:

```
Dream Cycle running every 1h...
 [1/6] Evict      -> Hapus memori stale (>90d, jarang diakses)
 [2/6] Contra     -> Deteksi kontradiksi (suka vs tidak suka)
 [3/6] Dedup      -> Merge near-duplicates (>92% overlap)
 [4/6] Link       -> Perkuat koneksi entity yang sering co-occur
 [5/6] Prioritize -> Boost memori populer, decay yang diabaikan
 [6/6] Snapshot   -> Kompres memori lama jadi ringkasan
```

Tanpa LLM. Tanpa API. Semua algorithmic 100% gratis dan private.

---

## Installasi

### Linux / macOS

```bash
# Dari source (recommended)
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa && make build
sudo make install   # -> /usr/local/bin/nyawa
```

### Pre-built binary

Download dari [Releases](https://github.com/rezkyauliapratama/nyawa/releases):

```bash
curl -L https://github.com/rezkyauliapratama/nyawa/releases/latest/download/nyawa-linux-amd64.gz | gunzip > nyawa
chmod +x ./nyawa
```

---

## CLI Reference

| Perintah | Fungsi |
|----------|--------|
| `nyawa init <db>` | Init database baru |
| `nyawa store <db> <content>` | Simpan memori |
| `nyawa recall <db> <query>` | Semantic search |
| `nyawa import <db> <file.json>` | Batch import dari JSON |
| `nyawa stats <db>` | Statistik engine |
| `nyawa ns <db>` | List namespaces |
| `nyawa serve <db>` | Start server + dashboard |
| `nyawa mcp <db>` | Start MCP server |
| `nyawa dream <db>` | Manual Dream Cycle |
| `nyawa archive <db> <out>` | Archive memori lama |
| `nyawa version` | Cek versi |

### API Endpoints

```
POST   /v1/memories          Store memory
POST   /v1/memories/batch    Batch store
GET    /v1/memories          List (paginated)
GET    /v1/memories/:id      Get by ID
DELETE /v1/memories/:id      Delete
POST   /v1/recall            Search (query, namespace, time_travel)
GET    /v1/stats             Statistics
GET    /v1/health            Health check
GET    /v1/namespaces        List namespaces
DELETE /v1/forget/:id        Forget memory
GET    /dashboard            Web dashboard
```

---

## Arsitektur

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
|                    Dream Cycle (bg)                       |
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

| Fase | Status | Fitur |
|------|--------|-------|
| Phase 1 | Selesai | SQLite, FTS5, RRF, CLI, HTTP API, MCP |
| Phase 2 | Selesai | HNSW, BGE embedder, entity extraction |
| Phase 3 | Selesai | Entity graph, Dream Cycle |
| Phase 4 | Selesai | Namespaces, time-travel, archival, dashboard |
| Phase 5 | Coming | Metrics (Prometheus), auth, TLS, rate limiting |

---

## Testing

```bash
# Unit tests + race detection
make test

# E2E test suite
make test-e2e

# Build check
make build

# All in one
make commit
```

---

## Kontribusi

Nyawa open-source dan welcoming! Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan.

```
1. Fork repository
2. Buat branch: git checkout -b feat/keren-banget
3. Commit: git commit -m 'feat: tambah fitur keren'
4. Push: git push origin feat/keren-banget
5. Buka Pull Request
```

---

## License

MIT (c) [Rezky Aulia Pratama](https://github.com/rezkyauliapratama)

---

<p align="center">
  <sub>Dibuat dengan hati pake <a href="https://go.dev/">Go</a> karena kadang solusi paling cerdas adalah yang paling sederhana.</sub>
  <br>
  <sub>14MB | 11ms search | Dream Cycle | Zero LLM</sub>
</p>
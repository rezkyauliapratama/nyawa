<div align="center">
  <h1>Nyawa</h1>
  <p><strong>Offline-First AI Memory Engine</strong></p>
  <p>
    <a href="https://github.com/rezkyauliapratama/nyawa/actions"><img src="https://img.shields.io/github/actions/workflow/status/rezkyauliapratama/nyawa/go-test.yml?branch=main&label=CI&logo=github" alt="CI"></a>
    <a href="https://github.com/rezkyauliapratama/nyawa/blob/main/LICENSE"><img src="https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue" alt="License"></a>
    <a href="https://github.com/rezkyauliapratama/nyawa/releases"><img src="https://img.shields.io/github/v/release/rezkyauliapratama/nyawa?include_prereleases&label=release" alt="Release"></a>
    <a href="https://go.dev/"><img src="https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go" alt="Go"></a>
    <img src="https://img.shields.io/badge/binary-14MB-green" alt="Size">
    <a href="https://github.com/rezkyauliapratama/nyawa/stargazers"><img src="https://img.shields.io/github/stars/rezkyauliapratama/nyawa?style=flat&logo=github" alt="Stars"></a>
  </p>
  <p>
    <b>Indonesian</b> - <i>"nyawa" means "soul" or "spirit"</i>
  </p>
  <br>
</div>

---

**Nyawa** is a portable, offline-first memory engine for AI agents. It stores, searches, and maintains memories using hybrid search (semantic + keyword), entity graph traversal, and a proactive Dream Cycle that consolidates knowledge while idle.

> **Key Philosophy:** Zero external dependencies. No Docker. No external vector databases. Just a single 14MB Go binary with SQLite.

---

## Quick Start

```bash
# Install (Linux x86_64)
curl -L https://github.com/rezkyauliapratama/nyawa/releases/latest/download/nyawa-linux-amd64.gz | gunzip > nyawa
chmod +x ./nyawa

# Or build from source
git clone https://github.com/rezkyauliapratama/nyawa.git
cd nyawa
make build

# Initialize and store memories
./nyawa init /tmp/nyawa.db
./nyawa store /tmp/nyawa.db "Go backend with PostgreSQL running on GKE"
./nyawa store /tmp/nyawa.db "Team decided to use microservices architecture"

# Semantic search
./nyawa recall /tmp/nyawa.db "infrastructure architecture"

# Start web dashboard
./nyawa serve /tmp/nyawa.db
# Open http://localhost:3300/dashboard
```

---

## Features

### Hybrid Search
Combines semantic (HNSW vector index) and keyword (SQLite FTS5) search with Reciprocal Rank Fusion (RRF) for best-in-class recall.

### Entity Graph
Automatically extracts entities (people, technologies, URLs) from memories and builds a traversable knowledge graph.

### Dream Cycle
Six-phase automatic memory maintenance that runs in the background: Evict, Contradiction, Dedup, Link, Prioritize, Snapshot.

### Web Dashboard
Built-in dashboard at /dashboard with stats, search, browse, and delete. No frontend framework needed.

### Namespace Isolation
Organize memories into isolated namespaces (backend, infra, personal).

### Time-Travel Queries
Query memories as they existed at any point in time.
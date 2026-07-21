# Nyawa

**Offline-First AI Memory Engine**

![CI](https://github.com/rezkyauliapratama/nyawa/actions/workflows/go-test.yml/badge.svg)
![License](https://img.shields.io/github/license/rezkyauliapratama/nyawa?color=blue)
![Go](https://img.shields.io/badge/Go-1.23+-00ADD8?logo=go)
![Size](https://img.shields.io/badge/binary-14MB-green)

Nyawa is a portable, offline-first memory engine for AI agents.

> Zero external dependencies. No Docker. No external vector databases. Single 14MB Go binary with SQLite.

### Quick Start

```bash
# Build from source
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

### Features

- Hybrid search: HNSW vector + SQLite FTS5 + RRF fusion
- Entity graph: auto-extract, traverse, boost
- Dream Cycle: 6-phase autonomous memory maintenance
- Web dashboard: real-time UI at /dashboard
- Namespace isolation: organize memories by context
- Time-travel queries: recall past states
- Batch import: JSON file or stdin
- MCP protocol: AI agent integration

### Docs

Full README: [github.com/rezkyauliapratama/nyawa](https://github.com/rezkyauliapratama/nyawa)
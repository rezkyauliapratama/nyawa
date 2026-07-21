# Contributing to Nyawa

Thank you for your interest! Nyawa is an offline-first AI memory engine, and every contribution helps.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/nyawa.git`
3. Install Go 1.23+ and gcc (for SQLite)
4. Build: `go build -tags "sqlite_fts5" -o nyawa ./cmd/nyawa/`
5. Run tests: `go test -tags "sqlite_fts5" ./...`

## Development Workflow

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Make changes
3. Run lint: `gofmt -s -w .`
4. Run tests: `go test -tags "sqlite_fts5" -count=1 -race ./...`
5. Build: `go build -tags "sqlite_fts5" -o nyawa ./cmd/nyawa/`
6. E2E test:
   ```bash
   ./nyawa init /tmp/test.db
   ./nyawa store /tmp/test.db "Test memory"
   ./nyawa recall /tmp/test.db "Test"
   ./nyawa stats /tmp/test.db
   ```
7. Commit and push, then open a PR

## Code Style

- Follow standard Go conventions (`gofmt -s`)
- Organize imports: std backu2192 external backu2192 internal
- Use `_` for unused parameters instead of commenting them out
- Prefer `sync.Pool` for hot-path allocations
- Use `types.SearchConfig` for pipeline tuning parameters
- Document exported functions with comments

## Project Structure

```
cmd/nyawa/          — CLI entry point
internal/
  dream/            — Dream Cycle (memory maintenance)
  embedder/         — Embedding (BGE, Ollama, priority chain)
  extract/          — Entity and type extraction
  graph/            — Entity graph (nodes, edges, traversal)
  index/            — HNSW vector index
  mcp/              — MCP protocol server
  pool/             — Memory pools
  search/           — Search pipeline (RRF, FTS5, post-processing)
  security/         — Content security filter
  server/           — HTTP API server + dashboard
  store/            — SQLite store
  types/            — Shared types
```

## Testing

- Unit tests: `go test -tags "sqlite_fts5" ./internal/...`
- Race detection: `go test -tags "sqlite_fts5" -race ./internal/...`
- E2E: Run `test_nyawa.sh`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
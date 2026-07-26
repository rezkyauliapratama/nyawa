# Nyawa — Offline-First AI Memory Engine

## MCP Fix: CallToolResult Format Compliance

**PR #23** — Fixes MCP response format to comply with MCP SDK v1.26+ `CallToolResult` spec.

### Problem
Nyawa's MCP server returned tool results as raw data directly in `result` field:
```json
{"result": {"count": 192, "results": [...]}}
```
Hermes MCP client requires `content` array wrapper:
```json
{"result": {"content": [{"type": "text", "text": "{\"count\":192,...}"}]}}
```

### Fix
- Added `callToolResult` + `contentItem` structs in `internal/mcp/server.go`
- New `writeToolResult()` method wraps responses in standard `CallToolResult` format
- All tool handlers (store/recall/stats/forget) use `writeToolResult`
- Non-tool methods (initialize, tools/list) use `writeResult` unchanged

### Files Changed
- `internal/mcp/server.go` — +20 lines (structs + writeToolResult)

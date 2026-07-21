# Nyawa MCP Skill -- AI Agent Memory Integration

Give any AI agent (Hermes, Claude Code, Codex, Cline) long-term memory using Nyawa's MCP protocol.

## Quick Start

```bash
# 1. Start Nyawa MCP server in background
nyawa init ~/.nyawa/memory.db
nyawa mcp ~/.nyawa/memory.db &
MCP_PID=$!

# 2. Test connection
echo '{"jsonrpc":"2.0","id":1,"method":"initialize"}' | nyawa mcp ~/.nyawa/memory.db

# 3. Store a memory
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nyawa_store","arguments":{"content":"User prefers concise responses","namespace":"preferences","type":"preference"}}}' | nyawa mcp ~/.nyawa/memory.db

# 4. Search memories
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"nyawa_recall","arguments":{"query":"user preferences","namespace":"preferences","limit":5}}}' | nyawa mcp ~/.nyawa/memory.db
```

## MCP Protocol

Nyawa uses **JSON-RPC 2.0** over **stdio**. Each line = one request, one response.

### Step 1: Initialize

```json
{"jsonrpc":"2.0","id":1,"method":"initialize"}
```

### Step 2: List Tools

```json
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

### Step 3: Use Tools

#### nyawa_store -- Store a memory

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "nyawa_store",
    "arguments": {
      "content": "Go backend with PostgreSQL on GKE",
      "namespace": "infra",
      "type": "fact"
    }
  }
}
```

#### nyawa_recall -- Semantic search

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "nyawa_recall",
    "arguments": {
      "query": "database infrastructure",
      "namespace": "infra",
      "limit": 5
    }
  }
}
```

#### nyawa_stats -- Engine statistics

```json
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"nyawa_stats","arguments":{}}}
```

#### nyawa_forget -- Delete a memory

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "nyawa_forget",
    "arguments": { "id": "mem_1234567890" }
  }
}
```

## Integration Examples

### Hermes Agent

Add to your Hermes workflow:

```yaml
# .hermes/workflows/memory.yaml
name: memory
steps:
  - name: store
    tool: nyawa_store
    params:
      content: "{message}"
      namespace: "default"

  - name: recall
    tool: nyawa_recall
    params:
      query: "{query}"
      limit: 5
```

Or use the MCP server directly in any Hermes session:

```
nyawa store ~/.nyawa/db "Important information to remember"
nyawa recall ~/.nyawa/db "What did I learn about X?"
```

### Claude Code / Codex

```bash
# Start MCP server before Claude Code
nyawa mcp ~/.nyawa/memory.db &

# Claude Code can then use the tools via stdio MCP
claude
```

### Custom Agent (Python)

```python
import subprocess
import json

class NyawaMCP:
    def __init__(self, db_path: str):
        self.proc = subprocess.Popen(
            ["nyawa", "mcp", db_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        self._call("initialize")

    def _call(self, method: str, params: dict = None) -> dict:
        req = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params:
            req["params"] = params
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def store(self, content: str, namespace="default", mem_type="note"):
        return self._call("tools/call", {
            "name": "nyawa_store",
            "arguments": {"content": content, "namespace": namespace, "type": mem_type}
        })

    def recall(self, query: str, namespace="", limit=10):
        args = {"query": query, "limit": limit}
        if namespace:
            args["namespace"] = namespace
        return self._call("tools/call", {
            "name": "nyawa_recall",
            "arguments": args
        })

    def stats(self):
        return self._call("tools/call", {"name": "nyawa_stats", "arguments": {}})

    def close(self):
        self.proc.kill()

# Usage
mcp = NyawaMCP("~/.nyawa/memory.db")
mcp.store("Deploy ke production pake GitHub Actions", "infra")
results = mcp.recall("deployment strategy")
print(results)
mcp.close()
```

### Shell Script (any agent)

```bash
#!/bin/bash
# Simple wrapper using the bin/nyawa-mcp.sh helper
bin/nyawa-mcp.sh ~/.nyawa/memory.db store '{"content":"test","namespace":"default"}'
bin/nyawa-mcp.sh ~/.nyawa/memory.db recall '{"query":"test","limit":5}'
bin/nyawa-mcp.sh ~/.nyawa/memory.db stats
```

## Tips

- **Namespace isolation**: Use separate namespaces for different contexts (work, personal, projects)
- **Memory types**: Use appropriate types for better filtering (fact, insight, decision, etc.)
- **Batch import**: For bulk loading, use `nyawa import <db> <file.json>` instead of individual MCP calls
- **Dream Cycle**: Run `nyawa dream <db>` periodically or enable it via `nyawa serve` for automatic memory maintenance
- **Database location**: Keep your DB in `~/.nyawa/` for consistency across sessions

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `command not found: nyawa` | Build from source: `cd nyawa && make build && sudo make install` |
| MCP server won't start | Ensure DB exists: `nyawa init <db>` |
| Search returning 0 results | Memories need vector embedding -- check embedder is running |
| `Invalid arguments` on recall | Make sure `limit` is a number, not a string |
| Memory not persisted across sessions | Use an absolute path for the database file |
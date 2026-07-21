#!/bin/bash
# Nyawa MCP Wrapper -- JSON-RPC client for any AI agent
# Usage: ./bin/nyawa-mcp.sh <db> <method> [json_args]
# 
# Examples:
#   nyawa init ~/.nyawa/db
#   bin/nyawa-mcp.sh ~/.nyawa/db initialize
#   bin/nyawa-mcp.sh ~/.nyawa/db store '{"content":"Hello world","namespace":"default","type":"note"}'
#   bin/nyawa-mcp.sh ~/.nyawa/db recall '{"query":"hello"}'
#   bin/nyawa-mcp.sh ~/.nyawa/db stats
#   bin/nyawa-mcp.sh ~/.nyawa/db forget '{"id":"mem_123"}'

set -e

DB="$1"
METHOD="$2"
ARGS="${3:-{}}"
ID="${4:-1}"

if [ -z "$DB" ] || [ -z "$METHOD" ]; then
  echo "Usage: $0 <db> <method> [json_args] [request_id]"
  echo "Methods: initialize, store, recall, stats, forget, tools"
  echo ""
  echo "Shortcut methods (auto-mapped to tools/call):"
  echo "  store  -> nyawa_store"
  echo "  recall -> nyawa_recall"
  echo "  stats  -> nyawa_stats"
  echo "  forget -> nyawa_forget"
  echo "  tools  -> tools/list"
  exit 1
fi

# Map shortcuts
case "$METHOD" in
  initialize)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"initialize"}'
    ;;
  tools)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"tools/list"}'
    ;;
  store)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"tools/call","params":{"name":"nyawa_store","arguments":'"$ARGS"'}}'
    ;;
  recall)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"tools/call","params":{"name":"nyawa_recall","arguments":'"$ARGS"'}}'
    ;;
  stats)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"tools/call","params":{"name":"nyawa_stats","arguments":{}}}'
    ;;
  forget)
    REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"tools/call","params":{"name":"nyawa_forget","arguments":'"$ARGS"'}}'
    ;;
  *)
    if [ "$ARGS" = "{}" ]; then
      REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"'$METHOD'"}'
    else
      REQUEST='{"jsonrpc":"2.0","id":'$ID',"method":"'$METHOD'","params":'"$ARGS"'}'
    fi
    ;;
esac

echo "$REQUEST" | nyawa mcp "$DB" 2>/dev/null | head -1
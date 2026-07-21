package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Server struct {
	store  *store.Store
	reader *bufio.Scanner
	writer *json.Encoder
}

func NewServer(st *store.Store) *Server {
	return &Server{store: st, reader: bufio.NewScanner(os.Stdin), writer: json.NewEncoder(os.Stdout)}
}

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string     `json:"jsonrpc"`
	ID      any        `json:"id"`
	Result  any        `json:"result,omitempty"`
	Error   *rpcError  `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type toolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	InputSchema inputSchema `json:"inputSchema"`
}

type inputSchema struct {
	Type       string                    `json:"type"`
	Properties map[string]propertySchema `json:"properties"`
	Required   []string                  `json:"required,omitempty"`
}

type propertySchema struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

func (s *Server) tools() []toolDefinition {
	return []toolDefinition{
		{Name: "nyawa_store", Description: "Store a new memory. Returns the memory ID and classification status.",
			InputSchema: inputSchema{Type: "object",
				Properties: map[string]propertySchema{
					"content":   {Type: "string", Description: "Memory content to store"},
					"namespace": {Type: "string", Description: "Namespace (default: 'default')"},
					"type":      {Type: "string", Description: "Memory type: decision, insight, procedure, fact, preference, context, note, event, reference", Enum: []string{"decision", "insight", "procedure", "fact", "preference", "context", "note", "event", "reference"}},
				},
				Required: []string{"content"},
			}},
		{Name: "nyawa_recall", Description: "Search memories by query. Uses FTS5 keyword search, falls back gracefully if vector embedder unavailable.",
			InputSchema: inputSchema{Type: "object",
				Properties: map[string]propertySchema{
					"query":     {Type: "string", Description: "Natural language search query"},
					"namespace": {Type: "string", Description: "Namespace filter"},
					"limit":     {Type: "string", Description: "Max results (default: 10)"},
				},
				Required: []string{"query"},
			}},
		{Name: "nyawa_stats", Description: "Get memory statistics: total memories, pinned count, FTS5 index size.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{}}},
		{Name: "nyawa_forget", Description: "Soft-delete a memory by ID. Sets superseded_at timestamp.",
			InputSchema: inputSchema{Type: "object",
				Properties: map[string]propertySchema{"id": {Type: "string", Description: "Memory ID to delete"}},
				Required:   []string{"id"},
			}},
	}
}

func (s *Server) Run() error {
	log.Println("Nyawa MCP server started (stdio)")
	for s.reader.Scan() {
		line := s.reader.Text()
		if line == "" {
			continue
		}
		var req jsonRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			s.writeError(nil, -32700, "Parse error")
			continue
		}
		s.handleRequest(req)
	}
	return s.reader.Err()
}

func (s *Server) handleRequest(req jsonRPCRequest) {
	switch req.Method {
	case "initialize":
		s.writeResult(req.ID, map[string]any{"protocolVersion": "2025-03-26", "capabilities": map[string]any{"tools": map[string]bool{"listChanged": false}}, "serverInfo": map[string]string{"name": "nyawa", "version": "0.3.0"}})
	case "tools/list":
		s.writeResult(req.ID, map[string]any{"tools": s.tools()})
	case "tools/call":
		s.handleToolCall(req)
	default:
		s.writeError(req.ID, -32601, fmt.Sprintf("Method not found: %s", req.Method))
	}
}

type callParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func (s *Server) handleToolCall(req jsonRPCRequest) {
	var params callParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.writeError(req.ID, -32602, "Invalid params"); return
	}
	switch params.Name {
	case "nyawa_store": s.handleStore(req.ID, params.Arguments)
	case "nyawa_recall": s.handleRecall(req.ID, params.Arguments)
	case "nyawa_stats": s.handleStats(req.ID)
	case "nyawa_forget": s.handleForget(req.ID, params.Arguments)
	default: s.writeError(req.ID, -32601, fmt.Sprintf("Unknown tool: %s", params.Name))
	}
}

type storeArgs struct {
	Content   string `json:"content"`
	Namespace string `json:"namespace"`
	Type      string `json:"type"`
}

func (s *Server) handleStore(id any, raw json.RawMessage) {
	var args storeArgs
	if err := json.Unmarshal(raw, &args); err != nil || args.Content == "" {
		s.writeError(id, -32602, "Invalid or missing content"); return
	}
	if args.Namespace == "" {
		args.Namespace = "default"
	}
	memType := types.MemoryType(args.Type)
	if memType == "" {
		memType = types.TypeNote
	}
	memID := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	mem := &types.Memory{ID: memID, Content: args.Content, Type: memType, Namespace: args.Namespace, CreatedAt: time.Now(), UpdatedAt: time.Now()}
	if err := s.store.InsertMemory(mem); err != nil {
		s.writeError(id, -32603, fmt.Sprintf("store failed: %v", err)); return
	}
	s.writeResult(id, map[string]any{"id": memID, "content": args.Content, "type": string(memType), "status": "stored"})
}

type recallArgs struct {
	Query     string `json:"query"`
	Namespace string `json:"namespace"`
	Limit     int    `json:"limit"`
}

func (s *Server) handleRecall(id any, raw json.RawMessage) {
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil || args.Query == "" {
		s.writeError(id, -32602, "Invalid or missing query"); return
	}
	if args.Limit <= 0 {
		args.Limit = 10
	}
	ftsIDs, err := s.store.FTS5Search(args.Query, args.Limit*3, args.Namespace)
	if err != nil {
		s.writeError(id, -32603, fmt.Sprintf("search failed: %v", err)); return
	}
	memories, err := s.store.GetMemoriesByIDs(ftsIDs)
	if err != nil {
		s.writeError(id, -32603, fmt.Sprintf("get memories failed: %v", err)); return
	}
	type resultItem struct {
		ID, Content, Type, Namespace, CreatedAt string
	}
	results := make([]resultItem, 0, len(memories))
	for _, m := range memories {
		results = append(results, resultItem{ID: m.ID, Content: m.Content, Type: string(m.Type), Namespace: m.Namespace, CreatedAt: m.CreatedAt.Format(time.RFC3339)})
	}
	s.writeResult(id, map[string]any{"results": results, "count": len(results)})
}

func (s *Server) handleStats(id any) {
	stats, err := s.store.Stats()
	if err != nil {
		s.writeError(id, -32603, fmt.Sprintf("stats failed: %v", err)); return
	}
	s.writeResult(id, stats)
}

type forgetArgs struct{ ID string }

func (s *Server) handleForget(id any, raw json.RawMessage) {
	var args forgetArgs
	if err := json.Unmarshal(raw, &args); err != nil || args.ID == "" {
		s.writeError(id, -32602, "id required"); return
	}
	if err := s.store.DeleteMemory(args.ID); err != nil {
		s.writeError(id, -32603, fmt.Sprintf("delete failed: %v", err)); return
	}
	s.writeResult(id, map[string]string{"status": "deleted", "id": args.ID})
}

func (s *Server) writeResult(id any, result any) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
}

func (s *Server) writeError(id any, code int, message string) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: message}})
}

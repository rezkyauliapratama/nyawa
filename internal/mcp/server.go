// Package mcp implements a Model Context Protocol server for Nyawa.
// Runs over stdio using JSON-RPC 2.0.
package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

// Server is the MCP tool server for Nyawa.
type Server struct {
	store    *store.Store
	pipeline *search.Pipeline
	reader   *bufio.Scanner
	writer   *json.Encoder
}

// NewServer creates an MCP server backed by the given store and pipeline.
func NewServer(st *store.Store, p *search.Pipeline) *Server {
	return &Server{
		store:    st,
		pipeline: p,
		reader:   bufio.NewScanner(os.Stdin),
		writer:   json.NewEncoder(os.Stdout),
	}
}

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type jsonRPCResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id"`
	Result  any       `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
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

// MCP Tool Result Content types (MCP SDK v1.26.0+)
type toolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolResult struct {
	Content []toolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

func (s *Server) tools() []toolDefinition {
	return []toolDefinition{
		{
			Name:        "nyawa_store",
			Description: "Store a new memory with content, optional namespace (default: 'default'), and optional type (note, insight, decision, fact, etc). Returns the memory ID.",
			InputSchema: inputSchema{
				Type: "object",
				Properties: map[string]propertySchema{
					"content":   {Type: "string", Description: "Memory content to store"},
					"namespace": {Type: "string", Description: "Namespace (default: 'default')"},
					"type":      {Type: "string", Description: "Memory type: decision, insight, procedure, fact, preference, context, note, event, reference", Enum: []string{"decision", "insight", "procedure", "fact", "preference", "context", "note", "event", "reference"}},
				},
				Required: []string{"content"},
			},
		},
		{
			Name:        "nyawa_recall",
			Description: "Semantic search across memories. Uses hybrid search (vector + FTS5 + RRF). Returns ranked results with relevance scores.",
			InputSchema: inputSchema{
				Type: "object",
				Properties: map[string]propertySchema{
					"query":     {Type: "string", Description: "Natural language search query"},
					"namespace": {Type: "string", Description: "Namespace filter (optional)"},
					"limit":     {Type: "number", Description: "Max results (default: 10)"},
				},
				Required: []string{"query"},
			},
		},
		{
			Name:        "nyawa_stats",
			Description: "Get memory statistics: total memories per namespace, pinned count, FTS5 index size, entity graph size.",
			InputSchema: inputSchema{
				Type:       "object",
				Properties: map[string]propertySchema{},
			},
		},
		{
			Name:        "nyawa_forget",
			Description: "Soft-delete a memory by its ID. The memory is marked as superseded and excluded from search results.",
			InputSchema: inputSchema{
				Type: "object",
				Properties: map[string]propertySchema{
					"id": {Type: "string", Description: "Memory ID to delete (e.g. mem_1234567890)"},
				},
				Required: []string{"id"},
			},
		},
	}
}

func (s *Server) Run() error {
	log.Println("Nyawa MCP server started (stdio)")
	log.SetOutput(os.Stderr)
	for s.reader.Scan() {
		line := s.reader.Text()
		if line == "" { continue }
		var req jsonRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			s.writeError(nil, -32700, "Parse error: invalid JSON")
			continue
		}
		s.handleRequest(req)
	}
	return s.reader.Err()
}

func (s *Server) handleRequest(req jsonRPCRequest) {
	// Skip JSON-RPC notifications (no id) — never respond
	if req.ID == nil {
		return
	}
	switch req.Method {
	case "initialize":
		s.handleInitialize(req)
	case "tools/list":
		s.handleToolList(req)
	case "tools/call":
		s.handleToolCall(req)
	default:
		s.writeError(req.ID, -32601, fmt.Sprintf("Method not found: %s", req.Method))
	}
}

func (s *Server) handleInitialize(req jsonRPCRequest) {
	s.writeResult(req.ID, map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities":    map[string]any{"tools": map[string]bool{"listChanged": false}},
		"serverInfo":      map[string]string{"name": "nyawa", "version": "0.9.0"},
	})
}

func (s *Server) handleToolList(req jsonRPCRequest) {
	s.writeResult(req.ID, map[string]any{"tools": s.tools()})
}

type callParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func (s *Server) handleToolCall(req jsonRPCRequest) {
	var params callParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.writeToolError(req.ID, "Invalid params")
		return
	}
	switch params.Name {
	case "nyawa_store":  s.handleStore(req.ID, params.Arguments)
	case "nyawa_recall": s.handleRecall(req.ID, params.Arguments)
	case "nyawa_stats":  s.handleStats(req.ID)
	case "nyawa_forget": s.handleForget(req.ID, params.Arguments)
	default: s.writeToolError(req.ID, fmt.Sprintf("Unknown tool: %s", params.Name))
	}
}

type storeArgs struct {
	Content   string `json:"content"`
	Namespace string `json:"namespace"`
	Type      string `json:"type"`
}

func (s *Server) handleStore(id any, raw json.RawMessage) {
	var args storeArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		s.writeToolError(id, "Invalid arguments")
		return
	}
	if args.Content == "" {
		s.writeToolError(id, "content required")
		return
	}
	if args.Namespace == "" { args.Namespace = "default" }
	memType := types.MemoryType(args.Type)
	if memType == "" { memType = types.TypeNote }
	memID := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	if err := s.store.InsertMemory(&types.Memory{
		ID: memID, Content: args.Content, Type: memType,
		Namespace: args.Namespace, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}); err != nil {
		s.writeToolError(id, fmt.Sprintf("store failed: %v", err))
		return
	}
	result := map[string]any{"id": memID, "content": args.Content, "type": string(memType), "status": "stored"}
	s.writeToolResult(id, result)
}

type recallArgs struct {
	Query     string  `json:"query"`
	Namespace string  `json:"namespace"`
	Limit     float64 `json:"limit"`
}

func (s *Server) handleRecall(id any, raw json.RawMessage) {
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		s.writeToolError(id, "Invalid arguments")
		return
	}
	if args.Query == "" {
		s.writeToolError(id, "query required")
		return
	}
	limit := int(args.Limit)
	if limit <= 0 { limit = 10 }
	results, err := s.pipeline.Search(types.StoreQuery{QueryText: args.Query, Namespace: args.Namespace, Limit: limit})
	if err != nil {
		s.writeToolError(id, fmt.Sprintf("search failed: %v", err))
		return
	}
	defer s.pipeline.ReleaseResults(results)
	type resultItem struct {
		ID, Content, Type, Namespace, CreatedAt string
		Score                                  float64
	}
	items := make([]resultItem, 0, len(results))
	for _, r := range results {
		items = append(items, resultItem{
			ID: r.ID, Content: r.Content, Type: string(r.Type),
			Namespace: r.Namespace, Score: r.Score,
			CreatedAt: r.CreatedAt.Format(time.RFC3339),
		})
	}
	result := map[string]any{"results": items, "count": len(items)}
	s.writeToolResult(id, result)
}

func (s *Server) handleStats(id any) {
	stats, err := s.store.Stats()
	if err != nil {
		s.writeToolError(id, fmt.Sprintf("stats failed: %v", err))
		return
	}
	s.writeToolResult(id, stats)
}

type forgetArgs struct{ ID string `json:"id"` }

func (s *Server) handleForget(id any, raw json.RawMessage) {
	var args forgetArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		s.writeToolError(id, "Invalid arguments")
		return
	}
	if args.ID == "" {
		s.writeToolError(id, "id required")
		return
	}
	if err := s.store.DeleteMemory(args.ID); err != nil {
		s.writeToolError(id, fmt.Sprintf("delete failed: %v", err))
		return
	}
	result := map[string]string{"status": "deleted", "id": args.ID}
	s.writeToolResult(id, result)
}

// writeResult writes a generic JSON-RPC response (used for initialize, tools/list).
func (s *Server) writeResult(id any, result any) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
}

// writeToolResult wraps tool call result in standard MCP CallToolResult format
// { content: [{ type: "text", text: "..." }] }
// Required by MCP SDK v1.26.0+ — the raw result dict is not valid.
func (s *Server) writeToolResult(id any, data any) {
	jsonBytes, err := json.Marshal(data)
	var text string
	if err != nil {
		text = fmt.Sprintf("{\"error\":\"marshal failed: %v\"}", err)
	} else {
		text = string(jsonBytes)
	}
	s.writer.Encode(jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result: toolResult{
			Content: []toolContent{{Type: "text", Text: text}},
		},
	})
}

// writeToolError writes a tool call error as a proper CallToolResult with isError=true.
func (s *Server) writeToolError(id any, message string) {
	s.writer.Encode(jsonRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result: toolResult{
			Content: []toolContent{{Type: "text", Text: fmt.Sprintf("{\"error\":\"%s\"}", message)}},
			IsError: true,
		},
	})
}

func (s *Server) writeError(id any, code int, message string) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: message}})
}

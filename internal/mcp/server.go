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

	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Server struct {
	store    *store.Store
	pipeline *search.Pipeline
	ragStore *rag.RAGStore
	reader   *bufio.Scanner
	writer   *json.Encoder
}

func NewServer(st *store.Store, p *search.Pipeline, rs *rag.RAGStore) *Server {
	return &Server{store: st, pipeline: p, ragStore: rs, reader: bufio.NewScanner(os.Stdin), writer: json.NewEncoder(os.Stdout)}
}

type callToolResult struct {
	Content []contentItem `json:"content"`
}
type contentItem struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type jsonRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}
type jsonRPCResponse struct {
	JSONRPC string   `json:"jsonrpc"`
	ID      any      `json:"id"`
	Result  any      `json:"result,omitempty"`
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

func (s *Server) tools() []toolDefinition {
	return []toolDefinition{
		{Name: "nyawa_store", Description: "Store a new memory.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"content": {Type: "string"}, "namespace": {Type: "string"},
				"type": {Type: "string", Enum: []string{"decision","insight","procedure","fact","preference","context","note","event","reference"}},
			}, Required: []string{"content"}}},
		{Name: "nyawa_recall", Description: "Semantic search across memories.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"query": {Type: "string"}, "namespace": {Type: "string"}, "limit": {Type: "number"},
			}, Required: []string{"query"}}},
		{Name: "nyawa_stats", Description: "Memory statistics.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{}}},
		{Name: "nyawa_forget", Description: "Soft-delete a memory by ID.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"id": {Type: "string"},
			}, Required: []string{"id"}}},
		{Name: "rag_create_collection", Description: "Create a new RAG collection.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"name": {Type: "string"}, "description": {Type: "string"}, "chunk_size": {Type: "number"},
			}, Required: []string{"name"}}},
		{Name: "rag_list_collections", Description: "List all RAG collections.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{}}},
		{Name: "rag_delete_collection", Description: "Delete a RAG collection.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"name": {Type: "string"},
			}, Required: []string{"name"}}},
		{Name: "rag_ingest_file", Description: "Ingest a file into a RAG collection.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"file_path": {Type: "string"}, "collection": {Type: "string"},
			}, Required: []string{"file_path"}}},
		{Name: "rag_query", Description: "Query RAG collections for relevant chunks.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"query": {Type: "string"}, "top_k": {Type: "number"}, "collection": {Type: "string"},
			}, Required: []string{"query"}}},
		{Name: "rag_stats", Description: "Get RAG statistics.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{}}},
	}
}

func (s *Server) Run() error {
	log.Println("Nyawa MCP server started (stdio)")
	for s.reader.Scan() {
		line := s.reader.Text()
		if line == "" { continue }
		var req jsonRPCRequest
		if err := json.Unmarshal([]byte(line), &req); err != nil {
			s.writeError(nil, -32700, "Parse error: invalid JSON"); continue
		}
		s.handleRequest(req)
	}
	return s.reader.Err()
}

func (s *Server) handleRequest(req jsonRPCRequest) {
	if req.ID == nil { return }
	switch req.Method {
	case "initialize": s.handleInitialize(req)
	case "tools/list": s.handleToolList(req)
	case "tools/call": s.handleToolCall(req)
	default: s.writeError(req.ID, -32601, fmt.Sprintf("Method not found: %s", req.Method))
	}
}

func (s *Server) handleInitialize(req jsonRPCRequest) {
	s.writeResult(req.ID, map[string]any{
		"protocolVersion": "2025-03-26",
		"capabilities": map[string]any{"tools": map[string]bool{"listChanged": false}},
		"serverInfo": map[string]string{"name": "nyawa", "version": "0.9.0"},
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
		s.writeError(req.ID, -32602, "Invalid params"); return
	}
	switch params.Name {
	case "nyawa_store":  s.handleStore(req.ID, params.Arguments)
	case "nyawa_recall": s.handleRecall(req.ID, params.Arguments)
	case "nyawa_stats":  s.handleStats(req.ID)
	case "nyawa_forget": s.handleForget(req.ID, params.Arguments)
	case "rag_create_collection":  s.handleRAGCreateCollection(req.ID, params.Arguments)
	case "rag_list_collections":   s.handleRAGListCollections(req.ID)
	case "rag_delete_collection":  s.handleRAGDeleteCollection(req.ID, params.Arguments)
	case "rag_ingest_file":        s.handleRAGIngestFile(req.ID, params.Arguments)
	case "rag_query":              s.handleRAGQuery(req.ID, params.Arguments)
	case "rag_stats":              s.handleRAGStats(req.ID)
	default: s.writeError(req.ID, -32601, fmt.Sprintf("Unknown tool: %s", params.Name))
	}
}

type storeArgs struct{ Content, Namespace, Type string }

func (s *Server) handleStore(id any, raw json.RawMessage) {
	var args storeArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Content == "" { s.writeError(id, -32602, "content required"); return }
	if args.Namespace == "" { args.Namespace = "default" }
	memType := types.MemoryType(args.Type)
	if memType == "" { memType = types.TypeNote }
	memID := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	mem := &types.Memory{ID: memID, Content: args.Content, Type: memType, Namespace: args.Namespace, CreatedAt: time.Now(), UpdatedAt: time.Now()}
	if err := s.store.InsertMemory(mem); err != nil {
		s.writeError(id, -32603, fmt.Sprintf("store failed: %v", err)); return
	}
	s.writeToolResult(id, map[string]any{"id": memID, "content": args.Content, "type": string(memType), "status": "stored"})
}

type recallArgs struct{ Query, Namespace string; Limit float64 }

func (s *Server) handleRecall(id any, raw json.RawMessage) {
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Query == "" { s.writeError(id, -32602, "query required"); return }
	limit := int(args.Limit)
	if limit <= 0 { limit = 10 }
	q := types.StoreQuery{QueryText: args.Query, Namespace: args.Namespace, Limit: limit}
	results, err := s.pipeline.Search(q)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("search failed: %v", err)); return }
	defer s.pipeline.ReleaseResults(results)
	type resultItem struct {
		ID, Content, Type, Namespace, CreatedAt string
		Score float64
	}
	items := make([]resultItem, 0, len(results))
	for _, r := range results {
		items = append(items, resultItem{ID: r.ID, Content: r.Content, Type: string(r.Type),
			Namespace: r.Namespace, Score: r.Score, CreatedAt: r.CreatedAt.Format(time.RFC3339)})
	}
	s.writeToolResult(id, map[string]any{"results": items, "count": len(items)})
}

func (s *Server) handleStats(id any) {
	stats, err := s.store.Stats()
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("stats failed: %v", err)); return }
	s.writeToolResult(id, stats)
}

type forgetArgs struct{ ID string }

func (s *Server) handleForget(id any, raw json.RawMessage) {
	var args forgetArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.ID == "" { s.writeError(id, -32602, "id required"); return }
	if err := s.store.DeleteMemory(args.ID); err != nil {
		s.writeError(id, -32603, fmt.Sprintf("delete failed: %v", err)); return
	}
	s.writeToolResult(id, map[string]string{"status": "deleted", "id": args.ID})
}

// ─── RAG tool implementations ──────────────────────

type ragCreateCollectionArgs struct{ Name, Description string; ChunkSize float64 }

func (s *Server) handleRAGCreateCollection(id any, raw json.RawMessage) {
	var args ragCreateCollectionArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Name == "" { s.writeError(id, -32602, "name required"); return }
	chunkSize := int(args.ChunkSize)
	if chunkSize <= 0 { chunkSize = 500 }
	col, err := s.ragStore.CreateCollection(args.Name, args.Description, chunkSize)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("create collection failed: %v", err)); return }
	type colResult struct{ ID int; Name, Description string; ChunkSize int; Status string }
	s.writeToolResult(id, colResult{ID: col.ID, Name: col.Name, Description: col.Description, ChunkSize: col.ChunkSize, Status: "created"})
}

func (s *Server) handleRAGListCollections(id any) {
	cols, err := s.ragStore.ListCollections()
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("list collections failed: %v", err)); return }
	s.writeToolResult(id, map[string]any{"collections": cols, "count": len(cols)})
}

type ragDeleteCollectionArgs struct{ Name string }

func (s *Server) handleRAGDeleteCollection(id any, raw json.RawMessage) {
	var args ragDeleteCollectionArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Name == "" { s.writeError(id, -32602, "name required"); return }
	if err := s.ragStore.DeleteCollection(args.Name); err != nil {
		s.writeError(id, -32603, fmt.Sprintf("delete collection failed: %v", err)); return
	}
	type delResult struct{ Name, Status string }
	s.writeToolResult(id, delResult{Name: args.Name, Status: "deleted"})
}

type ragIngestFileArgs struct{ FilePath, Collection string }

func (s *Server) handleRAGIngestFile(id any, raw json.RawMessage) {
	var args ragIngestFileArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.FilePath == "" { s.writeError(id, -32602, "file_path required"); return }
	if args.Collection == "" { args.Collection = "default" }
	doc, err := s.ragStore.IngestFile(args.FilePath, args.Collection, nil)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("ingest failed: %v", err)); return }
	type ingestResult struct{ ID, Filename, Collection, SourceType string; ChunkCount int; Status string }
	s.writeToolResult(id, ingestResult{ID: doc.ID, Filename: doc.Filename, Collection: args.Collection,
		ChunkCount: doc.ChunkCount, SourceType: doc.SourceType, Status: "ingested"})
}

type ragQueryArgs struct{ Query, Collection string; TopK float64 }

func (s *Server) handleRAGQuery(id any, raw json.RawMessage) {
	var args ragQueryArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Query == "" { s.writeError(id, -32602, "query required"); return }
	topK := int(args.TopK)
	if topK <= 0 { topK = 5 }
	results, err := s.ragStore.Query(args.Query, topK, args.Collection)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("query failed: %v", err)); return }
	s.writeToolResult(id, map[string]any{"results": results, "count": len(results)})
}

func (s *Server) handleRAGStats(id any) {
	stats := s.ragStore.Stats()
	s.writeToolResult(id, stats)
}

func (s *Server) writeResult(id any, result any) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Result: result})
}

func (s *Server) writeToolResult(id any, result any) {
	text, _ := json.Marshal(result)
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id,
		Result: callToolResult{Content: []contentItem{{Type: "text", Text: string(text)}}},
	})
}

func (s *Server) writeError(id any, code int, message string) {
	s.writer.Encode(jsonRPCResponse{JSONRPC: "2.0", ID: id, Error: &rpcError{Code: code, Message: message}})
}

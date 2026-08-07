// Package mcp implements a Model Context Protocol server for Nyawa.
// Runs over stdio using JSON-RPC 2.0.
package mcp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/compact"
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
		{Name: "nyawa_graph_query", Description: "Traverse the entity graph from query-matched seeds.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"query": {Type: "string", Description: "Text containing entity names to seed the traversal."},
				"depth": {Type: "number", Description: "Traversal depth (default 2)."},
				"limit": {Type: "number", Description: "Max results (default 10)."},
			}, Required: []string{"query"}}},
		{Name: "nyawa_graph_entities", Description: "List/filter entity nodes in the graph.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"name":     {Type: "string", Description: "Optional name substring filter."},
				"category": {Type: "string", Description: "Optional category filter (person, tech, place, org, etc.)."},
				"limit":    {Type: "number", Description: "Max results (default 50)."},
			}}},
		{Name: "nyawa_graph_path", Description: "Find shortest path between two entity nodes.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"source":    {Type: "string", Description: "Source entity name."},
				"target":    {Type: "string", Description: "Target entity name."},
				"max_depth": {Type: "number", Description: "Max BFS depth (default 4)."},
			}, Required: []string{"source", "target"}}},
		{Name: "compact_context", Description: "Compress a conversation transcript (OpenAI-format messages) into a compact summary block. Stores the summary in Nyawa (namespace=context) and returns the compacted message list + stats. LLM summarization is used when NYAWA_LLM_API_KEY is set, otherwise deterministic fallback.",
			InputSchema: inputSchema{Type: "object", Properties: map[string]propertySchema{
				"messages":          {Type: "string", Description: "JSON array of OpenAI-format messages: [{\"role\": \"user\", \"content\": \"...\"}]"},
				"focus_topic":       {Type: "string", Description: "Optional focus topic to prioritize in summarization."},
				"preserve_recent_n": {Type: "number", Description: "Keep last N messages verbatim (default 30)."},
				"segment_size":      {Type: "number", Description: "Messages per segment (default 40)."},
				"segment_overlap":   {Type: "number", Description: "Overlap between segments (default 5)."},
				"dry_run":           {Type: "string", Description: "true = return stats without storing anything."},
				"session_key":       {Type: "string", Description: "Session identifier for compaction_log."},
				"old_session_id":    {Type: "string", Description: "Previous Hermes session id (compression boundary)."},
				"new_session_id":    {Type: "string", Description: "New Hermes session id (compression boundary)."},
			}, Required: []string{"messages"}}},
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
	case "nyawa_graph_query":      s.handleGraphQuery(req.ID, params.Arguments)
	case "nyawa_graph_entities":   s.handleGraphEntities(req.ID, params.Arguments)
	case "nyawa_graph_path":       s.handleGraphPath(req.ID, params.Arguments)
	case "compact_context":        s.handleCompactContext(req.ID, params.Arguments)
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

// ─── Graph tool implementations ──────────────────

type graphQueryArgs struct{ Query string; Depth, Limit float64 }

func (s *Server) handleGraphQuery(id any, raw json.RawMessage) {
	var args graphQueryArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Query == "" { s.writeError(id, -32602, "query required"); return }
	depth := int(args.Depth); if depth <= 0 { depth = 2 }
	limit := int(args.Limit); if limit <= 0 { limit = 10 }

	seeds := matchGraphSeeds(s.store, args.Query)
	if len(seeds) == 0 {
		s.writeToolResult(id, map[string]any{"seeds": []string{}, "results": []graph.TraversalResult{}, "count": 0})
		return
	}

	results, err := s.store.TraverseGraph(seeds, depth, limit)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("graph query failed: %v", err)); return }
	if results == nil { results = []graph.TraversalResult{} }
	s.writeToolResult(id, map[string]any{"query": args.Query, "seeds": seeds, "results": results, "count": len(results)})
}

type graphEntitiesArgs struct{ Name, Category string; Limit float64 }

func (s *Server) handleGraphEntities(id any, raw json.RawMessage) {
	var args graphEntitiesArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	limit := int(args.Limit); if limit <= 0 { limit = 50 }

	entities, err := s.store.ListEntities(args.Name, args.Category, limit)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("list entities failed: %v", err)); return }
	if entities == nil { entities = []graph.Entity{} }
	s.writeToolResult(id, map[string]any{"entities": entities, "count": len(entities)})
}

type graphPathArgs struct{ Source, Target string; MaxDepth float64 }

func (s *Server) handleGraphPath(id any, raw json.RawMessage) {
	var args graphPathArgs
	if err := json.Unmarshal(raw, &args); err != nil { s.writeError(id, -32602, "Invalid arguments"); return }
	if args.Source == "" || args.Target == "" { s.writeError(id, -32602, "source and target required"); return }
	maxDepth := int(args.MaxDepth); if maxDepth <= 0 { maxDepth = 4 }

	path, err := s.store.FindGraphPath(args.Source, args.Target, maxDepth)
	if err != nil { s.writeError(id, -32603, fmt.Sprintf("find path failed: %v", err)); return }
	if path == nil { path = []graph.PathHop{} }
	s.writeToolResult(id, map[string]any{"path": path, "length": len(path)})
}

// ─── Context compaction tool ─────────────────────

type compactArgs struct {
	Messages        string `json:"messages"`
	FocusTopic      string `json:"focus_topic"`
	PreserveRecentN float64 `json:"preserve_recent_n"`
	SegmentSize     float64 `json:"segment_size"`
	SegmentOverlap  float64 `json:"segment_overlap"`
	DryRun          string `json:"dry_run"`
	SessionKey      string `json:"session_key"`
	OldSessionID    string `json:"old_session_id"`
	NewSessionID    string `json:"new_session_id"`
}

func (s *Server) handleCompactContext(id any, raw json.RawMessage) {
	var args compactArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		s.writeError(id, -32602, "Invalid arguments"); return
	}
	if args.Messages == "" {
		s.writeError(id, -32602, "messages required (JSON array string)"); return
	}

	var rawMsgs []map[string]interface{}
	if err := json.Unmarshal([]byte(args.Messages), &rawMsgs); err != nil {
		s.writeError(id, -32602, "messages must be a JSON array: "+err.Error()); return
	}

	preserveRecentN := int(args.PreserveRecentN)
	if preserveRecentN <= 0 { preserveRecentN = 30 }
	segmentSize := int(args.SegmentSize)
	if segmentSize <= 0 { segmentSize = 40 }
	segmentOverlap := int(args.SegmentOverlap)
	if segmentOverlap < 0 { segmentOverlap = 5 }

	msgs := compact.FromOpenAI(rawMsgs)
	inputTokens := compact.TokensIn(msgs)

	prunedMsgs, pruned := compact.PruneToolResults(msgs, 8192)
	segments := compact.Segment(prunedMsgs, segmentSize, segmentOverlap, preserveRecentN)

	// No-op guard: too-short transcript -> return unchanged.
	if len(segments) == 0 {
		s.writeToolResult(id, map[string]any{
			"summary_block": compact.ToOpenAIList(msgs),
			"memory_ids":    []string{},
			"stats": compact.CompactionStats{
				InputMessages:  len(msgs),
				OutputMessages: len(msgs),
				InputTokens:    inputTokens,
				OutputTokens:   inputTokens,
				ReductionPct:   0,
			},
			"noop": true,
		})
		return
	}

	var tail []compact.Message
	if len(prunedMsgs) > preserveRecentN {
		tail = prunedMsgs[len(prunedMsgs)-preserveRecentN:]
	}

	summaries := compact.SummarizeSegments(segments, args.FocusTopic)
	summaryText := strings.Join(summaries, "\n\n---\n\n")

	summaryBlock := compact.BuildSummaryBlock(summaryText, args.SessionKey)
	outputMsgs := append([]compact.Message{}, summaryBlock...)
	if len(tail) > 0 {
		outputMsgs = append(outputMsgs, tail...)
	}
	outputTokens := compact.TokensIn(outputMsgs)

	reductionPct := 0.0
	if inputTokens > 0 {
		reductionPct = float64(inputTokens-outputTokens) / float64(inputTokens) * 100
		if reductionPct < 0 { reductionPct = 0 }
	}

	stats := compact.CompactionStats{
		InputMessages:     len(msgs),
		OutputMessages:    len(outputMsgs),
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		ReductionPct:      reductionPct,
		Segments:          len(segments),
		PrunedToolResults: pruned,
	}

	memoryIDs := []string{}
	if !strings.EqualFold(args.DryRun, "true") && s.store != nil {
		summaryID := fmt.Sprintf("summary_%d", time.Now().UnixNano())
		summaryMem := &types.Memory{
			ID:        summaryID,
			Content:   summaryText,
			Type:      types.TypeContext,
			Namespace: "context",
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		if err := s.store.InsertMemory(summaryMem); err == nil {
			memoryIDs = append(memoryIDs, summaryID)
		}
		sessionKey := args.SessionKey
		if sessionKey == "" {
			sessionKey = args.OldSessionID
		}
		if sessionKey != "" {
			_ = s.store.LogCompaction(sessionKey, args.OldSessionID, args.NewSessionID, summaryID)
		}
	}

	s.writeToolResult(id, map[string]any{
		"summary_block": compact.ToOpenAIList(outputMsgs),
		"memory_ids":    memoryIDs,
		"stats":         stats,
	})
}

// matchGraphSeeds resolves entity names from query text via substring match.
// Matches the graphSeeds pattern from internal/search/pipeline.go.
func matchGraphSeeds(st *store.Store, query string) []string {
	names, err := st.ListEntityNames(10000)
	if err != nil || len(names) == 0 {
		return nil
	}
	queryLower := strings.ToLower(query)
	var seeds []string
	for _, name := range names {
		if len(name) < 3 {
			continue
		}
		if strings.Contains(queryLower, strings.ToLower(name)) {
			seeds = append(seeds, name)
			if len(seeds) >= 3 {
				break
			}
		}
	}
	return seeds
}

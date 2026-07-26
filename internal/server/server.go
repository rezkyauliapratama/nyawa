package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/security"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Server struct {
	store     *store.Store
	pipeline  *search.Pipeline
	embedder  *embedder.PriorityChain
	security  *security.Filter
	ragStore  *rag.RAGStore
	config    Config
	mux       *http.ServeMux
	srv       *http.Server
}

type Config struct {
	Host        string        `yaml:"host"`
	Port        int           `yaml:"port"`
	ReadTimeout time.Duration `yaml:"read_timeout"`
}

func DefaultServerConfig() Config {
	return Config{Host: "0.0.0.0", Port: 3300, ReadTimeout: 30 * time.Second}
}

func New(st *store.Store, pipe *search.Pipeline, emb *embedder.PriorityChain, rs *rag.RAGStore, cfg Config) *Server {
	s := &Server{store: st, pipeline: pipe, embedder: emb, security: security.NewFilter(), ragStore: rs, config: cfg, mux: http.NewServeMux()}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/memories", s.handleMemories)
	s.mux.HandleFunc("/v1/memories/", s.handleMemoryByID)
	s.mux.HandleFunc("/v1/recall", s.handleRecall)
	s.mux.HandleFunc("/v1/stats", s.handleStats)
	s.mux.HandleFunc("/v1/health", s.handleHealth)
	s.mux.HandleFunc("/v1/namespaces", s.handleNamespaces)
	s.mux.HandleFunc("/v1/forget/", s.handleForget)
	s.mux.HandleFunc("/v1/memories/batch", s.handleBatchStore)
	s.mux.HandleFunc("/v1/rag/collections", s.handleRAGCollections)
	s.mux.HandleFunc("/v1/rag/collections/", s.handleRAGCollectionsByName)
	s.mux.HandleFunc("/v1/rag/ingest", s.handleRAGIngest)
	s.mux.HandleFunc("/v1/rag/query", s.handleRAGQuery)
	s.mux.HandleFunc("/v1/rag/stats", s.handleRAGStats)
	s.mux.HandleFunc("/dashboard", s.handleDashboard)
	s.mux.HandleFunc("/", s.handleRoot)
}

func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	s.srv = &http.Server{Addr: addr, Handler: s.withMiddleware(s.mux), ReadTimeout: s.config.ReadTimeout, WriteTimeout: s.config.ReadTimeout}
	log.Printf("Nyawa API server listening on %s", addr)
	go func() {
		time.Sleep(500 * time.Millisecond)
		if _, err := s.pipeline.Search(types.StoreQuery{QueryText: "warmup", Limit: 1}); err != nil {
			log.Printf("warmup search: %v", err)
		}
	}()
	return s.srv.ListenAndServe()
}

func (s *Server) Shutdown() error { if s.srv != nil { return s.srv.Close() }; return nil }

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer func() {
			if rec := recover(); rec != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal server error"})
			}
		}()
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
		if !strings.HasPrefix(r.URL.Path, "/dashboard") { w.Header().Set("Content-Type", "application/json") }
		next.ServeHTTP(w, r)
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" { writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"}); return }
	writeJSON(w, http.StatusOK, map[string]string{"service": "nyawa", "version": "0.1.0", "status": "running"})
}

func (s *Server) handleMemories(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost: s.handleStore(w, r)
	case http.MethodGet: s.handleList(w, r)
	default: writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func (s *Server) handleStore(w http.ResponseWriter, r *http.Request) {
	var req struct{ Content, Namespace, Type string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return }
	req.Content = strings.TrimSpace(req.Content)
	if req.Content == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content required"}); return }
	if cls, reason := s.security.Classify(req.Content); cls == security.Secret {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "content blocked", "reason": reason}); return
	}
	ns := req.Namespace; if ns == "" { ns = "default" }
	memType := types.MemoryType(req.Type); if memType == "" { memType = types.TypeNote }
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	if err := s.store.InsertMemory(&types.Memory{ID: id, Content: req.Content, Type: memType, Namespace: ns, CreatedAt: time.Now(), UpdatedAt: time.Now()}); err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "store failed"}); return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"id": id})
}

func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	ns := r.URL.Query().Get("ns")
	page := parseInt(r.URL.Query().Get("page"), 1)
	perPage := parseInt(r.URL.Query().Get("per_page"), 20)
	offset := (page - 1) * perPage
	var total int
	s.store.GetDB().QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL`).Scan(&total)
	query := `SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE superseded_at IS NULL`
	args := []any{}
	if ns != "" { query += ` AND namespace=?`; args = append(args, ns) }
	query += ` ORDER BY created_at DESC LIMIT ? OFFSET ?`
	args = append(args, perPage, offset)
	rows, err := s.store.GetDB().Query(query, args...)
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "query failed"}); return }
	defer rows.Close()
	type item struct{ ID, Content, Type, Namespace, CreatedAt string; Importance float64; Pinned bool; EdgeCount int }
	var items []item
	for rows.Next() {
		var m item; var mt, cs, us string; var pi, ei, ac int; var ss *string
		rows.Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &ac, &pi, &cs, &us, &ss, &ei)
		m.Type = mt; m.Pinned = pi != 0; m.EdgeCount = ei; m.CreatedAt = cs
		items = append(items, m)
	}
	writeJSON(w, http.StatusOK, map[string]any{"memories": items, "total": total, "page": page, "per_page": perPage})
}

func (s *Server) handleMemoryByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/memories/")
	if id == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"}); return }
	switch r.Method {
	case http.MethodGet:
		mem, err := s.store.GetMemory(id)
		if err != nil { writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"}); return }
		writeJSON(w, http.StatusOK, mem)
	case http.MethodDelete:
		if err := s.store.DeleteMemory(id); err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "delete failed"}); return }
		writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
	default: writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return }
	var req struct{ Query, Namespace string; Limit int }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return }
	if req.Query == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query required"}); return }
	if req.Limit <= 0 { req.Limit = 10 }
	results, err := s.pipeline.Search(types.StoreQuery{QueryText: req.Query, Namespace: req.Namespace, Limit: req.Limit})
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "search failed"}); return }
	defer s.pipeline.ReleaseResults(results)
	type item struct{ ID, Content, Type string; Score, RRFScore, TemporalBoost, ImportanceBoost float64; Rank int; Pinned bool; CreatedAt string }
	items := make([]item, len(results))
	for i, r := range results {
		items[i] = item{ID: r.ID, Content: r.Content, Type: string(r.Type), Score: r.Score, RRFScore: r.RRFScore, TemporalBoost: r.TemporalBoost, ImportanceBoost: r.ImportanceBoost, Rank: r.Rank, Pinned: r.Pinned, CreatedAt: r.CreatedAt.Format(time.RFC3339)}
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": items, "count": len(items)})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return }
	storeStats, err := s.store.Stats()
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "stats failed"}); return }
	writeJSON(w, http.StatusOK, map[string]any{"store": storeStats, "version": "0.1.0"})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	status := http.StatusOK; statusText := "healthy"
	if !s.store.Ready() { status = http.StatusServiceUnavailable; statusText = "degraded" }
	writeJSON(w, status, map[string]any{"status": statusText, "version": "0.1.0"})
}

func (s *Server) handleNamespaces(w http.ResponseWriter, r *http.Request) {
	ns, err := s.store.ListNamespaces()
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	writeJSON(w, http.StatusOK, ns)
}

func (s *Server) handleForget(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/forget/")
	if id == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"}); return }
	if err := s.store.DeleteMemory(id); err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	writeJSON(w, http.StatusOK, map[string]string{"status": "forgotten"})
}

func (s *Server) handleBatchStore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { writeJSON(w, 405, map[string]string{"error": "use POST"}); return }
	var req struct{ Memories []struct{ Content, Namespace, Type string } `json:"memories"` }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, 400, map[string]string{"error": "invalid JSON"}); return }
	if len(req.Memories) == 0 { writeJSON(w, 400, map[string]string{"error": "memories required"}); return }
	results := make([]map[string]any, 0, len(req.Memories))
	for i, m := range req.Memories {
		if m.Content == "" { continue }
		ns := m.Namespace; if ns == "" { ns = "default" }
		mt := types.MemoryType(m.Type); if mt == "" { mt = types.TypeNote }
		id := fmt.Sprintf("mem_%d_%d", time.Now().UnixNano(), i)
		if err := s.store.InsertMemory(&types.Memory{ID: id, Content: m.Content, Type: mt, Namespace: ns}); err != nil {
			results = append(results, map[string]any{"id": id, "status": "error", "error": err.Error()}); continue
		}
		results = append(results, map[string]any{"id": id, "status": "stored"})
	}
	writeJSON(w, 201, map[string]any{"results": results, "count": len(results)})
}

func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	s.writeDashboardHTML(w)
}

// ─── RAG Handlers ──────────────────────────────────────────

func (s *Server) handleRAGCollections(w http.ResponseWriter, r *http.Request) {
	if s.ragStore == nil { writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "RAG not available"}); return }
	switch r.Method {
	case http.MethodGet:
		cols, err := s.ragStore.ListCollections()
		if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
		if cols == nil { cols = []rag.Collection{} }
		writeJSON(w, http.StatusOK, map[string]any{"collections": cols})
	case http.MethodPost:
		var req struct{ Name, Description string; ChunkSize int }
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return }
		if req.Name == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "name required"}); return }
		col, err := s.ragStore.CreateCollection(req.Name, req.Description, req.ChunkSize)
		if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
		writeJSON(w, http.StatusCreated, col)
	default: writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func (s *Server) handleRAGCollectionsByName(w http.ResponseWriter, r *http.Request) {
	if s.ragStore == nil { writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "RAG not available"}); return }
	name := strings.TrimPrefix(r.URL.Path, "/v1/rag/collections/")
	if name == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "name required"}); return }
	if r.Method != http.MethodDelete { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use DELETE"}); return }
	if err := s.ragStore.DeleteCollection(name); err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func (s *Server) handleRAGIngest(w http.ResponseWriter, r *http.Request) {
	if s.ragStore == nil { writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "RAG not available"}); return }
	if r.Method != http.MethodPost { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"}); return }
	var req struct{ FilePath, Collection string }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return }
	if req.FilePath == "" || req.Collection == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "file_path and collection required"}); return }
	doc, err := s.ragStore.IngestFile(req.FilePath, req.Collection, nil)
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	writeJSON(w, http.StatusCreated, doc)
}

func (s *Server) handleRAGQuery(w http.ResponseWriter, r *http.Request) {
	if s.ragStore == nil { writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "RAG not available"}); return }
	if r.Method != http.MethodPost { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"}); return }
	var req struct{ Query, Collection string; TopK int }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return }
	if req.Query == "" { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query required"}); return }
	if req.TopK <= 0 { req.TopK = 5 }
	results, err := s.ragStore.Query(req.Query, req.TopK, req.Collection)
	if err != nil { writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()}); return }
	if results == nil { results = []rag.RAGResult{} }
	writeJSON(w, http.StatusOK, map[string]any{"results": results, "count": len(results)})
}

func (s *Server) handleRAGStats(w http.ResponseWriter, r *http.Request) {
	if s.ragStore == nil { writeJSON(w, http.StatusServiceUnavailable, map[string]string{"error": "RAG not available"}); return }
	if r.Method != http.MethodGet { writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use GET"}); return }
	stats := s.ragStore.Stats()
	writeJSON(w, http.StatusOK, stats)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func parseInt(s string, defaultVal int) int {
	if s == "" { return defaultVal }
	var v int
	if _, err := fmt.Sscanf(s, "%d", &v); err != nil { return defaultVal }
	if v < 1 { return defaultVal }
	return v
}

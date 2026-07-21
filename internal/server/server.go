// Nyawa server
package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/security"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Server struct {
	store    *store.Store
	pipeline *search.Pipeline
	embedder *embedder.PriorityChain
	security *security.Filter
	config   Config
	mux      *http.ServeMux
	srv      *http.Server
}
type Config struct{ Host string; Port int; ReadTimeout time.Duration }
func DefaultServerConfig() Config { return Config{Host: "0.0.0.0", Port: 3300, ReadTimeout: 30 * time.Second} }
func New(st *store.Store, pipe *search.Pipeline, emb *embedder.PriorityChain, cfg Config) *Server {
	s := &Server{store: st, pipeline: pipe, embedder: emb, security: security.NewFilter(), config: cfg, mux: http.NewServeMux()}
	s.registerRoutes(); return s
}
func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/memories", s.handleMemories)
	s.mux.HandleFunc("/v1/memories/batch", s.handleBatchStore)
	s.mux.HandleFunc("/v1/memories/", s.handleMemoryByID)
	s.mux.HandleFunc("/v1/recall", s.handleRecall)
	s.mux.HandleFunc("/v1/stats", s.handleStats)
	s.mux.HandleFunc("/v1/health", s.handleHealth)
	s.mux.HandleFunc("/v1/namespaces", s.handleNamespaces)
	s.mux.HandleFunc("/v1/forget/", s.handleForget)
	s.mux.HandleFunc("/dashboard", s.handleDashboard)
	s.mux.HandleFunc("/", s.handleRoot)
}
func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	s.srv = &http.Server{Addr: addr, Handler: s.withMiddleware(s.mux), ReadTimeout: s.config.ReadTimeout, WriteTimeout: s.config.ReadTimeout}
	log.Printf("Nyawa API server on %s", addr); return s.srv.ListenAndServe()
}
func (s *Server) Shutdown() error { if s.srv != nil { return s.srv.Close() }; return nil }
func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer func() {
			if rec := recover(); rec != nil { log.Printf("PANIC: %s %s: %v", r.Method, r.URL.Path, rec); writeJSON(w, 500, map[string]string{"error": "internal"}) }
		}()
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
		if !strings.HasPrefix(r.URL.Path, "/dashboard") { w.Header().Set("Content-Type", "application/json") }
		next.ServeHTTP(w, r)
	})
}
func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" { writeJSON(w, 404, map[string]string{"error": "not found"}); return }
	writeJSON(w, 200, map[string]string{"service": "nyawa", "status": "running"})
}
type storeRequest struct{ Content, Namespace, Type string }
func (s *Server) handleMemories(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost: s.handleStore(w, r)
	case http.MethodGet: s.handleList(w, r)
	default: writeJSON(w, 405, map[string]string{"error": "not allowed"})
	}
}
func (s *Server) handleStore(w http.ResponseWriter, r *http.Request) {
	var req storeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, 400, map[string]string{"error": "invalid JSON"}); return }
	if req.Content == "" { writeJSON(w, 400, map[string]string{"error": "content required"}); return }
	ns := req.Namespace; if ns == "" { ns = "default" }
	mt := types.MemoryType(req.Type); if mt == "" { mt = types.TypeNote }
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	if err := s.store.InsertMemory(&types.Memory{ID: id, Content: req.Content, Type: mt, Namespace: ns}); err != nil {
		writeJSON(w, 500, map[string]string{"error": "store failed"}); return
	}
	writeJSON(w, 201, map[string]any{"id": id})
}
func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	ns := r.URL.Query().Get("ns")
	page := parseInt(r.URL.Query().Get("page"), 1)
	pp := parseInt(r.URL.Query().Get("per_page"), 20)
	var total int
	s.store.GetDB().QueryRow(`SELECT COUNT(*) FROM memories WHERE superseded_at IS NULL`).Scan(&total)
	q := `SELECT id,content,mem_type,namespace,importance,access_count,pinned,created_at,updated_at,superseded_at,edge_count FROM memories WHERE superseded_at IS NULL`
	args := []any{}
	if ns != "" { q += ` AND namespace=?`; args = append(args, ns) }
	q += ` ORDER BY created_at DESC LIMIT ? OFFSET ?`; args = append(args, pp, (page-1)*pp)
	rows, err := s.store.GetDB().Query(q, args...)
	if err != nil { writeJSON(w, 500, map[string]string{"error": "query failed"}); return }
	defer rows.Close()
	type mi struct{ ID, Content, Type, Namespace, CreatedAt string; Importance float64; Pinned bool; EdgeCount int }
	var items []mi
	for rows.Next() {
		var m mi; var mt, cs, us string; var pi, ei, ac int; var ss *string
		rows.Scan(&m.ID, &m.Content, &mt, &m.Namespace, &m.Importance, &ac, &pi, &cs, &us, &ss, &ei)
		m.Type = mt; m.Pinned = pi != 0; m.EdgeCount = ei; m.CreatedAt = cs
		items = append(items, m)
	}
	writeJSON(w, 200, map[string]any{"memories": items, "total": total, "page": page, "per_page": pp})
}
func (s *Server) handleMemoryByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/memories/")
	if id == "" || id == "batch" { return } // batch handled separately
	if r.Method == http.MethodGet {
		m, err := s.store.GetMemory(id); if err != nil { writeJSON(w, 404, map[string]string{"error": "not found"}); return }
		writeJSON(w, 200, m)
	} else if r.Method == http.MethodDelete {
		if err := s.store.DeleteMemory(id); err != nil { writeJSON(w, 500, map[string]string{"error": "delete failed"}); return }
		writeJSON(w, 200, map[string]string{"status": "deleted"})
	} else { writeJSON(w, 405, map[string]string{"error": "not allowed"}) }
}
type recallRequest struct{ Query, Namespace string; Limit int }
func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { writeJSON(w, 405, map[string]string{"error": "not allowed"}); return }
	var req recallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, 400, map[string]string{"error": "invalid JSON"}); return }
	if req.Query == "" { writeJSON(w, 400, map[string]string{"error": "query required"}); return }
	if req.Limit <= 0 { req.Limit = 10 }
	results, err := s.pipeline.Search(types.StoreQuery{QueryText: req.Query, Namespace: req.Namespace, Limit: req.Limit})
	if err != nil { writeJSON(w, 500, map[string]string{"error": "search failed"}); return }
	defer s.pipeline.ReleaseResults(results)
	type ri struct{ ID, Content, Type, CreatedAt string; Score, RRFScore, TemporalBoost, ImportanceBoost float64; Rank int; Pinned bool }
	items := make([]ri, 0, len(results))
	for _, r := range results {
		items = append(items, ri{ID: r.ID, Content: r.Content, Type: string(r.Type), Score: r.Score, RRFScore: r.RRFScore, TemporalBoost: r.TemporalBoost, ImportanceBoost: r.ImportanceBoost, Rank: r.Rank, Pinned: r.Pinned, CreatedAt: r.CreatedAt.Format(time.RFC3339)})
	}
	writeJSON(w, 200, map[string]any{"results": items, "count": len(items)})
}
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	storeStats, _ := s.store.Stats()
	eName, eOK := s.embedder.HealthCheck()
	writeJSON(w, 200, map[string]any{"store": storeStats, "embedder": map[string]string{"active": eName, "status": map[bool]string{true: "available", false: "unavailable"}[eOK]}})
}
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	stOK := s.store.Ready(); eName, eOK := s.embedder.HealthCheck()
	st := "healthy"; sc := 200
	if !stOK || !eOK { st = "degraded"; sc = 503 }
	writeJSON(w, sc, map[string]any{"status": st, "store": stOK, "embedder": map[string]string{"active": eName, "status": map[bool]string{true: "available", false: "unavailable"}[eOK]}})
}
func (s *Server) handleNamespaces(w http.ResponseWriter, r *http.Request) {
	ns, err := s.store.ListNamespaces()
	if err != nil { writeJSON(w, 500, map[string]string{"error": err.Error()}); return }
	writeJSON(w, 200, ns)
}
func (s *Server) handleForget(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete { writeJSON(w, 405, map[string]string{"error": "use DELETE"}); return }
	id := strings.TrimPrefix(r.URL.Path, "/v1/forget/")
	if id == "" { writeJSON(w, 400, map[string]string{"error": "id required"}); return }
	if err := s.store.DeleteMemory(id); err != nil { writeJSON(w, 500, map[string]string{"error": err.Error()}); return }
	writeJSON(w, 200, map[string]string{"status": "forgotten"})
}
func (s *Server) handleBatchStore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost { writeJSON(w, 405, map[string]string{"error": "use POST"}); return }
	var req struct{ Memories []struct{ Content, Namespace, Type string } }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil { writeJSON(w, 400, map[string]string{"error": "invalid JSON"}); return }
	if len(req.Memories) == 0 { writeJSON(w, 400, map[string]string{"error": "memories required"}); return }
	results := make([]map[string]any, 0, len(req.Memories))
	for i, m := range req.Memories {
		if m.Content == "" { continue }
		ns := m.Namespace; if ns == "" { ns = "default" }
		mt := types.MemoryType(m.Type); if mt == "" { mt = types.TypeNote }
		id := fmt.Sprintf("mem_%d_%d", time.Now().UnixNano(), i)
		if err := s.store.InsertMemory(&types.Memory{ID: id, Content: m.Content, Type: mt, Namespace: ns}); err != nil {
			results = append(results, map[string]any{"id": id, "status": "error"})
			continue
		}
		results = append(results, map[string]any{"id": id, "status": "stored"})
	}
	writeJSON(w, 201, map[string]any{"results": results, "count": len(results)})
}
func (s *Server) handleDashboard(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(200); s.writeDashboardHTML(w)
}
func writeJSON(w http.ResponseWriter, s int, v any) { w.Header().Set("Content-Type", "application/json"); w.WriteHeader(s); json.NewEncoder(w).Encode(v) }
func parseInt(s string, def int) int {
	if s == "" { return def }
	var v int; if _, err := fmt.Sscanf(s, "%d", &v); err != nil { return def }
	if v < 1 { return def }
	return v
}

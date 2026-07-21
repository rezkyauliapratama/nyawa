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

type Config struct {
	Host        string        `yaml:"host"`
	Port        int           `yaml:"port"`
	ReadTimeout time.Duration `yaml:"read_timeout"`
}

func DefaultServerConfig() Config {
	return Config{Host: "0.0.0.0", Port: 3300, ReadTimeout: 30 * time.Second}
}

func New(st *store.Store, pipe *search.Pipeline, emb *embedder.PriorityChain, cfg Config) *Server {
	s := &Server{store: st, pipeline: pipe, embedder: emb, security: security.NewFilter(), config: cfg, mux: http.NewServeMux()}
	s.registerRoutes()
	return s
}

func (s *Server) registerRoutes() {
	s.mux.HandleFunc("/v1/memories", s.handleMemories)
	s.mux.HandleFunc("/v1/memories/", s.handleMemoryByID)
	s.mux.HandleFunc("/v1/recall", s.handleRecall)
	s.mux.HandleFunc("/v1/stats", s.handleStats)
	s.mux.HandleFunc("/v1/health", s.handleHealth)
	s.mux.HandleFunc("/", s.handleRoot)
}

func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	s.srv = &http.Server{Addr: addr, Handler: s.withMiddleware(s.mux), ReadTimeout: s.config.ReadTimeout, WriteTimeout: s.config.ReadTimeout}
	log.Printf("Nyawa API server listening on %s", addr)
	return s.srv.ListenAndServe()
}

func (s *Server) Shutdown() error {
	if s.srv != nil {
		return s.srv.Close()
	}
	return nil
}

func (s *Server) withMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("PANIC: %s %s: %v", r.Method, r.URL.Path, rec)
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "internal server error"})
			}
		}()
		w.Header().Set("Content-Type", "application/json")
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"}); return
	}
	writeJSON(w, http.StatusOK, map[string]string{"service": "nyawa", "version": "0.2.0", "status": "running"})
}

type storeRequest struct {
	Content   string `json:"content"`
	Namespace string `json:"namespace,omitempty"`
	Type      string `json:"type,omitempty"`
}

func (s *Server) handleMemories(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handleStore(w, r)
	case http.MethodGet:
		s.handleList(w, r)
	default:
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

func (s *Server) handleStore(w http.ResponseWriter, r *http.Request) {
	var req storeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return
	}
	if req.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content required"}); return
	}
	classification, reason := s.security.Classify(req.Content)
	if classification == security.Secret {
		writeJSON(w, http.StatusForbidden, map[string]string{"error": "content blocked", "reason": reason})
		return
	}
	ns := req.Namespace
	if ns == "" {
		ns = "default"
	}
	memType := types.MemoryType(req.Type)
	if memType == "" {
		memType = types.TypeNote
	}
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	mem := &types.Memory{ID: id, Content: req.Content, Type: memType, Namespace: ns, CreatedAt: time.Now(), UpdatedAt: time.Now()}
	if err := s.store.InsertMemory(mem); err != nil {
		log.Printf("store error: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "store failed"}); return
	}
	writeJSON(w, http.StatusCreated, map[string]any{"id": id, "classification": classification})
}

func (s *Server) handleList(w http.ResponseWriter, r *http.Request) {
	stats, err := s.store.Stats()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "stats failed"}); return
	}
	writeJSON(w, http.StatusOK, stats)
}

func (s *Server) handleMemoryByID(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/v1/memories/")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "id required"}); return
	}
	switch r.Method {
	case http.MethodGet:
		mem, err := s.store.GetMemory(id)
		if err != nil {
			writeJSON(w, http.StatusNotFound, map[string]string{"error": "not found"}); return
		}
		writeJSON(w, http.StatusOK, mem)
	case http.MethodDelete:
		if err := s.store.DeleteMemory(id); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "delete failed"}); return
		}
		writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
	default:
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
	}
}

type recallRequest struct {
	Query     string `json:"query"`
	Namespace string `json:"namespace,omitempty"`
	Limit     int    `json:"limit,omitempty"`
}

func (s *Server) handleRecall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return
	}
	var req recallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON"}); return
	}
	if req.Query == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query required"}); return
	}
	if req.Limit <= 0 {
		req.Limit = 10
	}
	results, err := s.pipeline.Search(types.StoreQuery{QueryText: req.Query, Namespace: req.Namespace, Limit: req.Limit})
	if err != nil {
		log.Printf("recall error: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "search failed"}); return
	}
	defer s.pipeline.ReleaseResults(results)
	type resultItem struct {
		ID             string  `json:"id"`
		Content        string  `json:"content"`
		Type           string  `json:"type"`
		Score          float64 `json:"score"`
		RRFScore       float64 `json:"rrf_score"`
		TemporalBoost  float64 `json:"temporal_boost"`
		ImportanceBoost float64 `json:"importance_boost"`
		Rank           int     `json:"rank"`
		Pinned         bool    `json:"pinned"`
		CreatedAt      string  `json:"created_at"`
	}
	items := make([]resultItem, 0, len(results))
	for _, r := range results {
		items = append(items, resultItem{ID: r.ID, Content: r.Content, Type: string(r.Type), Score: r.Score, RRFScore: r.RRFScore, TemporalBoost: r.TemporalBoost, ImportanceBoost: r.ImportanceBoost, Rank: r.Rank, Pinned: r.Pinned, CreatedAt: r.CreatedAt.Format(time.RFC3339)})
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": items, "count": len(items)})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"}); return
	}
	storeStats, err := s.store.Stats()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "stats failed"}); return
	}
	embName, embOK := s.embedder.HealthCheck()
	embStatus := "unavailable"
	if embOK {
		embStatus = "available"
	}
	writeJSON(w, http.StatusOK, map[string]any{"store": storeStats, "embedder": map[string]string{"active": embName, "status": embStatus}, "version": "0.2.0"})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	storeOK := s.store.Ready()
	embName, embOK := s.embedder.HealthCheck()
	status := http.StatusOK
	statusText := "healthy"
	if !storeOK || !embOK {
		status = http.StatusServiceUnavailable
		statusText = "degraded"
	}
	writeJSON(w, status, map[string]any{"status": statusText, "store": storeOK, "embedder": map[string]string{"active": embName, "status": map[bool]string{true: "available", false: "unavailable"}[embOK]}, "version": "0.2.0"})
}

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

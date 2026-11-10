package main

import (
	"database/sql"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/compactioneval"
	"github.com/rezkyauliapratama/nyawa/internal/dream"
	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/mcp"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/server"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
	"github.com/rezkyauliapratama/nyawa/internal/version"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("nyawa: ")
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}
	switch os.Args[1] {
	case "store":
		cmdStore()
	case "recall":
		cmdRecall()
	case "search":
		cmdSearch()
	case "stats":
		cmdStats()
	case "init":
		cmdInit()
	case "serve":
		cmdServe()
	case "mcp":
		cmdMCP()
	case "dream":
		cmdDream()
	case "ns":
		cmdNamespace()
	case "archive":
		cmdArchive()
	case "import":
		cmdImport()
	case "compaction-eval":
		if err := cmdCompactionEval(); err != nil {
			log.Fatalf("compaction-eval: %v", err)
		}
	case "graph":
		cmdGraph()
	case "reindex":
		cmdReindex()
	case "rag-reindex":
		cmdRAGReindex()
	case "version":
		fmt.Println(version.String())
	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`Nyawa — Offline-First AI Memory Engine v%s

Usage:
  nyawa init <db-path>
  nyawa store <db> <content>
  nyawa recall <db> <query> [--ns <ns>] [--at <time>]
  nyawa stats <db>
  nyawa ns <db>                           List namespaces
  nyawa archive <db> <output-path>        Archive superseded memories
  nyawa compaction-eval score <trace.json|->  Score an immutable trace offline
  nyawa compaction-eval assemble <events.jsonl|->  Assemble passive events offline
  nyawa compaction-eval recall-probe <suite.json> <snapshot.db> --snapshot  Run read-only probes
  nyawa compaction-eval baseline <observations.json> <window-start> <window-end>  Report baseline
  nyawa compaction-eval residual-pack <input.json>  Build a shadow residual pack (never persisted)
  nyawa serve <db>                        Start HTTP + Dream Cycle
  nyawa mcp <db>                          Start MCP server
  nyawa dream <db>                        Run Dream Cycle
  nyawa graph <db> <query> [--depth 2] [--limit 10]  Traverse entity graph
  nyawa reindex <db>                      Re-embed memories missing from HNSW
  nyawa rag-reindex [--apply] [--batch-size N] [--max-input-chars N] <db>  Backfill dedicated RAG HNSW
  nyawa version                           Show version
`, version.Number())
}

func parseFlags() (ns string, atTime time.Time) {
	for i := 2; i < len(os.Args); i++ {
		if os.Args[i] == "--ns" && i+1 < len(os.Args) {
			ns = os.Args[i+1]
		}
		if os.Args[i] == "--at" && i+1 < len(os.Args) {
			if t, err := time.Parse(time.RFC3339, os.Args[i+1]); err == nil {
				atTime = t
			} else {
				if t, err := time.Parse("2006-01-02", os.Args[i+1]); err == nil {
					atTime = t
				}
			}
		}
	}
	return
}

func getStore(p string, emb store.Embedder) *store.Store {
	s, err := store.NewStore(p, emb)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	return s
}

const (
	memoryEmbedderLocal = "local"
	memoryEmbedderAPI   = "api"
	ragEmbedderLocal    = "local"
	ragEmbedderAPI      = "api"

	ragTaskModeAuto   = "auto"
	ragTaskModeNone   = "none"
	ragTaskModeJinaV3 = "jina-v3"

	ragDefaultAPIBaseURL = "https://api.jina.ai/v1"
	ragDefaultAPIModel   = "jina-embeddings-v3"
	ragDefaultAPIDims    = 1024
)

type runtimeEmbedders struct {
	memory     *embedder.PriorityChain
	rag        *embedder.PriorityChain
	ragProfile rag.EmbeddingProfile
	ragErr     error
	bge        *embedder.PythonEmbedder
}

func (e *runtimeEmbedders) Stop() {
	if e != nil && e.bge != nil {
		e.bge.Stop()
	}
}

// startBGE starts one local BGE subprocess. The runtime pair builder shares
// this instance when both memory and RAG use the same local model.
func startBGE() (*embedder.PythonEmbedder, string) {
	dir, source := embedder.DefaultModelDir()
	bge := embedder.NewPythonEmbedder(dir)
	if err := bge.Start(); err != nil {
		log.Printf("BGE unavailable (model=%s, source=%s): %v", dir, source, err)
	} else {
		log.Printf("BGE embedder ready (model=%s, source=%s, dim=%d)", dir, source, bge.Dims())
	}
	return bge, dir
}

func newMemoryEmbedder(mode string, bge *embedder.PythonEmbedder, getenv func(string) string) (*embedder.PriorityChain, error) {
	if mode == memoryEmbedderAPI {
		api, err := embedder.NewOpenAIEmbedderFromEnv(getenv)
		if err != nil {
			return nil, fmt.Errorf("MEMORY_EMBEDDER=api: %w", err)
		}
		log.Printf("memory embedder: api (%s, dim=%d)", api.Name(), api.Dims())
		return embedder.NewPriorityChain(api), nil
	}
	ollama := embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"})
	return embedder.NewPriorityChain(bge, ollama), nil
}

func memoryEmbedderMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", memoryEmbedderLocal, "bge":
		return memoryEmbedderLocal, nil
	case memoryEmbedderAPI:
		return memoryEmbedderAPI, nil
	default:
		return "", fmt.Errorf("MEMORY_EMBEDDER must be %q or %q, got %q", memoryEmbedderLocal, memoryEmbedderAPI, raw)
	}
}

func ragEmbedderMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", ragEmbedderLocal, "bge":
		return ragEmbedderLocal, nil
	case ragEmbedderAPI:
		return ragEmbedderAPI, nil
	default:
		return "", fmt.Errorf("RAG_EMBEDDER must be %q or %q, got %q", ragEmbedderLocal, ragEmbedderAPI, raw)
	}
}

func ragTaskMode(raw string) (string, error) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", ragTaskModeAuto:
		return ragTaskModeAuto, nil
	case ragTaskModeNone:
		return ragTaskModeNone, nil
	case ragTaskModeJinaV3:
		return ragTaskModeJinaV3, nil
	default:
		return "", fmt.Errorf("RAG_EMBEDDING_TASK_MODE must be %q, %q, or %q, got %q", ragTaskModeAuto, ragTaskModeNone, ragTaskModeJinaV3, raw)
	}
}

func ragRetrievalTasksEnabled(mode, baseURL, model string) (bool, error) {
	mode, err := ragTaskMode(mode)
	if err != nil {
		return false, err
	}
	switch mode {
	case ragTaskModeNone:
		return false, nil
	case ragTaskModeJinaV3:
		return true, nil
	default:
		parsed, err := url.Parse(baseURL)
		if err != nil {
			return false, fmt.Errorf("parse embedding base URL for task capability: %w", err)
		}
		return strings.EqualFold(parsed.Hostname(), "api.jina.ai") && model == "jina-embeddings-v3", nil
	}
}

func newRAGAPIEmbedder(getenv func(string) string) (*embedder.PriorityChain, rag.EmbeddingProfile, error) {
	return newRAGAPIEmbedderWithHTTPClient(getenv, nil)
}

func newRAGAPIEmbedderWithHTTPClient(getenv func(string) string, client *http.Client) (*embedder.PriorityChain, rag.EmbeddingProfile, error) {
	api, err := embedder.NewOpenAIEmbedderFromEnvWithHTTPClient(getenv, client)
	if err != nil {
		return nil, rag.EmbeddingProfile{}, err
	}
	baseURL := strings.TrimSpace(getenv("EMBEDDING_BASE_URL"))
	if baseURL == "" {
		baseURL = ragDefaultAPIBaseURL
	}
	model := strings.TrimSpace(getenv("EMBEDDING_MODEL"))
	if model == "" {
		model = ragDefaultAPIModel
	}
	taskMode, err := ragTaskMode(getenv("RAG_EMBEDDING_TASK_MODE"))
	if err != nil {
		return nil, rag.EmbeddingProfile{}, err
	}
	tasksEnabled, err := ragRetrievalTasksEnabled(taskMode, baseURL, model)
	if err != nil {
		return nil, rag.EmbeddingProfile{}, err
	}
	api.EnableRetrievalTasks(tasksEnabled)
	profile := rag.EmbeddingProfile{Provider: "openai-compatible", Model: model, BaseURL: baseURL, Dimensions: api.Dims()}
	if tasksEnabled {
		profile.QueryTask = embedder.TaskRetrievalQuery
		profile.PassageTask = embedder.TaskRetrievalPassage
	}
	return embedder.NewPriorityChain(api), profile, nil
}

// getEmbedder remains the memory-only selector used by memory CLI commands.
func getEmbedder() (*embedder.PriorityChain, error) {
	mode, err := memoryEmbedderMode(os.Getenv("MEMORY_EMBEDDER"))
	if err != nil {
		return nil, err
	}
	var bge *embedder.PythonEmbedder
	if mode == memoryEmbedderLocal {
		bge, _ = startBGE()
	}
	return newMemoryEmbedder(mode, bge, os.Getenv)
}

// getRuntimeEmbedders selects memory and RAG independently. A local BGE process
// is shared when both selectors choose local; the RAG local chain deliberately
// excludes Ollama so it cannot switch vector spaces behind the RAG index.
func getRuntimeEmbedders() (*runtimeEmbedders, error) {
	return getRuntimeEmbeddersWith(os.Getenv, startBGE)
}

func getRuntimeEmbeddersWith(getenv func(string) string, startLocal func() (*embedder.PythonEmbedder, string)) (*runtimeEmbedders, error) {
	memoryMode, err := memoryEmbedderMode(getenv("MEMORY_EMBEDDER"))
	if err != nil {
		return nil, err
	}
	ragMode, ragModeErr := ragEmbedderMode(getenv("RAG_EMBEDDER"))

	var bge *embedder.PythonEmbedder
	var modelDir string
	if memoryMode == memoryEmbedderLocal || (ragModeErr == nil && ragMode == ragEmbedderLocal) {
		bge, modelDir = startLocal()
	}
	memory, err := newMemoryEmbedder(memoryMode, bge, getenv)
	if err != nil {
		if bge != nil {
			bge.Stop()
		}
		return nil, err
	}
	runtime := &runtimeEmbedders{memory: memory, bge: bge}
	if ragModeErr != nil {
		runtime.ragErr = ragModeErr
		return runtime, nil
	}
	if ragMode == ragEmbedderAPI {
		runtime.rag, runtime.ragProfile, err = newRAGAPIEmbedder(getenv)
		if err != nil {
			runtime.ragErr = fmt.Errorf("RAG_EMBEDDER=api: %w", err)
		}
		return runtime, nil
	}
	if bge == nil {
		runtime.ragErr = fmt.Errorf("RAG local embedder was not initialized")
		return runtime, nil
	}
	fingerprint, fingerprintErr := embedder.LocalModelFingerprint(modelDir)
	if fingerprintErr != nil {
		runtime.ragErr = fmt.Errorf("identify RAG local model artifacts: %w", fingerprintErr)
		return runtime, nil
	}
	runtime.rag = embedder.NewPriorityChain(bge)
	runtime.ragProfile = rag.EmbeddingProfile{Provider: "bge-onnx", Model: "minilm-multilingual@" + fingerprint, Dimensions: bge.Dims()}
	return runtime, nil
}

func mustGetEmbedder() *embedder.PriorityChain {
	emb, err := getEmbedder()
	if err != nil {
		log.Fatal(err)
	}
	return emb
}

func setupRAGStore(st *store.Store, dbPath string, runtime *runtimeEmbedders) *rag.RAGStore {
	if runtime.ragErr != nil {
		log.Printf("RAG disabled: %v", runtime.ragErr)
		return rag.NewUnavailableRAGStore(st.GetDB(), runtime.ragErr)
	}
	path := dbPath + ".rag.hnsw"
	hnsw, err := rag.OpenVectorIndex(st.GetDB(), path, runtime.ragProfile)
	if err != nil {
		log.Printf("RAG index disabled: %v", err)
		return rag.NewUnavailableRAGStore(st.GetDB(), err)
	}
	store := rag.NewRAGStore(st.GetDB(), hnsw, path, runtime.rag)
	if err := store.InitError(); err != nil {
		log.Printf("RAG store disabled: %v", err)
		return store
	}
	log.Printf("RAG embedder configured: %s/%s dim=%d ready=%t index=%s", runtime.ragProfile.Provider, runtime.ragProfile.Model, runtime.ragProfile.Dimensions, runtime.rag.Available(), path)
	return store
}

func cmdInit() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa init <db-path>")
	}
	s := getStore(os.Args[2], nil)
	defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.Marshal(stats)
	fmt.Println(string(b))
}

func cmdStore() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa store <db> <content>")
	}
	content := strings.TrimSpace(os.Args[3])
	if content == "" {
		log.Fatal("content cannot be empty")
	}
	emb := mustGetEmbedder()
	defer emb.StopAll()
	s := getStore(os.Args[2], emb)
	defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	if err := s.InsertMemory(&types.Memory{ID: id, Content: content, Type: types.TypeNote, Namespace: "default"}); err != nil {
		log.Fatalf("store: %v", err)
	}
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa recall <db> <query> [--ns <ns>] [--at <time>]")
	}
	ns, atTime := parseFlags()
	emb := mustGetEmbedder()
	defer emb.StopAll()
	s := getStore(os.Args[2], emb)
	defer s.Close()
	p, stopScorer, err := newSearchPipeline(s, emb, types.DefaultConfig().Search)
	if err != nil {
		log.Fatalf("initialize recall confidence scorer: %v", err)
	}
	defer stopScorer()

	q := types.StoreQuery{QueryText: os.Args[3], Limit: 10, Namespace: ns}
	if !atTime.IsZero() {
		q.TimeTravel = &atTime
	}

	results, err := p.Search(q)
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	defer p.ReleaseResults(results)
	for i, r := range results {
		fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content)
	}
}
func cmdSearch() { cmdRecall() }

func cmdStats() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa stats <db>")
	}
	s := getStore(os.Args[2], nil)
	defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.MarshalIndent(stats, "", "  ")
	fmt.Println(string(b))
}

func cmdNamespace() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa ns <db>")
	}
	s := getStore(os.Args[2], nil)
	defer s.Close()
	ns, _ := s.ListNamespaces()
	if len(ns) == 0 {
		fmt.Println("No namespaces.")
		return
	}
	for name, count := range ns {
		fmt.Printf("%s: %d memories\n", name, count)
	}
}

func cmdArchive() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa archive <db> <output-path>")
	}
	s := getStore(os.Args[2], nil)
	defer s.Close()
	count, err := s.ArchiveSuperseded(os.Args[3])
	if err != nil {
		log.Fatalf("archive: %v", err)
	}
	fmt.Printf("Archived %d memories to %s\n", count, os.Args[3])
}

func cmdCompactionEval() error {
	if len(os.Args) < 3 {
		return fmt.Errorf("usage: nyawa compaction-eval {score|assemble|recall-probe|baseline|residual-pack} ...")
	}
	command := os.Args[2]
	readInput := func(name string) ([]byte, error) {
		if name == "-" {
			return io.ReadAll(os.Stdin)
		}
		return os.ReadFile(name)
	}
	switch command {
	case "score":
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: nyawa compaction-eval score <trace.json|->")
		}
		data, err := readInput(os.Args[3])
		if err != nil {
			return fmt.Errorf("read trace: %w", err)
		}
		result, err := compactioneval.ScoreJSON(data)
		if err != nil {
			return fmt.Errorf("invalid trace: %w", err)
		}
		encoded, err := compactioneval.MarshalResultJSON(result)
		if err != nil {
			return fmt.Errorf("encode score result: %w", err)
		}
		_, err = fmt.Fprintln(os.Stdout, string(encoded))
		return err
	case "assemble":
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: nyawa compaction-eval assemble <events.jsonl|->")
		}
		data, err := readInput(os.Args[3])
		if err != nil {
			return fmt.Errorf("read events: %w", err)
		}
		trace, err := compactioneval.AssembleJSONL(data)
		if err != nil {
			return err
		}
		encoded, err := json.Marshal(trace)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(os.Stdout, string(encoded))
		return err
	case "recall-probe":
		if len(os.Args) != 6 || os.Args[5] != "--snapshot" {
			return fmt.Errorf("usage: nyawa compaction-eval recall-probe <suite.json> <snapshot.db> --snapshot")
		}
		data, err := readInput(os.Args[3])
		if err != nil {
			return fmt.Errorf("read probe suite: %w", err)
		}
		var suite compactioneval.RecallProbeSuite
		decoder := json.NewDecoder(strings.NewReader(string(data)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&suite); err != nil {
			return err
		}
		run, err := compactioneval.RunRecallProbeSuite(compactioneval.RecallRunnerConfig{DBPath: os.Args[4], Snapshot: true}, suite)
		if err != nil {
			return err
		}
		encoded, err := compactioneval.MarshalRecallProbeRun(run)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(os.Stdout, string(encoded))
		return err
	case "baseline":
		if len(os.Args) != 6 {
			return fmt.Errorf("usage: nyawa compaction-eval baseline <observations.json> <window-start> <window-end>")
		}
		data, err := readInput(os.Args[3])
		if err != nil {
			return err
		}
		observations, err := compactioneval.DecodeBaselineObservations(data)
		if err != nil {
			return err
		}
		report, err := compactioneval.BuildBaselineReport(observations, os.Args[4], os.Args[5])
		if err != nil {
			return err
		}
		encoded, err := compactioneval.MarshalBaselineJSON(report)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintln(os.Stdout, string(encoded))
		return err
	case "residual-pack":
		if len(os.Args) != 4 {
			return fmt.Errorf("usage: nyawa compaction-eval residual-pack <input.json|->")
		}
		data, err := readInput(os.Args[3])
		if err != nil {
			return err
		}
		var input compactioneval.ResidualPackInput
		decoder := json.NewDecoder(strings.NewReader(string(data)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			return err
		}
		result, err := compactioneval.BuildResidualPack(input)
		if err != nil {
			return err
		}
		if result.CanonicalJSON == nil {
			return errors.New("residual pack disabled or empty")
		}
		_, err = fmt.Fprintln(os.Stdout, string(result.CanonicalJSON))
		return err
	default:
		return fmt.Errorf("unknown compaction-eval command %q", command)
	}
}

func cmdImport() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa import <db> <file.json|->")
	}
	emb := mustGetEmbedder()
	defer emb.StopAll()
	s := getStore(os.Args[2], emb)
	defer s.Close()

	var data []byte
	if os.Args[3] == "-" {
		data = readStdin()
	} else {
		var err error
		data, err = os.ReadFile(os.Args[3])
		if err != nil {
			log.Fatalf("read file: %v", err)
		}
	}

	var entries []struct {
		Content   string `json:"content"`
		Namespace string `json:"namespace,omitempty"`
		Type      string `json:"type,omitempty"`
	}
	if err := json.Unmarshal(data, &entries); err != nil {
		log.Fatalf("parse JSON: %v (expected array of {content,namespace?,type?})", err)
	}

	now := time.Now()
	imported, failed := 0, 0
	for i, entry := range entries {
		if entry.Content == "" {
			continue
		}
		ns := entry.Namespace
		if ns == "" {
			ns = "default"
		}
		mt := types.MemoryType(entry.Type)
		if mt == "" {
			mt = types.TypeNote
		}
		id := fmt.Sprintf("mem_%d_%d", now.UnixNano(), i)
		if err := s.InsertMemory(&types.Memory{ID: id, Content: entry.Content, Type: mt, Namespace: ns}); err != nil {
			failed++
			continue
		}
		imported++
	}
	fmt.Printf("Imported %d memories (%d failed)\n", imported, failed)
}

func readStdin() []byte {
	var buf []byte
	b := make([]byte, 4096)
	for {
		n, err := os.Stdin.Read(b)
		if n > 0 {
			buf = append(buf, b[:n]...)
		}
		if err != nil {
			break
		}
	}
	return buf
}

func newSearchPipeline(st search.StoreReader, emb *embedder.PriorityChain, cfg types.SearchConfig) (*search.Pipeline, func(), error) {
	if !rag.LocalCrossEncoderEnabled() {
		return search.NewPipeline(st, emb, cfg), func() {}, nil
	}
	scorer := rag.NewPythonCrossEncoder()
	if !scorer.Available() {
		scorer.Stop()
		return nil, func() {}, fmt.Errorf("local cross-encoder was enabled but did not become ready")
	}
	return search.NewPipelineWithConfidenceScorer(st, emb, cfg, scorer), scorer.Stop, nil
}

func mcpHealthHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"status":"ready"}`+"\n")
	})
}

func startMCPHTTP(portValue, token string, st *store.Store, pipeline *search.Pipeline, ragStore *rag.RAGStore) (*http.Server, error) {
	if portValue == "" {
		return nil, nil
	}
	port, err := strconv.Atoi(portValue)
	if err != nil || port < 1 || port > 65535 {
		return nil, fmt.Errorf("NYAWA_MCP_HTTP_PORT must be an integer from 1 to 65535")
	}
	handler, err := mcp.NewHTTPHandler(st, pipeline, ragStore, token)
	if err != nil {
		return nil, err
	}
	mux := http.NewServeMux()
	mux.Handle("/mcp", handler)
	mux.Handle("/healthz", mcpHealthHandler())
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, fmt.Errorf("listen for MCP HTTP on port %d: %w", port, err)
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 180 * time.Second, IdleTimeout: 120 * time.Second}
	go func() {
		if err := srv.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("MCP HTTP server: %v", err)
		}
	}()
	log.Printf("MCP HTTP server listening on :%d/mcp", port)
	return srv, nil
}

func cmdServe() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa serve <db>")
	}
	runtime, err := getRuntimeEmbedders()
	if err != nil {
		log.Fatal(err)
	}
	defer runtime.Stop()
	st := getStore(os.Args[2], runtime.memory)
	defer st.Close()

	engine := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	engine.SetGraphStore(st.GetGraph())
	engine.Start(dream.DefaultConfig())

	// Periodic WAL checkpoint (safety net besides the post-Dream checkpoint):
	// keeps memory.db-wal from ballooning between Dream cycles, which once
	// reached 200MB and made every write slow + concurrent stores fail with
	// "database is locked".
	go func() {
		t := time.NewTicker(30 * time.Minute)
		defer t.Stop()
		for range t.C {
			if truncated, err := st.CheckpointWAL(); err != nil {
				log.Printf("WAL checkpoint ticker: %v", err)
			} else if truncated {
				log.Printf("WAL checkpoint ticker: truncated")
			}
		}
	}()

	hc := embedder.NewHealthCheckRunner(runtime.memory, 60*time.Second)
	hc.Start()
	defer hc.Stop()
	p, stopScorer, err := newSearchPipeline(st, runtime.memory, types.DefaultConfig().Search)
	if err != nil {
		log.Fatalf("initialize recall confidence scorer: %v", err)
	}
	defer stopScorer()
	rs := setupRAGStore(st, os.Args[2], runtime)
	srv := server.New(st, p, runtime.memory, rs, server.DefaultServerConfig())
	mcpHTTP, err := startMCPHTTP(os.Getenv("NYAWA_MCP_HTTP_PORT"), os.Getenv("NYAWA_MCP_HTTP_TOKEN"), st, p, rs)
	if err != nil {
		log.Fatalf("MCP HTTP: %v", err)
	}
	if mcpHTTP != nil {
		defer mcpHTTP.Close()
	}
	ragLabel := "disabled"
	if runtime.ragErr == nil {
		ragLabel = fmt.Sprintf("%s/%s/%d", runtime.ragProfile.Provider, runtime.ragProfile.Model, runtime.ragProfile.Dimensions)
	}
	log.Printf("Server — db=%s memory_embedder=%s rag_embedder=%s dream=%v", os.Args[2], runtime.memory.Current(), ragLabel, engine.Running())
	if err := srv.Start(); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func cmdMCP() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa mcp <db>")
	}
	runtime, err := getRuntimeEmbedders()
	if err != nil {
		log.Fatal(err)
	}
	defer runtime.Stop()
	st := getStore(os.Args[2], runtime.memory)
	log.Printf("MCP — db=%s memory_embedder=%s", os.Args[2], runtime.memory.Current())
	p, stopScorer, err := newSearchPipeline(st, runtime.memory, types.DefaultConfig().Search)
	if err != nil {
		log.Fatalf("initialize recall confidence scorer: %v", err)
	}
	defer stopScorer()
	rs := setupRAGStore(st, os.Args[2], runtime)
	if err := mcp.NewServer(st, p, rs).Run(); err != nil {
		log.Fatalf("mcp: %v", err)
	}
}

func cmdDream() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa dream <db-path>")
	}
	st := getStore(os.Args[2], nil)
	defer st.Close()
	stats, _ := st.Stats()
	b, _ := json.MarshalIndent(stats, "", "  ")
	fmt.Println(string(b))
	fmt.Println("--- Running Dream Cycle ---")
	engine := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	engine.SetGraphStore(st.GetGraph())
	cfg := dream.DefaultConfig()
	result := engine.Run(cfg)
	b2, _ := json.MarshalIndent(result, "", "  ")
	fmt.Println(string(b2))
}

func cmdGraph() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa graph <db> <query> [--depth 2] [--limit 10]")
	}
	dbPath := os.Args[2]
	query := os.Args[3]
	depth := 2
	limit := 10
	for i := 4; i < len(os.Args); i++ {
		if os.Args[i] == "--depth" && i+1 < len(os.Args) {
			if d, err := fmt.Sscanf(os.Args[i+1], "%d", &depth); d != 1 || err != nil {
				depth = 2
			}
		}
		if os.Args[i] == "--limit" && i+1 < len(os.Args) {
			if l, err := fmt.Sscanf(os.Args[i+1], "%d", &limit); l != 1 || err != nil {
				limit = 10
			}
		}
	}
	s := getStore(dbPath, nil)
	defer s.Close()

	// Seed entities from query text
	names, err := s.ListEntityNames(10000)
	if err != nil {
		log.Fatalf("list entity names: %v", err)
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

	if len(seeds) == 0 {
		fmt.Println(`{"seeds":[],"results":[],"count":0}`)
		return
	}

	results, err := s.TraverseGraph(seeds, depth, limit)
	if err != nil {
		log.Fatalf("traverse graph: %v", err)
	}
	if results == nil {
		results = []graph.TraversalResult{}
	}
	out, _ := json.MarshalIndent(map[string]any{"query": query, "seeds": seeds, "results": results, "count": len(results)}, "", "  ")
	fmt.Println(string(out))
}

// cmdReindex embeds every active memory that is missing a vector in the HNSW
// index and persists the updated index. Existing vectors are left untouched
// (HNSW.Contains guards against duplicates).
//
// Persisting goes through HNSW.MergeAndSave: under the cross-process lock it
// reloads the on-disk index, merges vectors written by a running gateway while
// the reindex was in flight, and writes the union once, atomically.
func cmdReindex() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa reindex <db>")
	}
	emb := mustGetEmbedder()
	defer emb.StopAll()
	s := getStore(os.Args[2], emb)
	defer s.Close()

	mems, err := s.ListAllMemories()
	if err != nil {
		log.Fatalf("list memories: %v", err)
	}
	total := len(mems)

	hnsw := s.GetHNSW()
	before := hnsw.Size()
	already, reindexed, failed := 0, 0, 0
	for _, m := range mems {
		if hnsw.Contains(m.ID) {
			already++
			continue
		}
		v, e := emb.Embed(m.Content)
		if e != nil || len(v) == 0 {
			if errors.Is(e, embedder.ErrEmbeddingDimensionMismatch) {
				log.Fatalf("reindex refused: %v", e)
			}
			failed++
			continue
		}
		if err := hnsw.Insert(m.ID, v); err != nil {
			log.Fatalf("reindex refused: %v", err)
		}
		reindexed++
	}
	after := hnsw.Size()

	if reindexed > 0 {
		merged, persisted, err := hnsw.MergeAndSave(s.GetHNSWPath())
		if err != nil {
			log.Fatalf("persist hnsw: %v", err)
		}
		coverage := 0.0
		if total > 0 {
			coverage = float64(persisted) / float64(total) * 100
		}
		log.Printf("hnsw merge: persisted %d vectors (%d recovered from disk during reindex, in-memory %d -> %d), coverage %.1f%% of %d active memories",
			persisted, merged, before, after, coverage, total)
	} else {
		log.Printf("hnsw: nothing to reindex, %d vectors in memory (coverage %.1f%% of %d active memories)",
			before, coveragePct(before, total), total)
	}
	fmt.Printf("Reindexed %d memories (%d already indexed, %d failed)\n", reindexed, already, failed)
}

func cmdRAGReindex() {
	fs := flag.NewFlagSet("rag-reindex", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	apply := fs.Bool("apply", false, "send RAG chunk text to the configured embedder and write the dedicated RAG index")
	batchSize := fs.Int("batch-size", 32, "maximum passages per provider request (1-64)")
	maxInputChars := fs.Int("max-input-chars", 300000, "fail closed if total unindexed text exceeds this character count")
	if err := fs.Parse(os.Args[2:]); err != nil {
		log.Fatalf("rag-reindex arguments: %v", err)
	}
	if fs.NArg() != 1 {
		log.Fatal("usage: nyawa rag-reindex [--apply] [--batch-size N] [--max-input-chars N] <db>")
	}
	if *batchSize < 1 || *batchSize > 64 || *maxInputChars < 1 {
		log.Fatal("rag-reindex requires batch-size 1-64 and a positive max-input-chars")
	}
	dbPath := fs.Arg(0)
	if info, err := os.Stat(dbPath); err != nil || !info.Mode().IsRegular() {
		log.Fatalf("rag-reindex database must be an existing regular file")
	}
	readOnlyDB, err := sql.Open("sqlite3", sqliteReadOnlyDSN(dbPath))
	if err != nil {
		log.Fatalf("open RAG backfill plan read-only: %v", err)
	}
	readOnlyDB.SetMaxOpenConns(1)
	pending, inputChars, err := inspectRAGBackfillPlan(readOnlyDB)
	closeErr := readOnlyDB.Close()
	if err != nil {
		log.Fatalf("inspect RAG backfill plan: %v", err)
	}
	if closeErr != nil {
		log.Fatalf("close RAG backfill plan: %v", closeErr)
	}
	model := strings.TrimSpace(os.Getenv("EMBEDDING_MODEL"))
	if model == "" {
		model = ragDefaultAPIModel
	}
	baseURL := strings.TrimSpace(os.Getenv("EMBEDDING_BASE_URL"))
	if baseURL == "" {
		baseURL = ragDefaultAPIBaseURL
	}
	dimensions := strings.TrimSpace(os.Getenv("EMBEDDING_DIMS"))
	if dimensions == "" {
		dimensions = fmt.Sprint(ragDefaultAPIDims)
	}
	mode := strings.TrimSpace(os.Getenv("RAG_EMBEDDING_TASK_MODE"))
	if mode == "" {
		mode = ragTaskModeAuto
	}
	tasks, taskErr := ragRetrievalTasksEnabled(mode, baseURL, model)
	if taskErr != nil {
		log.Fatalf("rag-reindex task profile: %v", taskErr)
	}
	if inputChars > *maxInputChars {
		log.Fatalf("rag-reindex blocked: %d input characters exceed cap %d", inputChars, *maxInputChars)
	}
	plannedRequests := 0
	if pending > 0 {
		plannedRequests = (pending + *batchSize - 1) / *batchSize
	}
	fmt.Printf("RAG backfill plan: chunks=%d input_chars=%d model=%s dimensions=%s task_mode=%s task_fields=%t batch_size=%d max_requests=%d\n",
		pending, inputChars, model, dimensions, mode, tasks, *batchSize, plannedRequests)
	if !*apply {
		fmt.Println("mode=dry-run provider_calls=0 database_writes=0; add --apply only for an approved maintenance window")
		return
	}

	ragMode, err := ragEmbedderMode(os.Getenv("RAG_EMBEDDER"))
	if err != nil || ragMode != ragEmbedderAPI {
		log.Fatal("rag-reindex --apply requires RAG_EMBEDDER=api")
	}
	if model != ragDefaultAPIModel || baseURL != ragDefaultAPIBaseURL || !tasks {
		log.Fatal("rag-reindex --apply requires the approved Jina v3 endpoint and retrieval task profile")
	}
	emb, profile, err := newRAGAPIEmbedder(os.Getenv)
	if err != nil {
		log.Fatalf("initialize RAG API embedder: %v", err)
	}
	defer emb.StopAll()
	if profile.Model != ragDefaultAPIModel || profile.Dimensions != ragDefaultAPIDims || profile.QueryTask != embedder.TaskRetrievalQuery || profile.PassageTask != embedder.TaskRetrievalPassage {
		log.Fatalf("rag-reindex profile mismatch: model=%s dimensions=%d query_task=%q passage_task=%q", profile.Model, profile.Dimensions, profile.QueryTask, profile.PassageTask)
	}
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatalf("open RAG database for backfill: %v", err)
	}
	db.SetMaxOpenConns(1)
	defer db.Close()
	indexPath := dbPath + ".rag.hnsw"
	hnsw, err := rag.OpenVectorIndex(db, indexPath, profile)
	if err != nil {
		log.Fatalf("open dedicated RAG index: %v", err)
	}
	ragStore := rag.NewRAGStore(db, hnsw, indexPath, emb)
	if err := ragStore.InitError(); err != nil {
		log.Fatalf("initialize RAG store: %v", err)
	}
	result, err := ragStore.BackfillUnindexed(*batchSize, *maxInputChars)
	if err != nil {
		log.Fatalf("RAG backfill failed; stop and verify backup before retry: %v", err)
	}
	fmt.Printf("RAG backfill complete: chunks=%d indexed=%d input_chars=%d provider_batches=%d index_vectors=%d\n",
		result.TotalChunks, result.IndexedChunks, result.InputCharacters, result.ProviderBatches, ragStore.Stats()["rag_index_vectors"])
}

func sqliteReadOnlyDSN(dbPath string) string {
	absolute, err := filepath.Abs(dbPath)
	if err != nil {
		return "file:" + dbPath + "?mode=ro"
	}
	return (&url.URL{Scheme: "file", Path: filepath.ToSlash(absolute)}).String() + "?mode=ro"
}

func inspectRAGBackfillPlan(db *sql.DB) (int, int, error) {
	var tableCount int
	if err := db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='rag_chunks'`).Scan(&tableCount); err != nil {
		return 0, 0, err
	}
	if tableCount != 1 {
		return 0, 0, fmt.Errorf("rag_chunks table is absent")
	}
	rows, err := db.Query(`PRAGMA table_info(rag_chunks)`)
	if err != nil {
		return 0, 0, err
	}
	hasVectorFlag := false
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			rows.Close()
			return 0, 0, err
		}
		if name == "vector_indexed" {
			hasVectorFlag = true
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return 0, 0, err
	}
	if err := rows.Close(); err != nil {
		return 0, 0, err
	}
	query := `SELECT COUNT(*),COALESCE(SUM(length(content)),0) FROM rag_chunks`
	if hasVectorFlag {
		query += ` WHERE vector_indexed=0`
	}
	var count, chars int
	if err := db.QueryRow(query).Scan(&count, &chars); err != nil {
		return 0, 0, err
	}
	return count, chars, nil
}

// coveragePct returns indexed as a percentage of total, or 0 when total is 0.
func coveragePct(indexed, total int) float64 {
	if total == 0 {
		return 0
	}
	return float64(indexed) / float64(total) * 100
}

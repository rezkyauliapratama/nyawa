package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/dream"
	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/mcp"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/server"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

var (
	bgeEmbedder    *embedder.PythonEmbedder
	apiEmbedder    *embedder.OpenAIEmbedder
	ollamaEmbedder *embedder.OllamaEmbedder
)

func main() {
	log.SetFlags(0); log.SetPrefix("nyawa: ")
	if len(os.Args) < 2 { printUsage(); os.Exit(1) }
	switch os.Args[1] {
	case "store": cmdStore()
	case "recall": cmdRecall()
	case "search": cmdSearch()
	case "stats": cmdStats()
	case "init": cmdInit()
	case "serve": cmdServe()
	case "mcp": cmdMCP()
	case "dream": cmdDream()
	case "ns": cmdNamespace()
	case "archive": cmdArchive()
	case "import": cmdImport()
	case "dashboard": cmdDashboard()
	case "version": fmt.Println("nyawa v0.9.0")
	case "rag": cmdRag()
	default: printUsage(); os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`Nyawa -- Offline-First AI Memory Engine

Usage:
  nyawa init <db>
  nyawa store <db> <content>
  nyawa import <db> <file.json|->
  nyawa recall <db> <q> [--ns <ns>] [--at <time>]
  nyawa stats <db>
  nyawa dashboard <db>
  nyawa ns <db>
  nyawa archive <db> <out>
  nyawa serve <db>
  nyawa mcp <db>
  nyawa dream <db>
  nyawa version
`)
}

func parseFlags() (ns string, atTime time.Time) {
	for i := 2; i < len(os.Args); i++ {
		if os.Args[i] == "--ns" && i+1 < len(os.Args) { ns = os.Args[i+1] }
		if os.Args[i] == "--at" && i+1 < len(os.Args) {
			if t, err := time.Parse(time.RFC3339, os.Args[i+1]); err == nil { atTime = t }
		}
	}
	return
}

func getStore(p string, emb store.Embedder) *store.Store {
	s, err := store.NewStore(p, emb)
	if err != nil { log.Fatalf("store: %v", err) }
	return s
}

func initEmbedders() {
	if bgeEmbedder == nil {
		bgeEmbedder = embedder.NewPythonEmbedder("/opt/data/nyawa/internal/embedder/model")
		if err := bgeEmbedder.Start(); err != nil {
			log.Printf("BGE unavailable: %v", err)
			bgeEmbedder = nil
		} else {
			log.Printf("BGE ready")
		}
	}
	if apiEmbedder == nil {
		apiEmbedder = embedder.NewOpenAIEmbedder()
	}
	if ollamaEmbedder == nil {
		ollamaEmbedder = embedder.NewOllamaEmbedder(embedder.OllamaConfig{
			BaseURL: "http://localhost:11434", Model: "nomic-embed-text",
		})
	}
}

func getMemoryEmbedder() *embedder.PriorityChain {
	initEmbedders()
	if bgeEmbedder == nil {
		bge := embedder.NewPythonEmbedder("/opt/data/nyawa/internal/embedder/model")
		if err := bge.Start(); err == nil { bgeEmbedder = bge }
	}
	mode := os.Getenv("MEMORY_EMBEDDER")
	if mode == "api" && apiEmbedder != nil && apiEmbedder.Available() {
		log.Printf("Memory embedder: api (%s)", apiEmbedder.Name())
		return embedder.NewPriorityChain(apiEmbedder)
	}
	if bgeEmbedder != nil {
		return embedder.NewPriorityChain(bgeEmbedder, ollamaEmbedder)
	}
	return embedder.NewPriorityChain(apiEmbedder, ollamaEmbedder)
}

func getRAGEmbedder() *embedder.PriorityChain {
	initEmbedders()
	if bgeEmbedder == nil {
		bge := embedder.NewPythonEmbedder("/opt/data/nyawa/internal/embedder/model")
		if err := bge.Start(); err == nil { bgeEmbedder = bge }
	}
	mode := os.Getenv("RAG_EMBEDDER")
	if mode == "api" && apiEmbedder != nil && apiEmbedder.Available() {
		log.Printf("RAG embedder: api (%s)", apiEmbedder.Name())
		return embedder.NewPriorityChain(apiEmbedder)
	}
	if bgeEmbedder != nil {
		return embedder.NewPriorityChain(bgeEmbedder, apiEmbedder, ollamaEmbedder)
	}
	return embedder.NewPriorityChain(apiEmbedder, ollamaEmbedder)
}

func cmdInit() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa init <db>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats(); b, _ := json.Marshal(stats); fmt.Println(string(b))
}

func cmdStore() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa store <db> <content>") }
	content := strings.TrimSpace(os.Args[3])
	if content == "" { log.Fatal("content cannot be empty") }
	emb := getMemoryEmbedder(); defer emb.StopAll()
	s := getStore(os.Args[2], emb); defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	s.InsertMemory(&types.Memory{ID: id, Content: content, Type: types.TypeNote, Namespace: "default"})
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa recall <db> <q> [--ns <ns>] [--at <time>]") }
	ns, atTime := parseFlags()
	emb := getMemoryEmbedder(); defer emb.StopAll()
	s := getStore(os.Args[2], emb); defer s.Close()
	p := search.NewPipeline(s, emb, types.DefaultConfig().Search)
	q := types.StoreQuery{QueryText: os.Args[3], Limit: 10, Namespace: ns}
	if !atTime.IsZero() { q.TimeTravel = &atTime }
	results, err := p.Search(q)
	if err != nil { log.Fatalf("search: %v", err) }
	defer p.ReleaseResults(results)
	for i, r := range results { fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content) }
}
func cmdSearch() { cmdRecall() }

func cmdStats() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa stats <db>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats(); b, _ := json.MarshalIndent(stats, "", "  "); fmt.Println(string(b))
}

func cmdNamespace() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa ns <db>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	ns, _ := s.ListNamespaces()
	for name, count := range ns { fmt.Printf("%s: %d memories\n", name, count) }
}

func cmdArchive() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa archive <db> <out>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	c, err := s.ArchiveSuperseded(os.Args[3])
	if err != nil { log.Fatalf("archive: %v", err) }
	fmt.Printf("Archived %d memories to %s\n", c, os.Args[3])
}

func cmdImport() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa import <db> <file.json|->") }
	emb := getMemoryEmbedder(); defer emb.StopAll()
	s := getStore(os.Args[2], emb); defer s.Close()
	var data []byte
	if os.Args[3] == "-" {
		b := make([]byte, 4096)
		for { n, err := os.Stdin.Read(b); if n > 0 { data = append(data, b[:n]...) }; if err != nil { break } }
	} else { var err error; data, err = os.ReadFile(os.Args[3]); if err != nil { log.Fatalf("read: %v", err) } }
	var entries []struct {
		Content   string `json:"content"`
		Namespace string `json:"namespace,omitempty"`
		Type      string `json:"type,omitempty"`
	}
	if err := json.Unmarshal(data, &entries); err != nil { log.Fatalf("parse: %v", err) }
	now := time.Now(); im, fa := 0, 0
	for i, e := range entries {
		if e.Content == "" { continue }
		ns := e.Namespace; if ns == "" { ns = "default" }
		mt := types.MemoryType(e.Type); if mt == "" { mt = types.TypeNote }
		if err := s.InsertMemory(&types.Memory{ID: fmt.Sprintf("mem_%d_%d", now.UnixNano(), i), Content: e.Content, Type: mt, Namespace: ns}); err != nil { fa++; continue }
		im++
	}
	fmt.Printf("Imported %d (%d failed)\n", im, fa)
}

func cmdServe() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa serve <db>") }
	emb := getMemoryEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb); defer st.Close()
	engine := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	engine.Start(dream.DefaultConfig())
	hc := embedder.NewHealthCheckRunner(emb, 60*time.Second); hc.Start(); defer hc.Stop()
	p := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	srv := server.New(st, p, emb, server.DefaultServerConfig())
	log.Printf("Server -- db=%s embedder=%s dream=%v", os.Args[2], emb.Current(), engine.Running())
	if err := srv.Start(); err != nil { log.Fatalf("server: %v", err) }
}

func cmdMCP() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa mcp <db>") }
	emb := getMemoryEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb)
	log.Printf("MCP -- db=%s embedder=%s", os.Args[2], emb.Current())
	p := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	if err := mcp.NewServer(st, p).Run(); err != nil { log.Fatalf("mcp: %v", err) }
}

func cmdDream() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa dream <db-path>") }
	st := getStore(os.Args[2], nil); defer st.Close()
	stats, _ := st.Stats(); b, _ := json.MarshalIndent(stats, "", "  "); fmt.Println(string(b))
	fmt.Println("--- Running Dream Cycle ---")
	e := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	res := e.Run(dream.DefaultConfig())
	b2, _ := json.MarshalIndent(res, "", "  "); fmt.Println(string(b2))
}

func cmdRag() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa rag <db> <command> [args...]") }
	dbPath := os.Args[2]
	sub := os.Args[3]

	emb := getRAGEmbedder(); defer emb.StopAll()
	st := getStore(dbPath, emb); defer st.Close()
	rs := rag.NewRAGStore(st.GetDB(), st.GetHNSW(), st.GetHNSWPath(), emb)

	switch sub {
	case "collection":
		if len(os.Args) < 5 { log.Fatal("usage: nyawa rag <db> collection <list|create|delete>") }
		switch os.Args[4] {
		case "list":
			cols, err := rs.ListCollections()
			if err != nil { log.Fatalf("collections: %v", err) }
			if len(cols) == 0 { fmt.Println("No collections"); return }
			for _, c := range cols { fmt.Printf("#%d %s (docs=%d chunk_size=%d)\n", c.ID, c.Name, c.DocCount, c.ChunkSize) }
		case "create":
			name := ""
			if len(os.Args) > 5 { name = os.Args[5] } else { log.Fatal("usage: nyawa rag <db> collection create <name>") }
			c, err := rs.CreateCollection(name, "", 500)
			if err != nil { log.Fatalf("create: %v", err) }
			fmt.Printf("Collection created: #%d %s\n", c.ID, c.Name)
		case "delete":
			if len(os.Args) > 5 {
				if err := rs.DeleteCollection(os.Args[5]); err != nil { log.Fatalf("delete: %v", err) }
				fmt.Println("Collection deleted")
			} else { log.Fatal("usage: nyawa rag <db> collection delete <name>") }
		default:
			log.Fatalf("unknown collection command: %s", os.Args[4])
		}

	case "ingest":
		if len(os.Args) < 5 { log.Fatal("usage: nyawa rag <db> ingest <file|dir> [--collection <name>]") }
		colName := "default"
		path := os.Args[4]
		for i := 5; i < len(os.Args)-1; i++ {
			if os.Args[i] == "--collection" { colName = os.Args[i+1] }
		}
		info, err := os.Stat(path)
		if err != nil { log.Fatalf("path: %v", err) }
		if info.IsDir() {
			n, err := rs.IngestDir(path, colName)
			if err != nil { log.Fatalf("ingest dir: %v", err) }
			fmt.Printf("Ingested %d files from %s into %s\n", n, path, colName)
		} else {
			doc, err := rs.IngestFile(path, colName, nil)
			if err != nil { log.Fatalf("ingest file: %v", err) }
			fmt.Printf("Ingested %s (%d chunks) into %s\n", doc.Filename, doc.ChunkCount, colName)
		}

	case "query":
		if len(os.Args) < 5 { log.Fatal("usage: nyawa rag <db> query \"<question>\" [--collection <name>] [--top-k <n>]") }
		question := os.Args[4]
		colName := ""
		topK := 5
		for i := 5; i < len(os.Args)-1; i++ {
			switch os.Args[i] {
			case "--collection": colName = os.Args[i+1]
			case "--top-k": fmt.Sscanf(os.Args[i+1], "%d", &topK)
			}
		}
		results, err := rs.Query(question, topK, colName)
		if err != nil { log.Fatalf("query: %v", err) }
		if len(results) == 0 { fmt.Println("No results"); return }
		fmt.Printf("=== RAG Results (top-%d) ===\n", len(results))
		for i, r := range results {
			fmt.Printf("\n--- Result #%d [%.4f] from %s chunk#%d ---\n%s\n", i+1, r.Score, r.Document, r.ChunkIdx, r.Content)
		}

	case "stats":
		stats := rs.Stats()
		b, _ := json.MarshalIndent(stats, "", "  ")
		fmt.Println(string(b))

	default:
		log.Fatalf("unknown rag command: %s", sub)
	}
}

func cmdDashboard() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa dashboard <db>") }
	dbPath := os.Args[2]

	initEmbedders()
	s := getStore(dbPath, nil)
	defer s.Close()

	storeStats, _ := s.Stats()
	total, _ := storeStats["total_memories"].(int)
	vecIdx, _ := storeStats["vector_indexed"].(int)
	pinned, _ := storeStats["pinned_memories"].(int)
	superseded, _ := storeStats["superseded"].(int)
	en, _ := storeStats["entity_nodes"].(int)
	ee, _ := storeStats["entity_edges"].(int)
	nsMap, _ := storeStats["namespaces"].(map[string]int)

	hnswInfo := s.GetHNSW().Info()
	hnswNodes, _ := hnswInfo["total_nodes"].(int)
	hnswLayers, _ := hnswInfo["layers"].(int)
	hnswEdges, _ := hnswInfo["edges"].(int)
	hnswEntry, _ := hnswInfo["entry_point"].(string)
	hnswCfg, _ := hnswInfo["config"].(map[string]any)
	hnswM, _ := hnswCfg["M"].(int)

	var ragCols, ragDocs, ragChunks int
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_collections`).Scan(&ragCols)
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_documents`).Scan(&ragDocs)
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_chunks`).Scan(&ragChunks)

	memEmbName, memEmbOK := "none", false
	if memEmb := getMemoryEmbedder(); memEmb != nil {
		memEmbName, memEmbOK = memEmb.HealthCheck()
		memEmb.StopAll()
	}
	ragEmbName, ragEmbOK := "none", false
	if ragEmb := getRAGEmbedder(); ragEmb != nil {
		ragEmbName, ragEmbOK = ragEmb.HealthCheck()
		ragEmb.StopAll()
	}

	hnswPath := s.GetHNSWPath()
	var hnswSize int64
	if fi, err := os.Stat(hnswPath); err == nil { hnswSize = fi.Size() }
	var dbSize int64
	if fi, err := os.Stat(dbPath); err == nil { dbSize = fi.Size() }

	status := func(ok bool) string { if ok { return "available" }; return "unavailable" }
	nsCount := len(nsMap)

	fmt.Println()
	fmt.Println("══════════════════════════════════════════════════════════════")
	fmt.Println("  NYAWA DASHBOARD")
	fmt.Println("══════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("── Memory ──")
	fmt.Printf("  %-20s %d\n", "Total:", total)
	fmt.Printf("  %-20s %d\n", "Vector Indexed:", vecIdx)
	if pinned > 0 { fmt.Printf("  %-20s %d\n", "Pinned:", pinned) }
	fmt.Printf("  %-20s %d\n", "Superseded:", superseded)
	fmt.Printf("  %-20s %d\n", "Namespaces:", nsCount)
	if nsCount > 0 {
		for name, cnt := range nsMap {
			fmt.Printf("    %-18s %d\n", name+":", cnt)
		}
	}
	fmt.Println()
	fmt.Println("── Embedder ──")
	fmt.Printf("  %-20s %s (%s)\n", "Memory Embedder:", memEmbName, status(memEmbOK))
	fmt.Printf("  %-20s %s (%s)\n", "RAG Embedder:", ragEmbName, status(ragEmbOK))
	fmt.Println()
	fmt.Println("── RAG ──")
	fmt.Printf("  %-20s %d\n", "Collections:", ragCols)
	fmt.Printf("  %-20s %d\n", "Documents:", ragDocs)
	fmt.Printf("  %-20s %d\n", "Chunks:", ragChunks)
	fmt.Println()
	fmt.Println("── HNSW Index ──")
	fmt.Printf("  %-20s %d\n", "Total Nodes:", hnswNodes)
	fmt.Printf("  %-20s %d\n", "Graph Layers:", hnswLayers)
	fmt.Printf("  %-20s %d\n", "Total Edges:", hnswEdges)
	fmt.Printf("  %-20s %d (M)\n", "M Config:", hnswM)
	if hnswEntry != "" {
		fmt.Printf("  %-20s %s\n", "Entry Point:", hnswEntry)
	}
	fmt.Println()
	fmt.Println("── Entity Graph ──")
	fmt.Printf("  %-20s %d\n", "Entity Nodes:", en)
	fmt.Printf("  %-20s %d\n", "Entity Edges:", ee)
	fmt.Println()
	fmt.Println("── System ──")
	fmt.Printf("  %-20s v0.9.0\n", "Version:")
	fmt.Printf("  %-20s %s\n", "DB File:", fmt.Sprintf("%.1f MB", float64(dbSize)/1024/1024))
	fmt.Printf("  %-20s %s\n", "HNSW File:", fmt.Sprintf("%.1f MB", float64(hnswSize)/1024/1024))
	fmt.Println("══════════════════════════════════════════════════════════════")
}

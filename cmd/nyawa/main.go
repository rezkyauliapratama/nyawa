// Nyawa — Offline-First AI Memory Engine
// Phase 2b: BGE-small ONNX Embedder + PriorityChain
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/mcp"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/server"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
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
	case "version": fmt.Println("nyawa v0.5.0 — Phase 2b")
	default: printUsage(); os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`Nyawa — Offline-First AI Memory Engine

Usage:
  nyawa init <db-path>      Initialize
  nyawa store ...            Store
  nyawa recall ...           Search
  nyawa stats ...            Statistics
  nyawa serve ...            HTTP server
  nyawa mcp ...              MCP server
  nyawa version              Version
`)
}

func getStore(p string, emb store.Embedder) *store.Store {
	s, err := store.NewStore(p, emb)
	if err != nil { log.Fatalf("store: %v", err) }
	return s
}

func getEmbedder() *embedder.PriorityChain {
	bge := embedder.NewPythonEmbedder("/opt/data/nyawa/internal/embedder/model")
	if err := bge.Start(); err != nil {
		log.Printf("BGE unavailable: %v", err)
	} else {
		log.Printf("BGE embedder ready")
	}
	ollama := embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"})
	return embedder.NewPriorityChain(bge, ollama)
}

func cmdInit() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa init <db-path>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats()
	fmt.Printf("Initialized: %s (%d mem)\n", os.Args[2], stats["total_memories"])
}

func cmdStore() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa store <db> <content>") }
	emb := getEmbedder(); defer emb.StopAll()
	s := getStore(os.Args[2], emb); defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	s.InsertMemory(&types.Memory{ID: id, Content: os.Args[3], Type: types.TypeNote, Namespace: "default"})
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa recall <db> <query>") }
	emb := getEmbedder(); defer emb.StopAll()
	s := getStore(os.Args[2], emb); defer s.Close()
	p := search.NewPipeline(s, emb, types.DefaultConfig().Search)
	results, err := p.Search(types.StoreQuery{QueryText: os.Args[3], Limit: 10})
	if err != nil { log.Fatalf("search: %v", err) }
	defer p.ReleaseResults(results)
	if len(results) == 0 { fmt.Println("No results."); return }
	for i, r := range results { fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content) }
}
func cmdSearch() { cmdRecall() }

func cmdStats() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa stats <db>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.MarshalIndent(stats, "", "  "); fmt.Println(string(b))
}

func cmdServe() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa serve <db>") }
	emb := getEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb); defer st.Close()
	hc := embedder.NewHealthCheckRunner(emb, 60*time.Second); hc.Start(); defer hc.Stop()
	p := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	srv := server.New(st, p, emb, server.DefaultServerConfig())
	log.Printf("Server — db=%s embedder=%s", os.Args[2], emb.Current())
	if err := srv.Start(); err != nil { log.Fatalf("server: %v", err) }
}

func cmdMCP() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa mcp <db>") }
	emb := getEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb)
	m := mcp.NewServer(st)
	log.Printf("MCP — db=%s", os.Args[2])
	if err := m.Run(); err != nil { log.Fatalf("mcp: %v", err) }
}

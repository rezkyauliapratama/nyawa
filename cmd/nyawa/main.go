// Nyawa — Offline-First AI Memory Engine
// Phase 2a: HNSW Vector Index + Embedder Integration
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
	case "version": fmt.Println("nyawa v0.4.0 — Phase 2a")
	default: fmt.Fprintf(os.Stderr, "unknown: %s\n", os.Args[1]); printUsage(); os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`Nyawa — Offline-First AI Memory Engine

Usage:
  nyawa init <db-path>                Initialize memory store
  nyawa store <db-path> <content>     Store a memory
  nyawa recall <db-path> <query>      Search memories
  nyawa search <db-path> <query>      Alias for recall
  nyawa stats <db-path>               Show statistics
  nyawa serve <db-path>               Start HTTP API server
  nyawa mcp <db-path>                 Start MCP stdio server
  nyawa version                       Show version
`)
}

func getStore(p string, emb store.Embedder) *store.Store {
	s, err := store.NewStore(p, emb)
	if err != nil { log.Fatalf("store: %v", err) }
	return s
}

func getEmbedder() *embedder.PriorityChain {
	return embedder.NewPriorityChain(embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"}))
}

func cmdInit() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa init <db-path>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats()
	fmt.Printf("Initialized: %s (%d memories, %d vectors)\n", os.Args[2], stats["total_memories"], stats["vector_indexed"])
}

func cmdStore() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa store <db-path> <content>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	s.InsertMemory(&types.Memory{ID: id, Content: os.Args[3], Type: types.TypeNote, Namespace: "default", CreatedAt: time.Now(), UpdatedAt: time.Now()})
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 { log.Fatal("usage: nyawa recall <db-path> <query>") }
	emb := getEmbedder()
	s := getStore(os.Args[2], emb); defer s.Close()
	p := search.NewPipeline(s, emb, types.DefaultConfig().Search)
	results, err := p.Search(types.StoreQuery{QueryText: os.Args[3], Limit: 10, Namespace: "default"})
	if err != nil { log.Fatalf("search: %v", err) }
	defer p.ReleaseResults(results)
	if len(results) == 0 { fmt.Println("No results."); return }
	fmt.Printf("%d results:\n", len(results))
	for i, r := range results { fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content) }
}
func cmdSearch() { cmdRecall() }

func cmdStats() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa stats <db-path>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.MarshalIndent(stats, "", "  ")
	fmt.Println(string(b))
}

func cmdServe() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa serve <db-path>") }
	emb := getEmbedder()
	st := getStore(os.Args[2], emb); defer st.Close()
	hc := embedder.NewHealthCheckRunner(emb, 60*time.Second); hc.Start(); defer hc.Stop()
	p := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	srv := server.New(st, p, emb, server.DefaultServerConfig())
	log.Printf("Nyawa server — db=%s embedder=%s", os.Args[2], emb.Current())
	if err := srv.Start(); err != nil { log.Fatalf("server error: %v", err) }
}

func cmdMCP() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa mcp <db-path>") }
	emb := getEmbedder()
	st := getStore(os.Args[2], emb)
	m := mcp.NewServer(st)
	log.Printf("Nyawa MCP — db=%s", os.Args[2])
	if err := m.Run(); err != nil { log.Fatalf("mcp error: %v", err) }
}

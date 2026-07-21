// Nyawa — Offline-First AI Memory Engine
// Phase 1b: HTTP API Server + Security Filter + Health Check
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/search"
	"github.com/rezkyauliapratama/nyawa/internal/server"
	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("nyawa: ")
	if len(os.Args) < 2 {
		printUsage(); os.Exit(1)
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
	case "version":
		fmt.Println("nyawa v0.2.0 — Phase 1b")
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage(); os.Exit(1)
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
  nyawa version                       Show version
`)
}

func getStore(dbPath string) *store.Store {
	s, err := store.NewStore(dbPath)
	if err != nil {
		log.Fatalf("open store %s: %v", dbPath, err)
	}
	return s
}

func getEmbedder() *embedder.PriorityChain {
	ollama := embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"})
	return embedder.NewPriorityChain(ollama)
}

func cmdInit() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa init <db-path>")
	}
	s := getStore(os.Args[2]); defer s.Close()
	stats, _ := s.Stats()
	fmt.Printf("Initialized: %s (%d memories)\n", os.Args[2], stats["total_memories"])
}

func cmdStore() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa store <db-path> <content>")
	}
	s := getStore(os.Args[2]); defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	s.InsertMemory(&types.Memory{ID: id, Content: os.Args[3], Type: types.TypeNote, Namespace: "default", CreatedAt: time.Now(), UpdatedAt: time.Now()})
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa recall <db-path> <query>")
	}
	s := getStore(os.Args[2]); defer s.Close()
	pipeline := search.NewPipeline(s, getEmbedder(), types.DefaultConfig().Search)
	results, err := pipeline.Search(types.StoreQuery{QueryText: os.Args[3], Limit: 10, Namespace: "default"})
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	defer pipeline.ReleaseResults(results)
	if len(results) == 0 {
		fmt.Println("No results."); return
	}
	fmt.Printf("%d results:\n", len(results))
	for i, r := range results {
		fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content)
	}
}

func cmdSearch() { cmdRecall() }

func cmdStats() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa stats <db-path>")
	}
	s := getStore(os.Args[2]); defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.MarshalIndent(stats, "", "  ")
	fmt.Println(string(b))
}

func cmdServe() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa serve <db-path>")
	}
	dbPath := os.Args[2]
	st := getStore(dbPath); defer st.Close()
	emb := getEmbedder()
	hc := embedder.NewHealthCheckRunner(emb, 60*time.Second)
	hc.Start(); defer hc.Stop()
	pipeline := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	srv := server.New(st, pipeline, emb, server.DefaultServerConfig())
	log.Printf("Nyawa server starting — db=%s embedder=%s", dbPath, emb.Current())
	if err := srv.Start(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

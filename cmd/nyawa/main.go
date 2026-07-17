package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/search"
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
		cmdRecall()
	case "stats":
		cmdStats()
	case "init":
		cmdInit()
	case "version":
		fmt.Println("nyawa v0.1.0 — Phase 1a")
	default:
		fmt.Fprintf(os.Stderr, "unknown: %s\n", os.Args[1])
		printUsage(); os.Exit(1)
	}
}

func printUsage() {
	fmt.Print(`Nyawa — Offline-First AI Memory Engine

Usage:
  nyawa init <db-path>
  nyawa store <db-path> <content>
  nyawa recall <db-path> <query>
  nyawa stats <db-path>
  nyawa version
`)
}

func getStore(p string) *store.Store {
	s, err := store.NewStore(p)
	if err != nil {
		log.Fatalf("store: %v", err)
	}
	return s
}

func cmdInit() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa init <db-path>")
	}
	s := getStore(os.Args[2])
	defer s.Close()
	stats, _ := s.Stats()
	fmt.Printf("Initialized: %s (%d memories)\n", os.Args[2], stats["total_memories"])
}

func cmdStore() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa store <db-path> <content>")
	}
	s := getStore(os.Args[2])
	defer s.Close()
	id := fmt.Sprintf("mem_%d", time.Now().UnixNano())
	s.InsertMemory(&types.Memory{ID: id, Content: os.Args[3], Type: types.TypeNote, Namespace: "default", CreatedAt: time.Now(), UpdatedAt: time.Now()})
	fmt.Printf("Stored: %s\n", id)
}

func cmdRecall() {
	if len(os.Args) < 4 {
		log.Fatal("usage: nyawa recall <db-path> <query>")
	}
	s := getStore(os.Args[2])
	defer s.Close()
	cfg := types.DefaultConfig()
	p := search.NewPipeline(s, embedder.NewPriorityChain(
		embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"}),
	), cfg.Search)
	results, err := p.Search(types.StoreQuery{QueryText: os.Args[3], Limit: 10, Namespace: "default"})
	if err != nil {
		log.Fatalf("search: %v", err)
	}
	defer p.ReleaseResults(results)
	if len(results) == 0 {
		fmt.Println("No results.")
		return
	}
	fmt.Printf("%d results:\n", len(results))
	for i, r := range results {
		fmt.Printf("#%d [%.4f] %s\n", i+1, r.Score, r.Content)
	}
}

func cmdStats() {
	if len(os.Args) < 3 {
		log.Fatal("usage: nyawa stats <db-path>")
	}
	s := getStore(os.Args[2])
	defer s.Close()
	stats, _ := s.Stats()
	b, _ := json.MarshalIndent(stats, "", "  ")
	fmt.Println(string(b))
}

// Nyawa — Offline-First AI Memory Engine
// Phase 4: Scale & Stability
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/dream"
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
	case "dream": cmdDream()
	case "ns": cmdNamespace()
	case "archive": cmdArchive()
	case "version": fmt.Println("nyawa v0.8.0 — Phase 4")
	default: printUsage(); os.Exit(1)
	}
}

func printUsage() {
	fmt.Printf(`Nyawa — Offline-First AI Memory Engine v0.8.0

Usage:
  nyawa init <db-path>
  nyawa store <db> <content>
  nyawa recall <db> <q> [--ns <ns>] [--at <time>]
  nyawa stats <db>
  nyawa ns <db>                    List namespaces
  nyawa archive <db> <out>         Archive superseded
  nyawa serve <db>                 HTTP + Dream Cycle
  nyawa mcp <db>                   MCP server
  nyawa dream <db>                 Dream Cycle
  nyawa version
`)
}

func getStore(p string, emb store.Embedder) *store.Store {
	s, err := store.NewStore(p, emb)
	if err != nil { log.Fatalf("store: %v", err) }
	return s
}
func getEmbedder() *embedder.PriorityChain {
	bge := embedder.NewPythonEmbedder("/opt/data/nyawa/internal/embedder/model")
	if err := bge.Start(); err != nil { log.Printf("BGE unavailable: %v", err) } else { log.Printf("BGE ready") }
	ollama := embedder.NewOllamaEmbedder(embedder.OllamaConfig{BaseURL: "http://localhost:11434", Model: "nomic-embed-text"})
	return embedder.NewPriorityChain(bge, ollama)
}
func cmdInit() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa init <db-path>") }
	s := getStore(os.Args[2], nil); defer s.Close()
	stats, _ := s.Stats(); b, _ := json.Marshal(stats); fmt.Println(string(b))
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
	if len(os.Args) < 4 { log.Fatal("usage: nyawa recall <db> <q> [--ns <ns>] [--at <time>]") }
	ns, atTime := "", time.Time{}
	for i := 2; i < len(os.Args); i++ {
		if os.Args[i] == "--ns" && i+1 < len(os.Args) { ns = os.Args[i+1] }
		if os.Args[i] == "--at" && i+1 < len(os.Args) {
			if t, err := time.Parse(time.RFC3339, os.Args[i+1]); err == nil { atTime = t }
		}
	}
	emb := getEmbedder(); defer emb.StopAll()
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
func cmdServe() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa serve <db>") }
	emb := getEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb); defer st.Close()
	engine := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	engine.Start(dream.DefaultConfig())
	hc := embedder.NewHealthCheckRunner(emb, 60*time.Second); hc.Start(); defer hc.Stop()
	p := search.NewPipeline(st, emb, types.DefaultConfig().Search)
	srv := server.New(st, p, emb, server.DefaultServerConfig())
	log.Printf("Server — db=%s embedder=%s dream=%v", os.Args[2], emb.Current(), engine.Running())
	if err := srv.Start(); err != nil { log.Fatalf("server: %v", err) }
}
func cmdMCP() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa mcp <db>") }
	emb := getEmbedder(); defer emb.StopAll()
	st := getStore(os.Args[2], emb)
	log.Printf("MCP — db=%s", os.Args[2])
	if err := mcp.NewServer(st).Run(); err != nil { log.Fatalf("mcp: %v", err) }
}
func cmdDream() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa dream <db-path>") }
	st := getStore(os.Args[2], nil); defer st.Close()
	stats, _ := st.Stats(); b, _ := json.MarshalIndent(stats, "", "  "); fmt.Println(string(b))
	fmt.Println("--- Running Dream Cycle ---")
	engine := dream.New(st.GetDB(), st.GetHNSW(), st.GetHNSWPath())
	cfg := dream.DefaultConfig()
	result := engine.Run(cfg)
	b2, _ := json.MarshalIndent(result, "", "  "); fmt.Println(string(b2))
}

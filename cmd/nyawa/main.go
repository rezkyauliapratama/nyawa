		emb := getEmbedder(); defer emb.StopAll()
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
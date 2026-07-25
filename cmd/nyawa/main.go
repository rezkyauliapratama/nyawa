func cmdDashboard() {
	if len(os.Args) < 3 { log.Fatal("usage: nyawa dashboard <db>") }
	dbPath := os.Args[2]

	initEmbedders()

	s := getStore(dbPath, nil)
	defer s.Close()

	storeStats, _ := s.Stats()
	total, _ := storeStats["total_memories"].(int)
	vecIdx, _ := storeStats["vector_indexed"].(int)
	superseded, _ := storeStats["superseded"].(int)
	en, _ := storeStats["entity_nodes"].(int)
	ee, _ := storeStats["entity_edges"].(int)
	nsMap, _ := storeStats["namespaces"].(map[string]int)

	var colCount, docCount, chunkCount int
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_collections`).Scan(&colCount)
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_documents`).Scan(&docCount)
	s.GetDB().QueryRow(`SELECT COUNT(*) FROM rag_chunks`).Scan(&chunkCount)

	hnsw := s.GetHNSW()
	hnswNodes := 0
	hnswLayers := 0
	hnswEdges := 0
	hnswM := 0
	hnswEP := ""
	if hnsw != nil {
		info := hnsw.Info()
		hnswNodes, _ = info["total_nodes"].(int)
		hnswLayers, _ = info["layers"].(int)
		hnswEdges, _ = info["edges"].(int)
		if cfg, ok := info["config"].(map[string]interface{}); ok {
			if m, ok := cfg["M"].(int); ok { hnswM = m }
		}
		hnswEP, _ = info["entry_point"].(string)
	}

	if nsMap == nil { nsMap = map[string]int{} }

	sep := strings.Repeat("\u2550", 64)
	fmt.Printf("\n%s\n  NYAWA DASHBOARD\n%s\n\n", sep, sep)

	fmt.Printf("\u2500\u2500 Memory \u2500\u2500\n")
	fmt.Printf("  Total:               %d\n", total)
	fmt.Printf("  Vector Indexed:      %d\n", vecIdx)
	fmt.Printf("  Superseded:          %d\n", superseded)
	fmt.Printf("  Namespaces:          %d\n", len(nsMap))
	for name, count := range nsMap {
		fmt.Printf("    %-20s %d\n", name+":", count)
	}

	fmt.Printf("\n\u2500\u2500 Embedder \u2500\u2500\n")
	eName := "unavailable"
	if bgeEmbedder != nil && bgeEmbedder.Available() {
		eName = "bge-small"
	} else if apiEmbedder != nil && apiEmbedder.Available() {
		eName = apiEmbedder.Name()
	}
	fmt.Printf("  Memory Embedder:     %s\n", eName)

	fmt.Printf("\n\u2500\u2500 RAG \u2500\u2500\n")
	fmt.Printf("  Collections:         %d\n", colCount)
	fmt.Printf("  Documents:           %d\n", docCount)
	fmt.Printf("  Chunks:              %d\n", chunkCount)

	fmt.Printf("\n\u2500\u2500 HNSW Index \u2500\u2500\n")
	fmt.Printf("  Total Nodes:         %d\n", hnswNodes)
	fmt.Printf("  Graph Layers:        %d\n", hnswLayers)
	fmt.Printf("  Total Edges:         %d\n", hnswEdges)
	fmt.Printf("  M Config:            %d (M)\n", hnswM)
	if hnswEP != "" { fmt.Printf("  Entry Point:         %s\n", hnswEP) }

	fmt.Printf("\n\u2500\u2500 Entity Graph \u2500\u2500\n")
	fmt.Printf("  Entity Nodes:        %d\n", en)
	fmt.Printf("  Entity Edges:        %d\n", ee)

	fmt.Printf("\n\u2500\u2500 System \u2500\u2500\n")
	fmt.Printf("  Version:             v0.9.0\n")
	dbFileInfo, _ := os.Stat(dbPath)
	if dbFileInfo != nil {
		fmt.Printf("  DB File:             %.1f MB\n", float64(dbFileInfo.Size())/1024/1024)
	}
	hnswFileInfo, _ := os.Stat(dbPath + ".hnsw")
	if hnswFileInfo != nil {
		fmt.Printf("  HNSW File:           %.1f MB\n", float64(hnswFileInfo.Size())/1024/1024)
	}
	fmt.Printf("%s\n", sep)
}
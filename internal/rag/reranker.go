// Package rag implements document-level RAG (Retrieval-Augmented Generation) for Nyawa.
package rag

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
	"time"
)

type Reranker interface {
	Rerank(query string, candidates []string) ([]float64, error)
	Available() bool
	Name() string
}

// ---- Python Cross-Encoder Reranker (Offline) ------------------------------

type PythonCrossEncoder struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  *json.Encoder
	stdout *bufio.Scanner
	ready  bool
}

const LocalCrossEncoderName = "Horizon-Labs/multilingual-reranker-small"

func LocalCrossEncoderEnabled() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("NYAWA_ENABLE_CROSS_ENCODER")))
	return value == "1" || value == "true" || value == "yes"
}

func NewPythonCrossEncoder() *PythonCrossEncoder {
	pe := &PythonCrossEncoder{}
	script := findRerankerScript()
	if script == "" {
		log.Printf("reranker: script not found")
		return pe
	}
	modelDir := strings.TrimSpace(os.Getenv("NYAWA_RERANKER_MODEL_DIR"))
	if modelDir == "" {
		modelDir = "/opt/nyawa/model/reranker"
	}
	pe.cmd = exec.Command(findPythonPath(), script)
	env := os.Environ()
	filtered := env[:0]
	for _, entry := range env {
		if !strings.HasPrefix(entry, "NYAWA_RERANKER_MODEL_DIR=") {
			filtered = append(filtered, entry)
		}
	}
	pe.cmd.Env = append(filtered, "NYAWA_RERANKER_MODEL_DIR="+modelDir)
	stdin, _ := pe.cmd.StdinPipe()
	stdout, _ := pe.cmd.StdoutPipe()
	stderr, _ := pe.cmd.StderrPipe()
	if err := pe.cmd.Start(); err != nil {
		log.Printf("reranker: start failed: %v", err)
		return pe
	}
	pe.stdin = json.NewEncoder(stdin)
	pe.stdout = bufio.NewScanner(stdout)

	errCh := make(chan error, 1)
	doneCh := make(chan struct{}, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.Contains(line, "READY") {
				doneCh <- struct{}{}
			}
			if strings.Contains(line, "Traceback") || strings.Contains(line, "Error") {
				errCh <- fmt.Errorf(line)
			}
		}
	}()
	select {
	case <-doneCh:
		pe.ready = true
		log.Printf("reranker: cross-encoder ready")
	case err := <-errCh:
		pe.cmd.Process.Kill()
		log.Printf("reranker: %v", err)
	case <-time.After(60 * time.Second):
		pe.cmd.Process.Kill()
		log.Printf("reranker: timeout")
	}
	return pe
}

func (r *PythonCrossEncoder) Rerank(query string, candidates []string) ([]float64, error) {
	if !r.ready {
		return nil, fmt.Errorf("reranker not ready")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.stdin.Encode(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "rerank",
		"params": map[string]any{"query": query, "texts": candidates},
	}); err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	if !r.stdout.Scan() {
		return nil, fmt.Errorf("no response")
	}
	var resp struct {
		Result *struct {
			Scores []float64 `json:"scores"`
		} `json:"result"`
		Error *struct{ Message string } `json:"error"`
	}
	if err := json.Unmarshal(r.stdout.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("parse: %w", err)
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("reranker: %s", resp.Error.Message)
	}
	if resp.Result == nil || resp.Result.Scores == nil {
		return nil, fmt.Errorf("empty response")
	}
	return resp.Result.Scores, nil
}

func (r *PythonCrossEncoder) Available() bool { return r.ready }
func (r *PythonCrossEncoder) Name() string    { return LocalCrossEncoderName }

func (r *PythonCrossEncoder) Stop() {
	if r.cmd != nil && r.cmd.Process != nil {
		r.cmd.Process.Kill()
		r.cmd.Wait()
	}
}

// ---- OpenAI-Compatible Reranker (3rd Party: Jina, Voyage, etc.) -----------

type OpenAIReranker struct {
	apiKey  string
	baseURL string
	model   string
	client  *http.Client
	name    string
}

type openAIRerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
	TopN      int      `json:"top_n,omitempty"`
}

type openAIRerankResponse struct {
	Results []struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	} `json:"results"`
}

func NewOpenAIReranker() *OpenAIReranker {
	baseURL := os.Getenv("RERANK_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.jina.ai/v1"
	}
	model := os.Getenv("RERANK_MODEL")
	if model == "" {
		model = "jina-reranker-v2-base-multilingual"
	}
	apiKey := os.Getenv("RERANK_API_KEY")
	name := fmt.Sprintf("openai/%s", model)

	return &OpenAIReranker{
		apiKey:  apiKey,
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		client:  &http.Client{Timeout: 30 * time.Second},
		name:    name,
	}
}

func (r *OpenAIReranker) Rerank(query string, candidates []string) ([]float64, error) {
	if r.apiKey == "" {
		return nil, fmt.Errorf("RERANK_API_KEY not set")
	}
	body := openAIRerankRequest{
		Model:     r.model,
		Query:     query,
		Documents: candidates,
		TopN:      len(candidates),
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", r.baseURL+"/rerank", bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+r.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("api call: %w", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result openAIRerankResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("parse: %v", err)
	}

	scores := make([]float64, len(candidates))
	for _, res := range result.Results {
		if res.Index < len(scores) {
			scores[res.Index] = res.RelevanceScore
		}
	}
	return scores, nil
}

func (r *OpenAIReranker) Available() bool { return r.apiKey != "" }
func (r *OpenAIReranker) Name() string    { return r.name }

// ---- Noop Reranker (passthrough) -----------------------------------------

type NoopReranker struct{}

func (n *NoopReranker) Rerank(query string, candidates []string) ([]float64, error) {
	scores := make([]float64, len(candidates))
	for i := range scores {
		scores[i] = 1.0
	}
	return scores, nil
}
func (n *NoopReranker) Available() bool { return true }
func (n *NoopReranker) Name() string    { return "noop" }

// ---- Rerank-aware Query ---------------------------------------------------

func (r *RAGStore) QueryWithRerank(query string, topK int, collectionName string, reranker Reranker) ([]RAGResult, error) {
	fetchK := topK * 5
	if reranker != nil && reranker.Available() && reranker.Name() != "noop" {
		fetchK = topK * 10
	}

	results, err := r.queryInternal(query, fetchK, collectionName)
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}

	if reranker != nil && reranker.Available() && reranker.Name() != "noop" {
		candidates := make([]string, len(results))
		for i, r := range results {
			candidates[i] = r.Content
		}
		scores, err := reranker.Rerank(query, candidates)
		if err == nil {
			for i := range results {
				if i < len(scores) {
					results[i].Score = scores[i]
				}
			}
			for i := 0; i < len(results); i++ {
				for j := i + 1; j < len(results); j++ {
					if (results[j].exactMatch && !results[i].exactMatch) ||
						(results[j].exactMatch == results[i].exactMatch && results[j].Score > results[i].Score) {
						results[i], results[j] = results[j], results[i]
					}
				}
			}
		}
	}

	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func (r *RAGStore) queryInternal(query string, topK int, collectionName string) ([]RAGResult, error) {
	if r.initErr != nil {
		return nil, fmt.Errorf("RAG index unavailable: %w", r.initErr)
	}
	if r.embedder == nil || !r.embedder.Available() {
		return nil, ErrRAGEmbedderUnavailable
	}
	if topK <= 0 {
		topK = 5
	}
	vec, err := embedRAGQuery(r.embedder, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	hnsw, err := r.refreshHNSW()
	if err != nil {
		return nil, err
	}
	indexSize := hnsw.Size()
	vectorByID := make(map[string]retrievalCandidate)
	searchK := topK * 3
	if searchK < topK || searchK > indexSize {
		searchK = indexSize
	}
	for searchK > 0 {
		vectorResults, err := hnsw.SearchCheckedStable(vec, searchK)
		if err != nil {
			return nil, fmt.Errorf("hnsw search: %w", err)
		}
		for _, match := range vectorResults {
			if !strings.HasPrefix(match.ID, "rag_chk_") {
				continue
			}
			var candidate retrievalCandidate
			var indexed int
			err := r.db.QueryRow(`SELECT c.content,c.chunk_index,d.filename,c.vector_indexed
				FROM rag_chunks c JOIN rag_documents d ON d.id=c.document_id
				JOIN rag_collections col ON col.id=d.collection_id
				WHERE c.id=? AND (?='' OR col.name=?)`, match.ID, collectionName, collectionName).Scan(
				&candidate.result.Content, &candidate.result.ChunkIdx, &candidate.result.Document, &indexed)
			if err != nil || indexed != 1 {
				continue
			}
			candidate.result.ChunkID = match.ID
			candidate.result.Score = 1 / (1 + match.Distance)
			vectorByID[match.ID] = candidate
		}
		if len(vectorByID) >= topK || searchK >= indexSize {
			break
		}
		nextSearchK := searchK * 2
		if nextSearchK <= searchK || nextSearchK > indexSize {
			nextSearchK = indexSize
		}
		searchK = nextSearchK
	}
	vectorCandidates := make([]retrievalCandidate, 0, len(vectorByID))
	for _, candidate := range vectorByID {
		vectorCandidates = append(vectorCandidates, candidate)
	}
	// The HNSW implementation does not define order for equal distances. Make
	// RRF ranks reproducible before fusion, including across map iteration.
	sortRAGVectorCandidates(vectorCandidates)
	lexicalCandidates, err := r.lexicalCandidates(query, collectionName, topK*3)
	if err != nil {
		return nil, err
	}
	results := fuseRAGCandidates(vectorCandidates, lexicalCandidates)
	if len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

func sortRAGVectorCandidates(candidates []retrievalCandidate) {
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].result.Score != candidates[j].result.Score {
			return candidates[i].result.Score > candidates[j].result.Score
		}
		return candidates[i].result.ChunkID < candidates[j].result.ChunkID
	})
}

func findRerankerScript() string {
	paths := []string{
		"/opt/data/nyawa/internal/rag/reranker.py",
		"internal/rag/reranker.py",
	}
	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	return ""
}

func findPythonPath() string {
	for _, name := range []string{"python3", "python"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return "python3"
}

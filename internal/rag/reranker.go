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
	"strings"
	"sync"
	"time"
)

// Reranker re-ranks retrieved chunks by semantic relevance to the query.
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

func NewPythonCrossEncoder() *PythonCrossEncoder {
	pe := &PythonCrossEncoder{}
	script := findRerankerScript()
	if script == "" {
		log.Printf("reranker: script not found")
		return pe
	}
	pe.cmd = exec.Command(findPythonPath(), script)
	pe.cmd.Env = append(os.Environ(), "NYAWA_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2")
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
			if strings.Contains(line, "READY") { doneCh <- struct{}{} }
			if strings.Contains(line, "Traceback") || strings.Contains(line, "Error") { errCh <- fmt.Errorf(line) }
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
		Result *struct{ Scores []float64 `json:"scores"` } `json:"result"`
		Error  *struct{ Message string } `json:"error"`
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
func (r *PythonCrossEncoder) Name() string    { return "cross-encoder/ms-marco-MiniLM-L6-v2" }

func (r *PythonCrossEncoder) Stop() {
	if r.cmd != nil && r.cmd.Process != nil {
		r.cmd.Process.Kill()
		r.cmd.Wait()
	}
}

// ---- Cohere Reranker (3rd Party) -----------------------------------------

type CohereReranker struct {
	apiKey string
	model  string
	client *http.Client
}

func NewCohereReranker(apiKey string) *CohereReranker {
	if apiKey == "" {
		apiKey = os.Getenv("COHERE_API_KEY")
	}
	return &CohereReranker{
		apiKey: apiKey,
		model:  "rerank-v3.5",
		client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *CohereReranker) Rerank(query string, candidates []string) ([]float64, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("COHERE_API_KEY not set")
	}
	body := map[string]any{
		"model":     c.model,
		"query":     query,
		"documents": candidates,
		"top_n":     len(candidates),
	}
	b, _ := json.Marshal(body)
	req, err := http.NewRequest("POST", "https://api.cohere.com/v2/rerank", bytes.NewReader(b))
	if err != nil { return nil, err }
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil { return nil, err }
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result struct {
		Results []struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("cohere: %v", err)
	}

	scores := make([]float64, len(candidates))
	for _, r := range result.Results {
		if r.Index < len(scores) {
			scores[r.Index] = r.RelevanceScore
		}
	}
	return scores, nil
}

func (c *CohereReranker) Available() bool { return c.apiKey != "" }
func (c *CohereReranker) Name() string    { return "cohere/" + c.model }

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
	if err != nil { return nil, err }
	if len(results) == 0 { return nil, nil }

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
					if results[j].Score > results[i].Score {
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
	if r.embedder == nil || !r.embedder.Available() {
		return nil, fmt.Errorf("embedder not available")
	}
	if topK <= 0 { topK = 5 }

	vec, err := r.embedder.Embed(query)
	if err != nil { return nil, fmt.Errorf("embed query: %w", err) }

	results := r.hnsw.Search(vec, topK*3)
	if len(results) == 0 { return nil, nil }

	var ragResults []RAGResult
	for _, res := range results {
		if !strings.HasPrefix(res.ID, "rag_chk_") { continue }
		var content string
		var docID string
		var chunkIdx int
		if err := r.db.QueryRow(`SELECT content, document_id, chunk_index FROM rag_chunks WHERE id=?`, res.ID).Scan(&content, &docID, &chunkIdx); err != nil {
			continue
		}
		if collectionName != "" {
			var colID int
			r.db.QueryRow(`SELECT collection_id FROM rag_documents WHERE id=?`, docID).Scan(&colID)
			if colID == 0 { continue }
			var colName string
			r.db.QueryRow(`SELECT name FROM rag_collections WHERE id=?`, colID).Scan(&colName)
			if colName != collectionName { continue }
		}
		var filename string
		r.db.QueryRow(`SELECT filename FROM rag_documents WHERE id=?`, docID).Scan(&filename)
		ragResults = append(ragResults, RAGResult{
			ChunkID: res.ID, Content: content, Score: 1.0 / (1.0 + res.Distance),
			Document: filename, ChunkIdx: chunkIdx,
		})
	}
	for i := 0; i < len(ragResults); i++ {
		for j := i + 1; j < len(ragResults); j++ {
			if ragResults[j].Score > ragResults[i].Score {
				ragResults[i], ragResults[j] = ragResults[j], ragResults[i]
			}
		}
	}
	if len(ragResults) > topK {
		ragResults = ragResults[:topK]
	}
	return ragResults, nil
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

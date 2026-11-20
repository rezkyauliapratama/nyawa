package main

import (
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/embedder"
	"github.com/rezkyauliapratama/nyawa/internal/rag"
)

func TestRAGReindexProfileAcceptsConfiguredLocalEndpoint(t *testing.T) {
	// A self-hosted OpenAI-compatible embedder reports no retrieval tasks; the
	// expectation must not demand the hosted Jina task profile in that case.
	expect := ragReindexProfileExpectation{
		Model:      "bge-m3",
		BaseURL:    "http://nyawa-embed:8080/v1",
		Dimensions: 1024,
		Tasks:      false,
		TaskMode:   ragTaskModeAuto,
	}
	profile := rag.EmbeddingProfile{
		Provider:   "openai-compatible",
		Model:      "bge-m3",
		BaseURL:    "http://nyawa-embed:8080/v1",
		Dimensions: 1024,
	}
	if err := expect.validate(profile); err != nil {
		t.Fatalf("expected acceptance, got %v", err)
	}
}

func TestRAGReindexProfileToleratesTrailingSlash(t *testing.T) {
	expect := ragReindexProfileExpectation{
		Model: "bge-m3", BaseURL: "http://nyawa-embed:8080/v1/", Dimensions: 1024,
	}
	profile := rag.EmbeddingProfile{
		Model: "bge-m3", BaseURL: "http://nyawa-embed:8080/v1", Dimensions: 1024,
	}
	if err := expect.validate(profile); err != nil {
		t.Fatalf("expected trailing slash tolerance, got %v", err)
	}
}

func TestRAGReindexProfileRejectsShapeDrift(t *testing.T) {
	base := rag.EmbeddingProfile{
		Model: "bge-m3", BaseURL: "http://nyawa-embed:8080/v1", Dimensions: 1024,
	}
	cases := []struct {
		name   string
		expect ragReindexProfileExpectation
		field  string
	}{
		{
			name:   "model",
			expect: ragReindexProfileExpectation{Model: "jina-embeddings-v3", BaseURL: base.BaseURL, Dimensions: 1024},
			field:  "model=",
		},
		{
			name:   "dimensions",
			expect: ragReindexProfileExpectation{Model: base.Model, BaseURL: base.BaseURL, Dimensions: 384},
			field:  "dimensions=",
		},
		{
			name:   "base_url",
			expect: ragReindexProfileExpectation{Model: base.Model, BaseURL: "https://api.jina.ai/v1", Dimensions: 1024},
			field:  "base_url=",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.expect.validate(base)
			if err == nil {
				t.Fatalf("expected mismatch for %s", tc.name)
			}
			if !strings.Contains(err.Error(), tc.field) {
				t.Fatalf("error %q does not mention %q", err.Error(), tc.field)
			}
		})
	}
}

func TestRAGReindexProfileRequiresRetrievalTasksOnlyWhenEnabled(t *testing.T) {
	expect := ragReindexProfileExpectation{
		Model: "jina-embeddings-v3", BaseURL: "https://api.jina.ai/v1",
		Dimensions: 1024, Tasks: true, TaskMode: ragTaskModeJinaV3,
	}
	missing := rag.EmbeddingProfile{
		Model: expect.Model, BaseURL: expect.BaseURL, Dimensions: expect.Dimensions,
	}
	if err := expect.validate(missing); err == nil {
		t.Fatal("expected missing tasks to be rejected when task mode requires them")
	}

	full := missing
	full.QueryTask = embedder.TaskRetrievalQuery
	full.PassageTask = embedder.TaskRetrievalPassage
	if err := expect.validate(full); err != nil {
		t.Fatalf("expected acceptance with both tasks set, got %v", err)
	}
}

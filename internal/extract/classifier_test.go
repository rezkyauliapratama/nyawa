package extract

import (
	"testing"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestInferType(t *testing.T) {
	c := NewClassifier()
	for _, tt := range []struct{ content string; want types.MemoryType }{
		{"We decided to use Go", types.TypeDecision},
		{"I discovered that HNSW is faster", types.TypeInsight},
		{"Step 1: Install Go. Step 2: Build", types.TypeProcedure},
		{"The database is PostgreSQL", types.TypeFact},
		{"I prefer dark mode", types.TypePreference},
		{"Currently working on HNSW index", types.TypeContext},
		{"The meeting happened yesterday", types.TypeEvent},
		{"See the docs at https://example.com", types.TypeReference},
		{"Just a random note", types.TypeNote},
	} {
		if got := c.InferType(tt.content); got != tt.want {
			t.Errorf("InferType(%q) = %v, want %v", tt.content[:20], got, tt.want)
		}
	}
}

func TestExtractEntities(t *testing.T) {
	c := NewClassifier()
	e := c.ExtractEntities("Deployed to GCP on 2026-07-21 using Go v1.23.0 at https://example.com")
	if len(e.URLs) == 0 { t.Error("expected URL") }
	if len(e.Dates) == 0 { t.Error("expected date") }
	_ = e
}

func TestProcess(t *testing.T) {
	c := NewClassifier()
	_, mt := c.Process("We decided to deploy to AWS using Terraform")
	if mt != types.TypeDecision { t.Errorf("expected decision, got %v", mt) }
}

package extract

import (
	"testing"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestInferType(t *testing.T) {
	c := NewClassifier()
	for _, tt := range []struct{ content string; want types.MemoryType }{
		{"We decided to use Go", types.TypeDecision},
		{"I discovered that HNSW outperforms", types.TypeInsight},
		{"Step 1: Install Go. Step 2: Build", types.TypeProcedure},
		{"The database is PostgreSQL", types.TypeFact},
		{"I prefer dark mode", types.TypePreference},
		{"Currently working on HNSW", types.TypeContext},
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
	entities := c.ExtractEntities("Deployed to GCP with Kubernetes at https://console.cloud.google.com on 2026-07-21")
	if len(entities.URLs) == 0 { t.Error("expected URL extraction") }
	if len(entities.Dates) == 0 { t.Error("expected date extraction") }
}

func TestProcess(t *testing.T) {
	c := NewClassifier()
	_, memType := c.Process("We decided to deploy to AWS using Terraform on 2026-07-20")
	if memType != types.TypeDecision { t.Errorf("expected decision, got %v", memType) }
}

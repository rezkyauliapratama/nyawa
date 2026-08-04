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
	if !contains(entities.Tech, "GCP") { t.Errorf("expected GCP tech, got %v", entities.Tech) }
	if !contains(entities.Tech, "Kubernetes") { t.Errorf("expected Kubernetes tech, got %v", entities.Tech) }
}

func TestProcess(t *testing.T) {
	c := NewClassifier()
	_, memType := c.Process("We decided to deploy to AWS using Terraform on 2026-07-20")
	if memType != types.TypeDecision { t.Errorf("expected decision, got %v", memType) }
}

// TestExtractEntitiesCaseInsensitive verifies tech detection is
// case-insensitive and aliases normalize to canonical names.
func TestExtractEntitiesCaseInsensitive(t *testing.T) {
	c := NewClassifier()
	tests := []struct{ content string; want []string }{
		{"Model: deepseek-v4-flash via API", []string{"DeepSeek"}},
		{"running on AWS Bedrock in Jakarta", []string{"AWS", "AWS Bedrock"}},
		{"using Amazon Bedrock for inference", []string{"AWS Bedrock"}},
		{"gcp landing zone with terraform", []string{"GCP", "Terraform"}},
		{"k8s cluster on google cloud", []string{"GCP", "Kubernetes"}},
		{"MCP server over stdio", []string{"MCP"}},
		{"chat via ollama llama", []string{"Ollama", "Llama"}},
		{"write golang microservices", []string{"Go"}},
		{"postgres and redis for cache", []string{"PostgreSQL", "Redis"}},
		{"uses langchain + crewai agents", []string{"LangChain", "CrewAI"}},
		{"Kafka streaming pipeline", []string{"Kafka"}},
	}
	for _, tt := range tests {
		entities := c.ExtractEntities(tt.content)
		for _, w := range tt.want {
			if !contains(entities.Tech, w) {
				t.Errorf("ExtractEntities(%q) missing %q; got %v", tt.content, w, entities.Tech)
			}
		}
	}
}

// TestExtractEntitiesWordBoundary ensures short aliases do not over-match
// inside larger words (e.g. "go" inside "google", "sql" inside "sqlite" is
// actually fine because SQLite has its own alias — but "go" must not match
// "google" or "golang" twice).
func TestExtractEntitiesWordBoundary(t *testing.T) {
	c := NewClassifier()
	entities := c.ExtractEntities("google cloud uses Go for backend")
	if contains(entities.Tech, "Go") != true { t.Errorf("expected Go, got %v", entities.Tech) }
	// "go" must NOT match inside "google"
	// (google is matched as GCP via alias, Go is separate)
	count := 0
	for _, tech := range entities.Tech { if tech == "Go" { count++ } }
	if count != 1 { t.Errorf("expected exactly one Go, got %d in %v", count, entities.Tech) }
	// case-insensitive: lowercase "go" still matches Go but not google twice
	entities2 := c.ExtractEntities("go is installed")
	if !contains(entities2.Tech, "Go") { t.Errorf("expected Go from lowercase 'go', got %v", entities2.Tech) }
}

func contains(list []string, s string) bool {
	for _, v := range list { if v == s { return true } }
	return false
}

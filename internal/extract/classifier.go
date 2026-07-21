// Package extract provides pattern-based entity extraction and memory type classification.
package extract

import (
	"regexp"
	"strings"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

type Entities struct {
	People    []string `json:"people,omitempty"`
	Tech      []string `json:"tech,omitempty"`
	URLs      []string `json:"urls,omitempty"`
	Dates     []string `json:"dates,omitempty"`
	Locations []string `json:"locations,omitempty"`
	Numbers   []string `json:"numbers,omitempty"`
}

type Classifier struct {
	entityPatterns []entityPattern
	typePatterns   []typePattern
}

type entityPattern struct{ category string; pattern *regexp.Regexp }
type typePattern struct{ memType types.MemoryType; keywords []string; weight int }

func NewClassifier() *Classifier {
	c := &Classifier{}
	c.registerEntityPatterns()
	c.registerTypePatterns()
	return c
}

func (c *Classifier) registerEntityPatterns() {
	for _, p := range []struct{ category, regex string }{
		{"URL", `https?://[A-Za-z0-9./?=#_%-]+`},
		{"URL", `[A-Za-z0-9.-]+\.(com|org|net|io|ai|dev)(/[A-Za-z0-9./?=#_%-]*)?`},
		{"Email", `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`},
		{"Date", `\d{4}-\d{2}-\d{2}`},
		{"Date", `(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}`},
		{"IP", `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`},
		{"Version", `v?\d+\.\d+\.\d+`},
	} {
		c.entityPatterns = append(c.entityPatterns, entityPattern{category: p.category, pattern: regexp.MustCompile(p.regex)})
	}
}

func (c *Classifier) registerTypePatterns() {
	c.typePatterns = []typePattern{
		{types.TypeDecision, []string{"decided", "decision", "chose", "agreed", "approved", "selected", "rejected", "signed off", "go with"}, 3},
		{types.TypeInsight, []string{"realized", "insight", "discovered", "found that", "noticed", "figured out", "breakthrough"}, 3},
		{types.TypeProcedure, []string{"step", "how to", "guide", "procedure", "workflow", "process", "steps", "setup", "configure", "deploy"}, 3},
		{types.TypeFact, []string{"always", "never", "is", "are", "was", "were", "fact", "truth"}, 1},
		{types.TypePreference, []string{"prefer", "like", "dislike", "favorite", "love", "better", "opinion"}, 2},
		{types.TypeContext, []string{"currently", "now", "today", "this week", "status", "progress", "update"}, 2},
		{types.TypeEvent, []string{"happened", "occurred", "event", "incident", "meeting", "call", "release", "launch"}, 2},
		{types.TypeReference, []string{"documentation", "docs", "reference", "manual", "spec", "link", "url", "http"}, 2},
	}
}

func (c *Classifier) ExtractEntities(content string) Entities {
	var entities Entities
	seen := make(map[string]bool)
	for _, ep := range c.entityPatterns {
		for _, m := range ep.pattern.FindAllString(content, -1) {
			if seen[m] { continue }; seen[m] = true
			switch ep.category {
			case "URL": entities.URLs = append(entities.URLs, m)
			case "Email": entities.People = append(entities.People, m)
			case "Date": entities.Dates = append(entities.Dates, m)
			case "IP": entities.Tech = append(entities.Tech, m)
			case "Version": entities.Tech = append(entities.Tech, m)
			}
		}
	}
	techs := []string{"GCP", "AWS", "Azure", "Kubernetes", "Docker", "Terraform", "Go", "Python", "Rust", "SQL", "Redis", "PostgreSQL", "SQLite", "Kafka"}
	for _, t := range techs { if strings.Contains(content, t) { entities.Tech = append(entities.Tech, t) } }
	return entities
}

func (c *Classifier) InferType(content string) types.MemoryType {
	lower := strings.ToLower(content)
	bestType := types.TypeNote; bestScore := 0
	for _, tp := range c.typePatterns {
		score := 0
		for _, kw := range tp.keywords { if strings.Contains(lower, kw) { score += tp.weight } }
		if score > bestScore { bestScore = score; bestType = tp.memType }
	}
	return bestType
}

func (c *Classifier) Process(content string) (Entities, types.MemoryType) { return c.ExtractEntities(content), c.InferType(content) }

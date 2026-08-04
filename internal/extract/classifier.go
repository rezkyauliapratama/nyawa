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
	techRegexes    []techMatch
}

type entityPattern struct{ category string; pattern *regexp.Regexp }
type typePattern struct{ memType types.MemoryType; keywords []string; weight int }

// techMatch maps a regex (word-boundary, case-insensitive) to a canonical
// entity name + category. Aliases normalize to the canonical display name
// so the graph does not fragment ("deepseek", "DeepSeek" -> "DeepSeek").
type techMatch struct {
	canonical string
	category  string
	re        *regexp.Regexp
}

func NewClassifier() *Classifier {
	c := &Classifier{}
	c.registerEntityPatterns()
	c.registerTypePatterns()
	c.registerTechPatterns()
	return c
}

func (c *Classifier) registerEntityPatterns() {
	for _, p := range []struct{ category, regex string }{
		{"URL", `https?://[A-Za-z0-9./?=#_%-]+`},
		{"URL", `[A-Za-z0-9.-]+\.(com|org|net|io|ai|dev|app|gov|edu)(/[A-Za-z0-9./?=#_%-]*)?`},
		{"Email", `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}`},
		{"Date", `\d{4}-\d{2}-\d{2}`}, {"Date", `\d{2}/\d{2}/\d{4}`},
		{"Date", `(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}`},
		{"IP", `\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}`}, {"Version", `v?\d+\.\d+\.\d+`}, {"Number", `\b\d{3,}\b`},
	} {
		c.entityPatterns = append(c.entityPatterns, entityPattern{category: p.category, pattern: regexp.MustCompile(p.regex)})
	}
}

func (c *Classifier) registerTypePatterns() {
	c.typePatterns = []typePattern{
		{types.TypeDecision, []string{"decided","decision","chose","agreed","approved","resolved","conclusion","selected","rejected","signed off","go with"}, 3},
		{types.TypeInsight, []string{"realized","insight","discovered","learned","found that","noticed","figured out","understood","revealed","breakthrough"}, 3},
		{types.TypeProcedure, []string{"step","how to","guide","procedure","workflow","process","instruction","runbook","tutorial","recipe","steps","setup","configure","deploy"}, 3},
		{types.TypeFact, []string{"always","never","is","are","was","were","has","have","known","fact","truth","established"}, 1},
		{types.TypePreference, []string{"prefer","like","dislike","favorite","hate","love","want","wish","rather","better","opinion"}, 2},
		{types.TypeContext, []string{"currently","now","today","this week","this month","status","progress","update","ongoing","in progress"}, 2},
		{types.TypeEvent, []string{"happened","occurred","took place","event","incident","meeting","call","conference","release","launch"}, 2},
		{types.TypeReference, []string{"documentation","docs","reference","manual","spec","specification","wiki","readme","guide","link","url","http"}, 2},
	}
}

// registerTechPatterns builds the tech dictionary. Each entry is matched with
// word boundaries and case-insensitively, then normalized to a canonical name.
// The canonical name is what gets stored in the graph, so aliases converge.
func (c *Classifier) registerTechPatterns() {
	// alias -> canonical name (keep canonical casing)
	dict := map[string]string{
		// cloud & infra
		"gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
		"aws": "AWS", "amazon web services": "AWS",
		"azure": "Azure", "microsoft azure": "Azure",
		"kubernetes": "Kubernetes", "k8s": "Kubernetes", "eks": "EKS", "gke": "GKE", "aks": "AKS",
		"terraform": "Terraform", "docker": "Docker", "docker compose": "Docker Compose",
		// data & storage
		"postgresql": "PostgreSQL", "postgres": "PostgreSQL",
		"mysql": "MySQL", "mongodb": "MongoDB", "redis": "Redis",
		"sqlite": "SQLite", "sql": "SQL", "nosql": "NoSQL",
		"kafka": "Kafka", "clickhouse": "ClickHouse", "elasticsearch": "Elasticsearch",
		"qdrant": "Qdrant", "chromadb": "ChromaDB", "pinecone": "Pinecone",
		"hnsw": "HNSW", "fts5": "FTS5", "vectordb": "VectorDB",
		// languages & runtimes
		"go": "Go", "golang": "Go", "python": "Python", "rust": "Rust",
		"typescript": "TypeScript", "javascript": "JavaScript", "node.js": "Node.js", "nodejs": "Node.js",
		"java": "Java", "c++": "C++", "c#": "C#", "php": "PHP", "ruby": "Ruby", "bash": "Bash", "shell": "Shell",
		// LLM & AI
		"deepseek": "DeepSeek", "bedrock": "AWS Bedrock", "amazon bedrock": "AWS Bedrock",
		"openai": "OpenAI", "anthropic": "Anthropic", "claude": "Claude", "sonnet": "Claude Sonnet",
		"gemini": "Gemini", "gpt": "GPT", "llama": "Llama", "ollama": "Ollama",
		"mcp": "MCP", "model context protocol": "MCP", "langchain": "LangChain", "langgraph": "LangGraph",
		"crewai": "CrewAI", "n8n": "n8n", "dify": "Dify", "rag": "RAG", "graphrag": "GraphRAG",
		"embeddings": "Embeddings", "embedding": "Embeddings", "tokenizer": "Tokenizer",
		// dev tools & platforms
		"github": "GitHub", "gitlab": "GitLab", "bitbucket": "Bitbucket", "jenkins": "Jenkins",
		"notion": "Notion", "slack": "Slack", "jira": "Jira", "confluence": "Confluence",
		"prometheus": "Prometheus", "grafana": "Grafana", "datadog": "Datadog",
		"openwebui": "Open WebUI", "streamlit": "Streamlit", "fastapi": "FastAPI",
	}
	for alias, canonical := range dict {
		// word-boundary, case-insensitive; skip pure "c" etc. to avoid over-match
		re := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(alias) + `\b`)
		c.techRegexes = append(c.techRegexes, techMatch{canonical: canonical, category: "tech", re: re})
	}
}

func (c *Classifier) ExtractEntities(content string) Entities {
	var entities Entities; seen := make(map[string]bool)
	for _, ep := range c.entityPatterns {
		for _, m := range ep.pattern.FindAllString(content, -1) {
			if seen[m] { continue }; seen[m] = true
			switch ep.category {
			case "URL": entities.URLs = append(entities.URLs, m)
			case "Email": entities.People = append(entities.People, m)
			case "Date": entities.Dates = append(entities.Dates, m)
			case "IP", "Version": entities.Tech = append(entities.Tech, m)
			case "Number": entities.Numbers = append(entities.Numbers, m)
			}
		}
	}
	// Tech dictionary: word-boundary + case-insensitive + alias normalization
	techSeen := make(map[string]bool)
	for _, tm := range c.techRegexes {
		if tm.re.MatchString(content) && !techSeen[tm.canonical] {
			techSeen[tm.canonical] = true
			entities.Tech = append(entities.Tech, tm.canonical)
		}
	}
	return entities
}

func (c *Classifier) InferType(content string) types.MemoryType {
	lower := strings.ToLower(content); bestType := types.TypeNote; bestScore := 0
	for _, tp := range c.typePatterns {
		score := 0
		for _, kw := range tp.keywords { if strings.Contains(lower, kw) { score += tp.weight } }
		if score > bestScore { bestScore = score; bestType = tp.memType }
	}
	return bestType
}

func (c *Classifier) Process(content string) (Entities, types.MemoryType) {
	return c.ExtractEntities(content), c.InferType(content)
}

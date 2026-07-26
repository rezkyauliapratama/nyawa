package security

import (
	"regexp"
)

type Classification string
const (Public Classification = "public"; Private Classification = "private"; Secret Classification = "secret")

type Filter struct{ patterns []secretPattern }
type secretPattern struct{ name, censor string; pattern *regexp.Regexp }

func NewFilter() *Filter {
	f := &Filter{}
	for _, p := range []struct{ name, regex, censor string }{
		{"OpenAI Key", `sk-[A-Za-z0-9]{32,}`, "sk-<REDACTED>"},
		{"Anthropic Key", `sk-ant-[A-Za-z0-9]{32,}`, "sk-ant-<REDACTED>"},
		{"AWS Access Key", `AKIA[0-9A-Z]{16}`, "AKIA<REDACTED>"},
		{"AWS Secret Key", `[A-Za-z0-9/+=]{40}`, "<REDACTED>"},
		{"Bearer Token", `Bearer [A-Za-z0-9\-_.]{20,}`, "Bearer <REDACTED>"},
		{"GitHub Token", `ghp_[A-Za-z0-9]{36}`, "ghp_<REDACTED>"},
		{"GitHub PAT", `github_pat_[A-Za-z0-9_]{80,}`, "github_pat_<REDACTED>"},
		{"JWT Token", `eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}`, "eyJ<REDACTED>"},
		{"PostgreSQL URL", `postgres(?:ql)?://[A-Za-z0-9]+:[^@]+@`, "<REDACTED>"},
		{"RSA Private Key", `-----BEGIN RSA PRIVATE KEY-----`, "<RSA PRIVATE KEY REDACTED>"},
		{"EC Private Key", `-----BEGIN EC PRIVATE KEY-----`, "<EC PRIVATE KEY REDACTED>"},
		{"ENV Variable", `(=|\s)([A-Za-z_]+_(KEY|SECRET|TOKEN|PASS|PASSWORD|APIKEY))=[^\s]+`, "$1$2=<REDACTED>"},
	} {
		f.patterns = append(f.patterns, secretPattern{name: p.name, pattern: regexp.MustCompile(p.regex), censor: p.censor})
	}
	return f
}

func (f *Filter) Classify(content string) (Classification, string) {
	for _, sp := range f.patterns { if sp.pattern.MatchString(content) { return Secret, sp.name } }
	return Public, ""
}

func (f *Filter) Censor(content string) string {
	result := content
	for _, sp := range f.patterns { result = sp.pattern.ReplaceAllString(result, sp.censor) }
	return result
}

func CensorAll(content string) string { return NewFilter().Censor(content) }

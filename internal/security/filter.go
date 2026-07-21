package security

import (
	"regexp"
	"strings"
)

type Classification string

const (
	Public  Classification = "public"
	Private Classification = "private"
	Secret  Classification = "secret"
)

type Filter struct {
	patterns []secretPattern
}

type secretPattern struct {
	name    string
	pattern *regexp.Regexp
	censor  string
}

func NewFilter() *Filter {
	f := &Filter{}
	f.registerDefaults()
	return f
}

func (f *Filter) registerDefaults() {
	patterns := []struct {
		name, regex, censor string
	}{
		{"OpenAI API Key", `sk-[A-Za-z0-9]{32,}`, "sk-<REDACTED>"},
		{"OpenAI Project Key", `sk-proj-[A-Za-z0-9]{32,}`, "sk-proj-<REDACTED>"},
		{"DeepSeek API Key", `sk-[A-Za-z0-9]{32,}`, "sk-<REDACTED>"},
		{"Anthropic API Key", `sk-ant-[A-Za-z0-9]{32,}`, "sk-ant-<REDACTED>"},
		{"AWS Access Key", `AKIA[0-9A-Z]{16}`, "AKIA<REDACTED>"},
		{"AWS Secret Key", `[A-Za-z0-9/+=]{40}`, "<REDACTED>"},
		{"GCP Service Account", `[A-Za-z0-9_-]+\.json`, "<REDACTED>.json"},
		{"Bearer Token", `Bearer [A-Za-z0-9\-_.]{20,}`, "Bearer <REDACTED>"},
		{"GitHub Token", `ghp_[A-Za-z0-9]{36}`, "ghp_<REDACTED>"},
		{"GitHub PAT", `github_pat_[A-Za-z0-9_]{80,}`, "github_pat_<REDACTED>"},
		{"JWT Token", `eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}`, "eyJ<REDACTED>"},
		{"PostgreSQL URL", `postgres(?:ql)?://[A-Za-z0-9]+:[^@]+@`, "<REDACTED>"},
		{"MongoDB URL", `mongodb(?:\+srv)?://[A-Za-z0-9]+:[^@]+@`, "<REDACTED>"},
		{"Redis URL", `redis://[^:]+:[^@]+@`, "<REDACTED>"},
		{"RSA Private Key", `-----BEGIN RSA PRIVATE KEY-----`, "<RSA PRIVATE KEY REDACTED>"},
		{"EC Private Key", `-----BEGIN EC PRIVATE KEY-----`, "<EC PRIVATE KEY REDACTED>"},
		{"OpenSSH Private Key", `-----BEGIN OPENSSH PRIVATE KEY-----`, "<OPENSSH PRIVATE KEY REDACTED>"},
		{"ENV Secret", `(=|\s)([A-Za-z_]+_(KEY|SECRET|TOKEN|PASS|PASSWORD|APIKEY))=[^\s]+`, "$1$2=<REDACTED>"},
	}
	for _, p := range patterns {
		f.patterns = append(f.patterns, secretPattern{name: p.name, pattern: regexp.MustCompile(p.regex), censor: p.censor})
	}
}

func (f *Filter) Classify(content string) (Classification, string) {
	for _, sp := range f.patterns {
		if sp.pattern.MatchString(content) {
			return Secret, sp.name
		}
	}
	if isPrivate(content) {
		return Private, ""
	}
	return Public, ""
}

func (f *Filter) Censor(content string) string {
	result := content
	for _, sp := range f.patterns {
		result = sp.pattern.ReplaceAllString(result, sp.censor)
	}
	return result
}

func isPrivate(content string) bool {
	indicators := []string{"password:", "password=", "passphrase:", "secret:", "secret=", "my phone number", "my address"}
	lower := strings.ToLower(content)
	for _, ind := range indicators {
		if strings.Contains(lower, ind) {
			return true
		}
	}
	return false
}

func (f *Filter) AddPattern(name, regex, censor string) error {
	re, err := regexp.Compile(regex)
	if err != nil {
		return err
	}
	f.patterns = append(f.patterns, secretPattern{name: name, pattern: re, censor: censor})
	return nil
}

func CensorAll(content string) string {
	return NewFilter().Censor(content)
}

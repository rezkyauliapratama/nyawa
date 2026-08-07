package compact

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// LLM configuration via env vars (OpenAI-compatible chat completions).
//   - NYAWA_LLM_API_KEY  — required for LLM summarization
//   - NYAWA_LLM_BASE_URL — default: https://api.deepseek.com/v1
//   - NYAWA_LLM_MODEL    — default: deepseek-chat
func llmConfig() (apiKey, baseURL, model string) {
	apiKey = os.Getenv("NYAWA_LLM_API_KEY")
	baseURL = os.Getenv("NYAWA_LLM_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.deepseek.com/v1"
	}
	model = os.Getenv("NYAWA_LLM_MODEL")
	if model == "" {
		model = "deepseek-chat"
	}
	return
}

// LLMConfigured reports whether an LLM API key is available.
func LLMConfigured() bool {
	k, _, _ := llmConfig()
	return k != ""
}

// summarizePrompt builds the structured summarization prompt for one segment.
func summarizePrompt(segment []Message, focus string) string {
	var sb strings.Builder
	sb.WriteString("You are a conversation summarizer for a context-compaction system.\n")
	sb.WriteString("Summarize the conversation segment below into a compact working state.\n")
	sb.WriteString("Preserve: user intent, key decisions, files/actions, errors and fixes, pending tasks.\n")
	sb.WriteString("Be concise. Use plain text, no markdown headers.\n")
	if focus != "" {
		sb.WriteString("Focus especially on: " + focus + "\n")
	}
	sb.WriteString("--- SEGMENT START ---\n")
	for _, m := range segment {
		sb.WriteString(m.Role + ": " + m.Content + "\n")
	}
	sb.WriteString("--- SEGMENT END ---\n")
	return sb.String()
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// SummarizeSegment summarizes one segment via the configured LLM.
// Returns an error if no API key is set or the call fails.
func SummarizeSegment(segment []Message, focus string) (string, error) {
	apiKey, baseURL, model := llmConfig()
	if apiKey == "" {
		return "", fmt.Errorf("NYAWA_LLM_API_KEY not set")
	}
	body, err := json.Marshal(chatRequest{
		Model: model,
		Messages: []chatMessage{
			{Role: "system", Content: summarizePrompt(segment, focus)},
		},
		MaxTokens:   2048,
		Temperature: 0.3,
	})
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	url := strings.TrimRight(baseURL, "/") + "/chat/completions"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("llm call: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("llm status %d: %s", resp.StatusCode, truncate(string(raw), 300))
	}

	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return "", fmt.Errorf("unmarshal: %w", err)
	}
	if cr.Error != nil {
		return "", fmt.Errorf("llm error: %s", cr.Error.Message)
	}
	if len(cr.Choices) == 0 {
		return "", fmt.Errorf("llm returned no choices")
	}
	return strings.TrimSpace(cr.Choices[0].Message.Content), nil
}

// SummarizeSegments summarizes all segments, returning one summary per segment.
// Falls back to a deterministic concatenation when the LLM is not configured
// or the call fails (best-effort compaction must never hard-fail).
func SummarizeSegments(segments [][]Message, focus string) []string {
	out := make([]string, 0, len(segments))
	for _, seg := range segments {
		s, err := SummarizeSegment(seg, focus)
		if err != nil {
			// Deterministic fallback: concatenate roles+content compactly.
			var sb strings.Builder
			for _, m := range seg {
				sb.WriteString(m.Role + ": " + m.Content + "\n")
			}
			out = append(out, sb.String())
			continue
		}
		out = append(out, s)
	}
	return out
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// Package compact provides deterministic context-compaction primitives.
// All functions are pure: no I/O, no LLM, stdlib only.
package compact

import (
	"fmt"
	"strings"
)

// CompactionStats holds metrics from a compaction run.
type CompactionStats struct {
	InputMessages     int     // messages before compaction
	OutputMessages    int     // messages after compaction
	InputTokens       int     // estimated tokens in
	OutputTokens      int     // estimated tokens out
	ReductionPct      float64 // percent reduction (0-100)
	Segments          int     // number of segments produced
	PrunedToolResults int     // tool results replaced with placeholders
}

// Message represents a single chat message.
// Compatible with OpenAI/Hermes message format.
type Message struct {
	Role    string // "system", "user", "assistant", "tool"
	Content string
}

// ToOpenAI converts a single message to the OpenAI map format.
func (m Message) ToOpenAI() map[string]string {
	return map[string]string{"role": m.Role, "content": m.Content}
}

// ToOpenAIList converts a slice of messages to the OpenAI-format list.
func ToOpenAIList(msgs []Message) []map[string]string {
	out := make([]map[string]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.ToOpenAI()
	}
	return out
}

// FromOpenAI converts OpenAI-format message maps to Message structs.
func FromOpenAI(raw []map[string]interface{}) []Message {
	out := make([]Message, 0, len(raw))
	for _, r := range raw {
		role, _ := r["role"].(string)
		content, _ := r["content"].(string)
		out = append(out, Message{Role: role, Content: content})
	}
	return out
}

// EstimateTokens returns a rough token count using the len/4 heuristic.
func EstimateTokens(s string) int {
	if len(s) == 0 {
		return 0
	}
	t := len(s) / 4
	if t < 1 {
		return 1
	}
	return t
}

// TokensIn returns the total estimated token count for a slice of messages.
func TokensIn(msgs []Message) int {
	n := 0
	for _, m := range msgs {
		n += EstimateTokens(m.Role) + EstimateTokens(m.Content)
	}
	return n
}

// Segment splits messages into overlapping segments.
// The last preserveRecentN messages are excluded from segmentation
// (kept as-is for recent context preservation).
//
// Step size = segmentSize - overlap. Every message in the segmentable
// region is covered: the final segment clamps to the end of the list.
func Segment(messages []Message, segmentSize, overlap, preserveRecentN int) [][]Message {
	if segmentSize < 1 || overlap < 0 || overlap >= segmentSize {
		return nil
	}
	n := len(messages)
	if preserveRecentN >= n {
		return nil
	}
	segmentable := messages[:n-preserveRecentN]
	if len(segmentable) == 0 {
		return nil
	}

	step := segmentSize - overlap
	if step < 1 {
		step = 1
	}

	var segments [][]Message
	for start := 0; start < len(segmentable); start += step {
		end := start + segmentSize
		if end > len(segmentable) {
			end = len(segmentable)
		}
		seg := make([]Message, end-start)
		copy(seg, segmentable[start:end])
		segments = append(segments, seg)
		if end == len(segmentable) {
			break
		}
	}
	return segments
}

// PruneToolResults replaces tool-role payloads that exceed maxBytes
// with a compact placeholder. Returns the new message list and the
// number of results pruned.
//
// This is Tier-1 microcompaction (analogous to Claude Code).
func PruneToolResults(messages []Message, maxBytes int) ([]Message, int) {
	out := make([]Message, len(messages))
	pruned := 0
	for i, m := range messages {
		if strings.ToLower(m.Role) == "tool" && len(m.Content) > maxBytes {
			out[i] = Message{
				Role:    m.Role,
				Content: fmt.Sprintf("[tool result stored in nyawa:%d]", pruned),
			}
			pruned++
		} else {
			out[i] = m
		}
	}
	return out, pruned
}

// BuildSummaryBlock creates an OpenAI-compatible summary insertion block.
// It returns three messages:
//  1. system boundary marker (delimits compaction)
//  2. system summary (the actual compressed content)
//  3. user continuation instruction
func BuildSummaryBlock(summaryText, sourceSession string) []Message {
	return []Message{
		{
			Role:    "system",
			Content: fmt.Sprintf("--- CONTEXT COMPACTION BOUNDARY ---\nSession: %s", sourceSession),
		},
		{
			Role:    "system",
			Content: summaryText,
		},
		{
			Role:    "user",
			Content: "Continue without re-asking. The context above has been compacted.",
		},
	}
}

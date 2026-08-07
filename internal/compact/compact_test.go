package compact

import (
	"strings"
	"testing"
)

func makeMsgs(n int, prefix string) []Message {
	msgs := make([]Message, n)
	for i := 0; i < n; i++ {
		msgs[i] = Message{
			Role:    "user",
			Content: prefix + " message " + string(rune('0'+i%10)),
		}
	}
	return msgs
}

func TestSegment_Basic(t *testing.T) {
	msgs := makeMsgs(10, "test")
	segments := Segment(msgs, 4, 1, 0)
	// step = 4-1 = 3 -> [0-3], [3-6], [6-9] = 3 segments
	if len(segments) != 3 {
		t.Fatalf("expected 3 segments, got %d", len(segments))
	}
	for i, seg := range segments {
		if len(seg) != 4 {
			t.Errorf("segment %d: expected len 4, got %d", i, len(seg))
		}
	}
}

func TestSegment_WithTail(t *testing.T) {
	msgs := makeMsgs(10, "test")
	// preserveRecentN=2 -> only first 8 are segmentable
	// size=4, overlap=2 -> step=2: [0-3], [2-5], [4-7] = 3 segments
	segments := Segment(msgs, 4, 2, 2)
	if len(segments) != 3 {
		t.Fatalf("expected 3 segments (8 msgs, size=4, overlap=2), got %d", len(segments))
	}
	for i, seg := range segments {
		if len(seg) != 4 {
			t.Errorf("segment %d: expected len 4, got %d", i, len(seg))
		}
	}
}

func TestSegment_CoversAll(t *testing.T) {
	// size=4, overlap=0 -> step=4: [0-3], [4-7], [8-9] = 3 segments, all covered
	msgs := makeMsgs(10, "test")
	segments := Segment(msgs, 4, 0, 0)
	if len(segments) != 3 {
		t.Fatalf("expected 3 segments, got %d", len(segments))
	}
	if len(segments[2]) != 2 {
		t.Errorf("final partial segment: expected len 2, got %d", len(segments[2]))
	}
	// verify no message lost: count total
	total := 0
	for _, seg := range segments {
		total += len(seg)
	}
	if total < 10 {
		t.Errorf("messages lost: total %d < 10", total)
	}
}

func TestSegment_TailOnly(t *testing.T) {
	msgs := makeMsgs(5, "test")
	segments := Segment(msgs, 3, 1, 5) // preserve all
	if segments != nil {
		t.Errorf("expected nil when preserveRecentN covers all, got %v", segments)
	}
}

func TestSegment_TooSmall(t *testing.T) {
	msgs := makeMsgs(3, "test")
	segments := Segment(msgs, 5, 1, 0) // segmentSize > len -> single partial segment
	if len(segments) != 1 {
		t.Fatalf("expected 1 partial segment, got %d", len(segments))
	}
	if len(segments[0]) != 3 {
		t.Errorf("partial segment: expected len 3, got %d", len(segments[0]))
	}
}

func TestSegment_InvalidParams(t *testing.T) {
	msgs := makeMsgs(10, "test")
	if seg := Segment(msgs, 0, 0, 0); seg != nil {
		t.Error("expected nil for segmentSize < 1")
	}
	if seg := Segment(msgs, 4, 4, 0); seg != nil {
		t.Error("expected nil for overlap >= segmentSize")
	}
	if seg := Segment(msgs, 4, -1, 0); seg != nil {
		t.Error("expected nil for negative overlap")
	}
}

func TestPruneToolResults_Large(t *testing.T) {
	bigPayload := strings.Repeat("x", 2000)
	msgs := []Message{
		{Role: "user", Content: "hello"},
		{Role: "tool", Content: bigPayload},
		{Role: "tool", Content: "small"},
		{Role: "assistant", Content: "ok"},
	}
	out, pruned := PruneToolResults(msgs, 500)
	if pruned != 1 {
		t.Fatalf("expected 1 pruned, got %d", pruned)
	}
	if out[0].Content != "hello" {
		t.Error("non-tool message should be unchanged")
	}
	if !strings.Contains(out[1].Content, "[tool result stored in nyawa:") {
		t.Errorf("large tool message not replaced, got: %s", out[1].Content)
	}
	if out[2].Content != "small" {
		t.Error("small tool message should be unchanged")
	}
	if out[3].Content != "ok" {
		t.Error("assistant message should be unchanged")
	}
}

func TestPruneToolResults_None(t *testing.T) {
	msgs := []Message{
		{Role: "tool", Content: "short"},
		{Role: "user", Content: "hi"},
	}
	out, pruned := PruneToolResults(msgs, 1000)
	if pruned != 0 {
		t.Fatalf("expected 0 pruned, got %d", pruned)
	}
	if out[0].Content != "short" {
		t.Error("short tool message should be unchanged")
	}
}

func TestPruneToolResults_CaseInsensitive(t *testing.T) {
	bigPayload := strings.Repeat("y", 2000)
	msgs := []Message{
		{Role: "TOOL", Content: bigPayload},
		{Role: "Tool", Content: bigPayload},
	}
	out, pruned := PruneToolResults(msgs, 500)
	if pruned != 2 {
		t.Fatalf("expected 2 pruned (case-insensitive), got %d", pruned)
	}
	_ = out
}

func TestBuildSummaryBlock(t *testing.T) {
	block := BuildSummaryBlock("summary text here", "sess-123")
	if len(block) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(block))
	}
	if block[0].Role != "system" {
		t.Errorf("msg 0: expected system, got %s", block[0].Role)
	}
	if !strings.Contains(block[0].Content, "CONTEXT COMPACTION BOUNDARY") {
		t.Error("msg 0: missing boundary marker")
	}
	if !strings.Contains(block[0].Content, "sess-123") {
		t.Error("msg 0: missing session")
	}
	if block[1].Role != "system" || block[1].Content != "summary text here" {
		t.Error("msg 1: unexpected content")
	}
	if block[2].Role != "user" {
		t.Errorf("msg 2: expected user, got %s", block[2].Role)
	}
	if !strings.Contains(block[2].Content, "Continue without re-asking") {
		t.Error("msg 2: missing continuation instruction")
	}
}

func TestEstimateTokens(t *testing.T) {
	if n := EstimateTokens(""); n != 0 {
		t.Errorf("empty string: expected 0, got %d", n)
	}
	if n := EstimateTokens("abcd"); n != 1 {
		t.Errorf("4 chars: expected 1, got %d", n)
	}
	if n := EstimateTokens("abcdefgh"); n != 2 {
		t.Errorf("8 chars: expected 2, got %d", n)
	}
	if n := EstimateTokens("abc"); n != 1 {
		t.Errorf("3 chars (<4, should floor to 1): got %d", n)
	}
}

func TestTokensIn(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "hello world"},
	}
	n := TokensIn(msgs)
	// role "system" = 6 bytes -> 6/4=1 token; content "hello world" = 11 bytes -> 11/4=2 tokens
	if n != 3 {
		t.Errorf("expected 3 tokens, got %d", n)
	}
}

func TestToOpenAI(t *testing.T) {
	m := Message{Role: "user", Content: "hello"}
	oai := m.ToOpenAI()
	if oai["role"] != "user" || oai["content"] != "hello" {
		t.Errorf("unexpected map: %v", oai)
	}
}

func TestToOpenAIList(t *testing.T) {
	msgs := []Message{
		{Role: "system", Content: "a"},
		{Role: "user", Content: "b"},
	}
	list := ToOpenAIList(msgs)
	if len(list) != 2 {
		t.Fatalf("expected 2, got %d", len(list))
	}
	if list[0]["role"] != "system" || list[0]["content"] != "a" {
		t.Error("msg 0 mismatch")
	}
}

func TestFromOpenAI(t *testing.T) {
	raw := []map[string]interface{}{
		{"role": "user", "content": "hello"},
		{"role": "assistant", "content": "hi there"},
	}
	msgs := FromOpenAI(raw)
	if len(msgs) != 2 {
		t.Fatalf("expected 2, got %d", len(msgs))
	}
	if msgs[0].Role != "user" || msgs[0].Content != "hello" {
		t.Error("msg 0 mismatch")
	}
	if msgs[1].Role != "assistant" || msgs[1].Content != "hi there" {
		t.Error("msg 1 mismatch")
	}
}

func TestCompactionStats_ZeroValue(t *testing.T) {
	var s CompactionStats
	if s.InputMessages != 0 || s.OutputMessages != 0 {
		t.Error("zero-value stats should have all zeros")
	}
}

package mcp

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestRecallAtIsDeclaredAndBound keeps the schema and the arg struct in step:
// a property nobody binds, or a field the schema never declares, is unreachable
// for every client.
func TestRecallAtIsDeclaredAndBound(t *testing.T) {
	raw := json.RawMessage(`{"query":"q","at":"2026-11-01T00:00:00Z"}`)
	var args recallArgs
	if err := json.Unmarshal(raw, &args); err != nil {
		t.Fatalf("unmarshal recall args: %v", err)
	}
	if args.At != "2026-11-01T00:00:00Z" {
		t.Fatalf("at not bound: %+v", args)
	}

	s := &Server{}
	var found bool
	for _, tool := range s.tools() {
		if tool.Name != "nyawa_recall" {
			continue
		}
		prop, ok := tool.InputSchema.Properties["at"]
		if !ok {
			t.Fatal("nyawa_recall schema does not declare at")
		}
		if prop.Type != "string" {
			t.Fatalf("at schema type = %q, want string", prop.Type)
		}
		found = true
	}
	if !found {
		t.Fatal("nyawa_recall tool missing from tools()")
	}
}

// TestHandleRecallRejectsUnparsableAt proves a bad timestamp fails loudly. The
// alternative, ignoring it, answers a question about the past with present-day
// memories and gives the caller no way to notice.
func TestHandleRecallRejectsUnparsableAt(t *testing.T) {
	s, buf := newRecallTestServer(t)
	s.handleRecall(1, json.RawMessage(`{"query":"q","at":"last tuesday"}`))

	var resp struct {
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(buf.Bytes(), &resp); err != nil {
		t.Fatalf("decode response %q: %v", buf.String(), err)
	}
	if resp.Error == nil {
		t.Fatalf("expected an error for an unparsable at, got %s", buf.String())
	}
	if resp.Error.Code != -32602 {
		t.Errorf("error code = %d, want -32602", resp.Error.Code)
	}
	if !strings.Contains(resp.Error.Message, "RFC3339") {
		t.Errorf("error message %q does not explain the expected format", resp.Error.Message)
	}
}

// TestHandleRecallAtFiltersByCreationTime is the behavioural proof: the same
// query returns nothing for an instant before the memories existed, and the
// full set for an instant after.
func TestHandleRecallAtFiltersByCreationTime(t *testing.T) {
	past := time.Now().Add(-1 * time.Hour).UTC().Format(time.RFC3339)
	future := time.Now().Add(1 * time.Hour).UTC().Format(time.RFC3339)

	s, buf := newRecallTestServer(t)
	s.handleRecall(1, json.RawMessage(`{"query":"q","at":"`+past+`"}`))
	if count, types := recallPayload(t, buf); count != 0 {
		t.Fatalf("at in the past returned %d results (%v), want 0", count, types)
	}

	s, buf = newRecallTestServer(t)
	s.handleRecall(1, json.RawMessage(`{"query":"q","at":"`+future+`"}`))
	if count, _ := recallPayload(t, buf); count != 3 {
		t.Fatalf("at in the future returned %d results, want 3", count)
	}

	s, buf = newRecallTestServer(t)
	s.handleRecall(1, json.RawMessage(`{"query":"q"}`))
	if count, _ := recallPayload(t, buf); count != 3 {
		t.Fatalf("unfiltered recall returned %d results, want 3", count)
	}
}

// TestRecallAtFutureMatchesNoFilter guards against a filter that is wired to the
// wrong field: an instant after every memory exists must not change the answer.
func TestRecallAtFutureMatchesNoFilter(t *testing.T) {
	future := time.Now().Add(24 * time.Hour).UTC().Format(time.RFC3339)

	s, buf := newRecallTestServer(t)
	s.handleRecall(1, json.RawMessage(`{"query":"q"}`))
	plain, _ := recallPayload(t, buf)

	s2, buf2 := newRecallTestServer(t)
	s2.handleRecall(1, json.RawMessage(`{"query":"q","at":"`+future+`"}`))
	filtered, _ := recallPayload(t, buf2)

	if plain != filtered {
		t.Fatalf("count differs: plain=%d future-at=%d", plain, filtered)
	}
	if !bytes.Equal(buf.Bytes(), buf2.Bytes()) && plain == 0 {
		t.Fatal("both empty, nothing proven")
	}
}

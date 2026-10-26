package compactioneval

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAssembleJSONLDeterministicAndPayloadFree(t *testing.T) {
	env := map[string]any{"actor_revision": "actor", "fingerprint_policy_id": DefaultFingerprintPolicyID, "nyawa_revision": "nyawa", "summary_mode": "captured"}
	base := func(recordType string, index int) map[string]any {
		return map[string]any{"schema_version": TraceSchemaVersion, "record_type": recordType, "record_schema": "collector-event-v1", "suite_id": "suite", "task_id": "task", "run_id": "run", "condition": ConditionSummaryOnly, "environment": env, "session_id": "s", "event_index": index, "timestamp": "2026-01-01T00:00:00Z"}
	}
	prepare := base("compaction_prepare", 1)
	prepare["messages_sha256"] = "sha256:" + strings.Repeat("a", 64)
	boundary := base("compaction", 2)
	boundary["checkpoint_id"] = "legacy-1"
	boundary["completed"] = true
	boundary["source_sha256"] = "sha256:" + strings.Repeat("a", 64)
	boundary["summary_sha256"] = "sha256:" + strings.Repeat("b", 64)
	boundary["before_session_id"] = "old"
	boundary["after_session_id"] = "new"
	event := base("tool_event", 3)
	event["tool_name"] = "terminal"
	event["output_sha256"] = "sha256:" + strings.Repeat("c", 64)
	event["outcome"] = "success"
	event["is_planning"] = false
	event["is_polling"] = false
	event["checkpoint_id"] = "legacy-1"
	event["secret_payload"] = "must not enter trace"
	one, _ := json.Marshal(event)
	two, _ := json.Marshal(prepare)
	three, _ := json.Marshal(boundary)
	first, err := AssembleJSONL(append(append(append([]byte{}, one...), '\n'), append(append(two, '\n'), three...)...))
	if err != nil {
		t.Fatal(err)
	}
	second, err := AssembleJSONL(append(append(append([]byte{}, three...), '\n'), append(append(two, '\n'), one...)...))
	if err != nil {
		t.Fatal(err)
	}
	left, _ := json.Marshal(first)
	right, _ := json.Marshal(second)
	if string(left) != string(right) {
		t.Fatalf("assembly changed with input order\n%s\n%s", left, right)
	}
	if len(first.ToolEvents) != 1 || first.Compactions[0].EventIndex != 3 {
		t.Fatalf("assembled trace=%+v", first)
	}
	if strings.Contains(string(left), "secret_payload") {
		t.Fatal("unknown/raw payload leaked into canonical trace")
	}
}

func TestAssembleRejectsMissingSummaryHash(t *testing.T) {
	data := `{"schema_version":"` + TraceSchemaVersion + `","record_type":"compaction","record_schema":"collector-event-v1","suite_id":"s","task_id":"t","run_id":"r","condition":"summary_only","environment":{"nyawa_revision":"n","actor_revision":"a","summary_mode":"captured","fingerprint_policy_id":"` + DefaultFingerprintPolicyID + `"},"session_id":"s","event_index":1,"timestamp":"x","checkpoint_id":"legacy","completed":true,"source_sha256":"sha256:` + strings.Repeat("a", 64) + `"}`
	if _, err := AssembleJSONL([]byte(data)); err == nil || !strings.Contains(err.Error(), "summary_sha256") {
		t.Fatalf("error=%v", err)
	}
}

func TestResidualPackBoundsIdempotencyAndRedaction(t *testing.T) {
	policy := DefaultResidualPolicy()
	policy.MaxItems = 1
	policy.MaxExcerptBytes = 5
	policy.MaxPackBytes = 4096
	input := ResidualPackInput{SessionKey: "s", OldSessionID: "old", NewSessionID: "new", Summary: "summary", CompactedThroughMessageIndex: 2, MetadataCoverage: "complete", Policy: policy, Messages: []ResidualSourceMessage{{Index: 0, Role: "tool", Content: "Bearer topsecret abcdef", Omitted: true}, {Index: 1, Role: "tool", Content: "later", Omitted: true}}, ToolEvents: []ResidualToolEvent{{EventIndex: 2, SourceMessageIndex: 0, ToolName: "terminal", Outcome: OutcomeFailure}, {EventIndex: 3, SourceMessageIndex: 1, ToolName: "terminal", Outcome: OutcomeSuccess}}}
	first, err := BuildResidualPack(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := BuildResidualPack(input)
	if err != nil {
		t.Fatal(err)
	}
	if string(first.CanonicalJSON) != string(second.CanonicalJSON) || first.PackSHA256 != second.PackSHA256 {
		t.Fatal("residual pack is not idempotent")
	}
	if len(first.Pack.Items) > 1 || len(first.CanonicalJSON) > policy.MaxPackBytes {
		t.Fatalf("bounds exceeded: %+v bytes=%d", first.Pack, len(first.CanonicalJSON))
	}
	if strings.Contains(string(first.CanonicalJSON), "topsecret") {
		t.Fatal("secret leaked")
	}
	if !strings.HasPrefix(first.CheckpointID, "ccp_v1_") || !strings.HasPrefix(first.Pack.Items[0].EntryID, "rpi_v1_") {
		t.Fatal("missing stable identities")
	}
}

func TestResidualPackRejectsInvalidMetadataAndDisabledMode(t *testing.T) {
	input := ResidualPackInput{SessionKey: "s", OldSessionID: "o", NewSessionID: "n", MetadataCoverage: "complete", Messages: []ResidualSourceMessage{{Index: 0, Role: "assistant", Content: "x", Omitted: true}}, ToolEvents: []ResidualToolEvent{{EventIndex: 1, SourceMessageIndex: 0, ToolName: "terminal", Outcome: OutcomeFailure}}}
	if _, err := BuildResidualPack(input); err == nil {
		t.Fatal("non-tool metadata was accepted")
	}
	input.Messages[0].Role = "tool"
	disabled, err := BuildResidualPackWithOptions(input, ResidualBuildOptions{Enabled: false})
	if err != nil || !disabled.Disabled || disabled.CanonicalJSON != nil {
		t.Fatalf("disabled result=%+v err=%v", disabled, err)
	}
}

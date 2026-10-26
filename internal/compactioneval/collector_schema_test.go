package compactioneval

import (
	"encoding/json"
	"testing"
)

func TestCollectorMetadataFieldsAreAdditive(t *testing.T) {
	trace := traceWithEvents("collector-metadata", []ToolEvent{toolEvent(1, "nyawa_recall", "a", OutcomeSuccess)})
	trace.SessionID = "session-1"
	trace.Compactions = []Compaction{{
		CheckpointID:       "legacy-checkpoint",
		EventIndex:         1,
		SourceSHA256:       testDigest("b"),
		SummarySHA256:      testDigest("c"),
		ResidualPackSHA256: nil,
		Namespace:          "context",
		BeforeSessionID:    "session-0",
		AfterSessionID:     "session-1",
		Timestamp:          "2026-10-23T06:00:00.123Z",
	}}
	query := "redacted query"
	trace.ToolEvents[0].ToolCallID = "call-1"
	trace.ToolEvents[0].Timestamp = "2026-10-23T06:00:00.123Z"
	trace.ToolEvents[0].RecallQuery = &query
	trace.ToolEvents[0].ReturnedMemoryIDs = []string{"mem-1", "mem-2"}
	trace.ToolEvents[0].CheckpointID = testStringPointer("legacy-checkpoint")
	trace.RecallProbes = []RecallProbe{{
		ProbeID:                "probe-1",
		CheckpointID:           "legacy-checkpoint",
		Query:                  query,
		Namespace:              "context",
		Limit:                  5,
		MinScore:               0,
		ExcludeTypes:           []string{},
		TargetMemoryIDs:        []string{"mem-1"},
		TargetResidualEntryIDs: []string{},
		Surface:                RecallSurfaceNyawa,
		ResultIDs:              []string{"mem-1"},
		Timestamp:              "2026-10-23T06:00:00.123Z",
		ToolCallID:             "call-1",
	}}

	data, err := json.Marshal(trace)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	decoded, err := DecodeTraceJSON(data)
	if err != nil {
		t.Fatalf("DecodeTraceJSON() error = %v", err)
	}
	if decoded.SessionID != trace.SessionID || decoded.ToolEvents[0].ToolCallID != "call-1" {
		t.Fatalf("additive metadata did not round-trip: %+v", decoded)
	}
	if decoded.Compactions[0].BeforeSessionID != "session-0" || decoded.Compactions[0].AfterSessionID != "session-1" {
		t.Fatalf("compaction session attribution did not round-trip: %+v", decoded.Compactions[0])
	}
}

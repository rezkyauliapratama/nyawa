package compactioneval

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

type optionalRate struct {
	set   bool
	value float64
}

type scoreExpectation struct {
	repeats           int
	eligible          int
	postRepeats       int
	errors            int
	attempts          int
	postErrors        int
	nonplanningErrors int
	cancelled         int
	unknown           int
	repeatedCalls     int
	warningCount      int
	repeatRate        optionalRate
	errorRate         optionalRate
}

func TestScoreNormativeCases(t *testing.T) {
	cases := []struct {
		name  string
		trace Trace
		want  scoreExpectation
		check func(*testing.T, Result)
	}{
		{
			name: "same tool same output three times",
			trace: traceWithEvents("same-tool-output", []ToolEvent{
				toolEvent(1, "terminal", "a", OutcomeSuccess),
				toolEvent(2, "terminal", "a", OutcomeSuccess),
				toolEvent(3, "terminal", "a", OutcomeSuccess),
			}),
			want: scoreExpectation{
				repeats: 2, eligible: 3, attempts: 3, repeatedCalls: 2,
				repeatRate: optionalRate{set: true, value: 2.0 / 3.0},
				errorRate:  optionalRate{set: true, value: 0},
			},
			check: func(t *testing.T, got Result) {
				t.Helper()
				if len(got.RepeatGroups) != 1 || got.RepeatGroups[0].Count != 3 || got.RepeatGroups[0].RepeatContribution != 2 {
					t.Fatalf("repeat groups = %+v, want one group with count=3 and repeats=2", got.RepeatGroups)
				}
			},
		},
		{
			name: "different tools with same output",
			trace: traceWithEvents("different-tools", []ToolEvent{
				toolEvent(1, "terminal", "a", OutcomeSuccess),
				toolEvent(2, "browser", "a", OutcomeSuccess),
			}),
			want: scoreExpectation{
				eligible: 2, attempts: 2,
				repeatRate: optionalRate{set: true, value: 0},
				errorRate:  optionalRate{set: true, value: 0},
			},
		},
		{
			name: "polling duplicates are excluded from repeats",
			trace: func() Trace {
				first := toolEvent(1, "terminal", "a", OutcomeSuccess)
				first.IsPolling = true
				second := toolEvent(2, "terminal", "a", OutcomeSuccess)
				second.IsPolling = true
				return traceWithEvents("polling-duplicates", []ToolEvent{first, second})
			}(),
			want: scoreExpectation{
				attempts:   2,
				repeatRate: optionalRate{},
				errorRate:  optionalRate{set: true, value: 0},
			},
		},
		{
			name: "planning failure remains an error while excluded from repeats",
			trace: func() Trace {
				event := toolEvent(1, "terminal", "a", OutcomeFailure)
				event.IsPlanning = true
				return traceWithEvents("planning-failure", []ToolEvent{event})
			}(),
			want: scoreExpectation{
				errors: 1, attempts: 1,
				repeatRate: optionalRate{},
				errorRate:  optionalRate{set: true, value: 1},
			},
		},
		{
			name: "repeated failure contributes to repeats and errors",
			trace: traceWithEvents("repeated-failure", []ToolEvent{
				toolEvent(1, "terminal", "a", OutcomeFailure),
				toolEvent(2, "terminal", "a", OutcomeFailure),
			}),
			want: scoreExpectation{
				repeats: 1, eligible: 2, errors: 2, attempts: 2, nonplanningErrors: 2, repeatedCalls: 1,
				repeatRate: optionalRate{set: true, value: 0.5},
				errorRate:  optionalRate{set: true, value: 1},
			},
		},
		{
			name: "failure without an output fingerprint remains an error",
			trace: func() Trace {
				event := toolEvent(1, "terminal", "a", OutcomeFailure)
				event.OutputSHA256 = nil
				return traceWithEvents("failure-without-output", []ToolEvent{event})
			}(),
			want: scoreExpectation{
				errors: 1, attempts: 1, nonplanningErrors: 1,
				repeatRate: optionalRate{},
				errorRate:  optionalRate{set: true, value: 1},
			},
		},
		{
			name: "cancellation is an attempt but not an error",
			trace: traceWithEvents("cancellation", []ToolEvent{
				toolEvent(1, "terminal", "a", OutcomeCancelled),
			}),
			want: scoreExpectation{
				eligible: 1, attempts: 1, cancelled: 1,
				repeatRate: optionalRate{set: true, value: 0},
				errorRate:  optionalRate{set: true, value: 0},
			},
		},
		{
			name: "unknown outcome is reported but not an error",
			trace: traceWithEvents("unknown-outcome", []ToolEvent{
				toolEvent(1, "terminal", "a", OutcomeUnknown),
			}),
			want: scoreExpectation{
				eligible: 1, attempts: 1, unknown: 1, warningCount: 1,
				repeatRate: optionalRate{set: true, value: 0},
				errorRate:  optionalRate{set: true, value: 0},
			},
		},
		{
			name: "post-compaction attribution uses the most recent boundary",
			trace: func() Trace {
				trace := traceWithEvents("checkpoint-attribution", []ToolEvent{
					toolEvent(1, "terminal", "a", OutcomeSuccess),
					toolEventWithCheckpoint(11, "terminal", "a", OutcomeFailure, "legacy-A"),
					toolEventWithCheckpoint(20, "terminal", "b", OutcomeFailure, "legacy-B"),
				})
				trace.Compactions = []Compaction{
					legacyCheckpoint("legacy-A", 10),
					legacyCheckpoint("legacy-B", 20),
				}
				return trace
			}(),
			want: scoreExpectation{
				repeats: 1, eligible: 3, postRepeats: 1, errors: 2, attempts: 3, postErrors: 2, nonplanningErrors: 2, repeatedCalls: 1,
				repeatRate: optionalRate{set: true, value: 1.0 / 3.0},
				errorRate:  optionalRate{set: true, value: 2.0 / 3.0},
			},
			check: func(t *testing.T, got Result) {
				t.Helper()
				if len(got.CheckpointMetrics) != 2 {
					t.Fatalf("checkpoint metrics = %+v, want two boundaries", got.CheckpointMetrics)
				}
				first, second := got.CheckpointMetrics[0], got.CheckpointMetrics[1]
				if first.CheckpointID != "legacy-A" || first.ToolAttempts != 1 || first.RepeatedToolOutputs != 1 || first.ToolErrors != 1 {
					t.Errorf("checkpoint A = %+v, want its between-boundary event only", first)
				}
				if second.CheckpointID != "legacy-B" || second.ToolAttempts != 1 || second.RepeatedToolOutputs != 0 || second.ToolErrors != 1 {
					t.Errorf("checkpoint B = %+v, want its boundary event only", second)
				}
			},
		},
		{
			name: "current recall hit uses first target rank",
			trace: func() Trace {
				trace := traceWithCheckpoint("current-recall-hit", "legacy-A", 10)
				trace.RecallProbes = []RecallProbe{nyawaProbe("probe-hit", "legacy-A", RecallTargetActive, []string{"other", "target-memory", "later"})}
				return trace
			}(),
			want: scoreExpectation{},
			check: func(t *testing.T, got Result) {
				t.Helper()
				recall := got.RecallAfterCompaction
				if recall.ValidRecallProbes != 1 || recall.UnobservableRecallProbes != 0 || recall.SupersededTargetProbes != 0 {
					t.Fatalf("recall counts = %+v", recall)
				}
				assertOptionalRate(t, "recall@1", recall.RecallAt1, optionalRate{set: true, value: 0})
				assertOptionalRate(t, "recall@3", recall.RecallAt3, optionalRate{set: true, value: 1})
				assertOptionalRate(t, "recall@5", recall.RecallAt5, optionalRate{set: true, value: 1})
				assertOptionalRate(t, "MRR@5", recall.MRRAt5, optionalRate{set: true, value: 0.5})
			},
		},
		{
			name: "unobservable current source is excluded from recall denominator",
			trace: func() Trace {
				trace := traceWithCheckpoint("unobservable-source", "legacy-A", 10)
				trace.RecallProbes = []RecallProbe{nyawaProbe("probe-unobservable", "legacy-A", RecallTargetUnobservable, nil)}
				return trace
			}(),
			want: scoreExpectation{},
			check: func(t *testing.T, got Result) {
				t.Helper()
				recall := got.RecallAfterCompaction
				if recall.ValidRecallProbes != 0 || recall.UnobservableRecallProbes != 1 || recall.SupersededTargetProbes != 0 {
					t.Fatalf("recall counts = %+v", recall)
				}
				assertOptionalRate(t, "recall@1", recall.RecallAt1, optionalRate{})
				assertOptionalRate(t, "recall@3", recall.RecallAt3, optionalRate{})
				assertOptionalRate(t, "recall@5", recall.RecallAt5, optionalRate{})
				assertOptionalRate(t, "MRR@5", recall.MRRAt5, optionalRate{})
			},
		},
		{
			name: "superseded target is reported separately",
			trace: func() Trace {
				trace := traceWithCheckpoint("superseded-target", "legacy-A", 10)
				trace.RecallProbes = []RecallProbe{nyawaProbe("probe-superseded", "legacy-A", RecallTargetSuperseded, nil)}
				return trace
			}(),
			want: scoreExpectation{},
			check: func(t *testing.T, got Result) {
				t.Helper()
				recall := got.RecallAfterCompaction
				if recall.ValidRecallProbes != 0 || recall.UnobservableRecallProbes != 0 || recall.SupersededTargetProbes != 1 {
					t.Fatalf("recall counts = %+v", recall)
				}
				assertOptionalRate(t, "recall@1", recall.RecallAt1, optionalRate{})
			},
		},
	}

	for _, testCase := range cases {
		testCase := testCase
		t.Run(testCase.name, func(t *testing.T) {
			got, err := Score(testCase.trace)
			if err != nil {
				t.Fatalf("Score() error = %v", err)
			}
			assertScoreExpectation(t, got, testCase.want)
			if testCase.check != nil {
				testCase.check(t, got)
			}
		})
	}
}

func TestScoreTracesMacroRatesAndDeterministicOrder(t *testing.T) {
	first := traceWithEvents("task-b", []ToolEvent{
		toolEvent(1, "terminal", "a", OutcomeSuccess),
	})
	second := traceWithEvents("task-a", []ToolEvent{
		toolEvent(1, "terminal", "a", OutcomeSuccess),
		toolEvent(2, "terminal", "a", OutcomeSuccess),
	})

	got, err := ScoreTraces([]Trace{first, second})
	if err != nil {
		t.Fatalf("ScoreTraces() error = %v", err)
	}
	if got.Runs != 2 || got.RepeatedToolOutputs != 1 || got.EligibleToolOutputs != 3 {
		t.Fatalf("aggregate counts = %+v", got)
	}
	assertOptionalRate(t, "repeat micro", got.RepeatedToolOutputRateMicro, optionalRate{set: true, value: 1.0 / 3.0})
	assertOptionalRate(t, "repeat macro", got.RepeatedToolOutputRateMacro, optionalRate{set: true, value: 0.25})
	if got.RepeatedToolOutputRateMacroContributingRuns != 2 {
		t.Errorf("repeat macro contributing runs = %d, want 2", got.RepeatedToolOutputRateMacroContributingRuns)
	}
	if len(got.RepeatGroups) != 2 || got.RepeatGroups[0].TaskID != "task-a" || got.RepeatGroups[1].TaskID != "task-b" {
		t.Errorf("repeat groups are not deterministically ordered: %+v", got.RepeatGroups)
	}

	forwardJSON, err := MarshalResultJSON(got)
	if err != nil {
		t.Fatalf("MarshalResultJSON(forward): %v", err)
	}
	reversed, err := ScoreTraces([]Trace{second, first})
	if err != nil {
		t.Fatalf("ScoreTraces(reversed): %v", err)
	}
	reversedJSON, err := MarshalResultJSON(reversed)
	if err != nil {
		t.Fatalf("MarshalResultJSON(reversed): %v", err)
	}
	if !bytes.Equal(forwardJSON, reversedJSON) {
		t.Errorf("aggregate JSON differs by input order:\nforward=%s\nreversed=%s", forwardJSON, reversedJSON)
	}
}

func TestScoreDoesNotMutateTraceAndProducesDeterministicJSON(t *testing.T) {
	trace := traceWithCheckpoint("deterministic", "legacy-A", 10)
	trace.ToolEvents = []ToolEvent{
		toolEvent(1, "terminal", "a", OutcomeSuccess),
		toolEventWithCheckpoint(11, "terminal", "a", OutcomeFailure, "legacy-A"),
	}
	trace.RecallProbes = []RecallProbe{nyawaProbe("probe", "legacy-A", RecallTargetActive, []string{"target-memory"})}

	before, err := json.Marshal(trace)
	if err != nil {
		t.Fatalf("marshal trace before scoring: %v", err)
	}
	first, err := Score(trace)
	if err != nil {
		t.Fatalf("first Score(): %v", err)
	}
	second, err := Score(trace)
	if err != nil {
		t.Fatalf("second Score(): %v", err)
	}
	after, err := json.Marshal(trace)
	if err != nil {
		t.Fatalf("marshal trace after scoring: %v", err)
	}
	if !bytes.Equal(before, after) {
		t.Error("Score() mutated its trace input")
	}
	firstJSON, err := MarshalResultJSON(first)
	if err != nil {
		t.Fatalf("marshal first result: %v", err)
	}
	secondJSON, err := MarshalResultJSON(second)
	if err != nil {
		t.Fatalf("marshal second result: %v", err)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Errorf("scoring the same trace produced different JSON:\nfirst=%s\nsecond=%s", firstJSON, secondJSON)
	}
}

func TestResidualPackLookupRecallIsReportedSeparately(t *testing.T) {
	trace := baseTrace("residual-pack-recall")
	trace.Condition = ConditionSummaryPlusResidualPack
	checkpointID := "ccp_v1_" + strings.Repeat("a", 64)
	trace.Compactions = []Compaction{{
		CheckpointID:       checkpointID,
		EventIndex:         10,
		SourceSHA256:       testDigest("b"),
		SummarySHA256:      testDigest("c"),
		ResidualPackSHA256: testStringPointer(testDigest("d")),
		Namespace:          "context",
	}}
	trace.RecallProbes = []RecallProbe{{
		ProbeID:                "residual-probe",
		CheckpointID:           checkpointID,
		Query:                  "fixture query",
		Namespace:              "context",
		Limit:                  5,
		MinScore:               0,
		ExcludeTypes:           []string{},
		TargetMemoryIDs:        []string{},
		TargetResidualEntryIDs: []string{"entry-1"},
		Surface:                RecallSurfaceResidualPack,
		TargetStatus:           RecallTargetActive,
		ResultIDs:              []string{"entry-1"},
	}}

	got, err := Score(trace)
	if err != nil {
		t.Fatalf("Score() error = %v", err)
	}
	if got.RecallAfterCompaction.ValidRecallProbes != 0 {
		t.Errorf("nyawa recall probes = %d, want 0", got.RecallAfterCompaction.ValidRecallProbes)
	}
	residual := got.ResidualPackLookupAfterCompaction
	if residual.ValidRecallProbes != 1 {
		t.Fatalf("residual valid probes = %d, want 1", residual.ValidRecallProbes)
	}
	assertOptionalRate(t, "residual recall@1", residual.RecallAt1, optionalRate{set: true, value: 1})
	assertOptionalRate(t, "residual recall@3", residual.RecallAt3, optionalRate{set: true, value: 1})
	assertOptionalRate(t, "residual recall@5", residual.RecallAt5, optionalRate{set: true, value: 1})
	assertOptionalRate(t, "residual MRR@5", residual.MRRAt5, optionalRate{set: true, value: 1})
}

func TestRecallMetricsRespectRecordedLimit(t *testing.T) {
	trace := traceWithCheckpoint("limited-recall", "legacy-A", 10)
	probe := nyawaProbe("limited-probe", "legacy-A", RecallTargetActive, []string{"target-memory"})
	probe.Limit = 1
	trace.RecallProbes = []RecallProbe{probe}

	got, err := Score(trace)
	if err != nil {
		t.Fatalf("Score() error = %v", err)
	}
	recall := got.RecallAfterCompaction
	if recall.ValidRecallProbes != 1 || recall.RecallAt1ContributingProbes != 1 {
		t.Fatalf("limited recall counts = %+v", recall)
	}
	assertOptionalRate(t, "limited recall@1", recall.RecallAt1, optionalRate{set: true, value: 1})
	assertOptionalRate(t, "limited recall@3", recall.RecallAt3, optionalRate{})
	assertOptionalRate(t, "limited recall@5", recall.RecallAt5, optionalRate{})
	assertOptionalRate(t, "limited MRR@5", recall.MRRAt5, optionalRate{})
}

func TestValidateTraceRejectsNormativeViolations(t *testing.T) {
	cases := []struct {
		name  string
		trace Trace
	}{
		{
			name: "tool event indexes must increase",
			trace: traceWithEvents("bad-event-order", []ToolEvent{
				toolEvent(2, "terminal", "a", OutcomeSuccess),
				toolEvent(1, "terminal", "b", OutcomeSuccess),
			}),
		},
		{
			name: "compaction boundary indexes must increase",
			trace: func() Trace {
				trace := traceWithEvents("bad-boundary-order", nil)
				trace.Compactions = []Compaction{legacyCheckpoint("legacy-A", 10), legacyCheckpoint("legacy-B", 10)}
				return trace
			}(),
		},
		{
			name: "checkpoint attribution must use the most recent boundary",
			trace: func() Trace {
				trace := traceWithEvents("stale-checkpoint", []ToolEvent{
					toolEventWithCheckpoint(15, "terminal", "a", OutcomeSuccess, "legacy-B"),
				})
				trace.Compactions = []Compaction{legacyCheckpoint("legacy-A", 10), legacyCheckpoint("legacy-B", 20)}
				return trace
			}(),
		},
		{
			name: "outcome must use the declared enum",
			trace: traceWithEvents("bad-outcome", []ToolEvent{
				toolEvent(1, "terminal", "a", Outcome("provider_error")),
			}),
		},
		{
			name: "all provided output digests must be strict SHA-256 values",
			trace: func() Trace {
				trace := traceWithEvents("bad-output-digest", []ToolEvent{toolEvent(1, "terminal", "a", OutcomeSuccess)})
				trace.ToolEvents[0].OutputSHA256 = testStringPointer("sha256:not-a-digest")
				return trace
			}(),
		},
		{
			name: "nyawa recall needs a memory target",
			trace: func() Trace {
				trace := traceWithCheckpoint("missing-memory-target", "legacy-A", 10)
				probe := nyawaProbe("probe", "legacy-A", RecallTargetActive, nil)
				probe.TargetMemoryIDs = []string{}
				trace.RecallProbes = []RecallProbe{probe}
				return trace
			}(),
		},
		{
			name: "no compaction condition cannot declare a completed boundary",
			trace: func() Trace {
				trace := traceWithCheckpoint("no-compaction-boundary", "legacy-A", 10)
				trace.Condition = "no_compaction"
				return trace
			}(),
		},
		{
			name: "summary plus residual pack requires canonical checkpoint and pack digest",
			trace: func() Trace {
				trace := traceWithCheckpoint("missing-residual-pack", "legacy-A", 10)
				trace.Condition = "summary_plus_residual_pack"
				return trace
			}(),
		},
	}

	for _, testCase := range cases {
		testCase := testCase
		t.Run(testCase.name, func(t *testing.T) {
			if err := ValidateTrace(testCase.trace); err == nil {
				t.Fatal("ValidateTrace() error = nil, want validation failure")
			}
		})
	}
}

func TestDecodeTraceJSONIsStrict(t *testing.T) {
	validTrace := traceWithEvents("json-strict", []ToolEvent{toolEvent(1, "terminal", "a", OutcomeSuccess)})
	validJSON, err := json.Marshal(validTrace)
	if err != nil {
		t.Fatalf("marshal valid trace: %v", err)
	}

	unknownField := mutateJSON(t, validJSON, func(root map[string]any) {
		root["unexpected"] = true
	})
	missingRequiredToolField := mutateJSON(t, validJSON, func(root map[string]any) {
		events := root["tool_events"].([]any)
		event := events[0].(map[string]any)
		delete(event, "is_polling")
	})
	nullRequiredTopLevel := mutateJSON(t, validJSON, func(root map[string]any) {
		root["condition"] = nil
	})

	cases := []struct {
		name string
		data []byte
		want bool
	}{
		{name: "valid", data: validJSON, want: true},
		{name: "unknown field", data: unknownField},
		{name: "missing required tool field", data: missingRequiredToolField},
		{name: "null required top-level field", data: nullRequiredTopLevel},
		{name: "trailing JSON", data: append(append([]byte{}, validJSON...), []byte("\n{}")...)},
	}

	for _, testCase := range cases {
		testCase := testCase
		t.Run(testCase.name, func(t *testing.T) {
			_, err := DecodeTraceJSON(testCase.data)
			if testCase.want && err != nil {
				t.Fatalf("DecodeTraceJSON() error = %v", err)
			}
			if !testCase.want && err == nil {
				t.Fatal("DecodeTraceJSON() error = nil, want strict decoding failure")
			}
		})
	}
}

func TestScoreTracesRejectsDuplicateRunIdentity(t *testing.T) {
	trace := traceWithEvents("duplicate-task", []ToolEvent{toolEvent(1, "terminal", "a", OutcomeSuccess)})
	duplicate := trace
	duplicate.SuiteID = "another-suite"
	if _, err := ScoreTraces([]Trace{trace, duplicate}); err == nil {
		t.Fatal("ScoreTraces() error = nil, want duplicate task/run/condition rejection")
	}
}

func assertScoreExpectation(t *testing.T, got Result, want scoreExpectation) {
	t.Helper()
	if got.RepeatedToolOutputs != want.repeats {
		t.Errorf("repeated_tool_outputs = %d, want %d", got.RepeatedToolOutputs, want.repeats)
	}
	if got.EligibleToolOutputs != want.eligible {
		t.Errorf("eligible_tool_outputs = %d, want %d", got.EligibleToolOutputs, want.eligible)
	}
	if got.PostCompactionRepeatedToolOutputs != want.postRepeats {
		t.Errorf("post_compaction_repeated_tool_outputs = %d, want %d", got.PostCompactionRepeatedToolOutputs, want.postRepeats)
	}
	if got.ToolErrors != want.errors {
		t.Errorf("tool_errors = %d, want %d", got.ToolErrors, want.errors)
	}
	if got.ToolAttempts != want.attempts {
		t.Errorf("tool_attempts = %d, want %d", got.ToolAttempts, want.attempts)
	}
	if got.PostCompactionToolErrors != want.postErrors {
		t.Errorf("post_compaction_tool_errors = %d, want %d", got.PostCompactionToolErrors, want.postErrors)
	}
	if got.NonplanningNonpollingToolErrors != want.nonplanningErrors {
		t.Errorf("nonplanning_nonpolling_tool_errors = %d, want %d", got.NonplanningNonpollingToolErrors, want.nonplanningErrors)
	}
	if got.CancelledToolOutputs != want.cancelled {
		t.Errorf("cancelled_tool_outputs = %d, want %d", got.CancelledToolOutputs, want.cancelled)
	}
	if got.UnknownToolOutcomes != want.unknown {
		t.Errorf("unknown_tool_outcomes = %d, want %d", got.UnknownToolOutcomes, want.unknown)
	}
	if got.RetryDiagnostic.RepeatedIdenticalCalls != want.repeatedCalls {
		t.Errorf("repeated_identical_calls = %d, want %d", got.RetryDiagnostic.RepeatedIdenticalCalls, want.repeatedCalls)
	}
	if len(got.Warnings) != want.warningCount {
		t.Errorf("warnings = %v, want %d warning(s)", got.Warnings, want.warningCount)
	}
	assertOptionalRate(t, "repeated_tool_output_rate_micro", got.RepeatedToolOutputRateMicro, want.repeatRate)
	assertOptionalRate(t, "tool_error_rate_micro", got.ToolErrorRateMicro, want.errorRate)
}

func assertOptionalRate(t *testing.T, name string, got *float64, want optionalRate) {
	t.Helper()
	if !want.set {
		if got != nil {
			t.Errorf("%s = %v, want nil", name, *got)
		}
		return
	}
	if got == nil {
		t.Errorf("%s = nil, want %v", name, want.value)
		return
	}
	if math.Abs(*got-want.value) > 1e-12 {
		t.Errorf("%s = %.16g, want %.16g", name, *got, want.value)
	}
}

func traceWithEvents(taskID string, events []ToolEvent) Trace {
	trace := baseTrace(taskID)
	trace.ToolEvents = events
	return trace
}

func traceWithCheckpoint(taskID, checkpointID string, eventIndex int) Trace {
	trace := baseTrace(taskID)
	trace.Compactions = []Compaction{legacyCheckpoint(checkpointID, eventIndex)}
	return trace
}

func baseTrace(taskID string) Trace {
	return Trace{
		SchemaVersion: TraceSchemaVersion,
		SuiteID:       "fixture-suite",
		TaskID:        taskID,
		RunID:         "run-" + taskID,
		Condition:     "summary_only",
		Environment: Environment{
			NyawaRevision:       "fixture-nyawa",
			ActorRevision:       "fixture-actor",
			SummaryMode:         "captured",
			FingerprintPolicyID: DefaultFingerprintPolicyID,
		},
		Compactions:  []Compaction{},
		ToolEvents:   []ToolEvent{},
		RecallProbes: []RecallProbe{},
	}
}

func legacyCheckpoint(id string, eventIndex int) Compaction {
	return Compaction{
		CheckpointID:       id,
		EventIndex:         eventIndex,
		SourceSHA256:       testDigest("b"),
		SummarySHA256:      testDigest("c"),
		ResidualPackSHA256: nil,
		Namespace:          "context",
	}
}

func toolEvent(eventIndex int, toolName, digestHex string, outcome Outcome) ToolEvent {
	return ToolEvent{
		EventIndex:   eventIndex,
		ToolName:     toolName,
		OutputSHA256: testStringPointer(testDigest(digestHex)),
		Outcome:      outcome,
		IsPlanning:   false,
		IsPolling:    false,
	}
}

func toolEventWithCheckpoint(eventIndex int, toolName, digestHex string, outcome Outcome, checkpointID string) ToolEvent {
	event := toolEvent(eventIndex, toolName, digestHex, outcome)
	event.CheckpointID = testStringPointer(checkpointID)
	return event
}

func nyawaProbe(probeID, checkpointID string, status RecallTargetStatus, resultIDs []string) RecallProbe {
	return RecallProbe{
		ProbeID:                probeID,
		CheckpointID:           checkpointID,
		Query:                  "fixture query",
		Namespace:              "context",
		Limit:                  5,
		MinScore:               0,
		ExcludeTypes:           []string{},
		TargetMemoryIDs:        []string{"target-memory"},
		TargetResidualEntryIDs: []string{},
		Surface:                RecallSurfaceNyawa,
		TargetStatus:           status,
		ResultIDs:              resultIDs,
	}
}

func testDigest(hexDigit string) string {
	return "sha256:" + strings.Repeat(hexDigit, 64)
}

func testStringPointer(value string) *string {
	copy := value
	return &copy
}

func mutateJSON(t *testing.T, data []byte, mutate func(map[string]any)) []byte {
	t.Helper()
	var root map[string]any
	if err := json.Unmarshal(data, &root); err != nil {
		t.Fatalf("unmarshal fixture JSON: %v", err)
	}
	mutate(root)
	out, err := json.Marshal(root)
	if err != nil {
		t.Fatalf("marshal mutated fixture JSON: %v", err)
	}
	return out
}

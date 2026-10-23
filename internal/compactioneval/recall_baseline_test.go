package compactioneval

import (
	"strings"
	"testing"
)

func TestRecallProbePathGuardRequiresExplicitSnapshot(t *testing.T) {
	if _, err := ValidateSnapshotDBPath("/does/not/exist.db", false); err == nil {
		t.Fatal("non-snapshot path was accepted")
	}
	if _, err := ValidateSnapshotDBPath(":memory:", true); err == nil {
		t.Fatal("memory URI was accepted")
	}
	if _, err := ValidateSnapshotDBPath("/opt/data/.nyawa/memory.db", true); err == nil {
		t.Fatal("known production path was accepted")
	}
}

func TestBaselineReportSummarizesImmutableObservations(t *testing.T) {
	trace := baseTrace("baseline-task")
	trace.SessionID = "session-1"
	result, err := Score(trace)
	if err != nil {
		t.Fatal(err)
	}
	in, out, cost := 10, 4, 0.25
	report, err := BuildBaselineReport([]BaselineObservation{{Trace: trace, Result: result, InputTokens: &in, OutputTokens: &out, CostUSD: &cost}}, "2026-01-01", "2026-01-08")
	if err != nil {
		t.Fatal(err)
	}
	if report.Observations != 1 || report.UniqueTasks != 1 || report.SessionsWithoutCompaction != 1 || report.TokenObservations != 1 || report.CostObservations != 1 {
		t.Fatalf("report=%+v", report)
	}
	md := RenderBaselineMarkdown(report)
	if !strings.Contains(md, "Tool attempts") || !strings.Contains(md, "2026-01-01") {
		t.Fatalf("markdown=%s", md)
	}
	if _, err := BuildBaselineReport(nil, "2026-01-01", "2026-01-08"); err == nil {
		t.Fatal("empty observation window was reported as collected")
	}
}

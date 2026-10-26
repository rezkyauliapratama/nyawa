package compactioneval

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"
)

const BaselineReportSchemaVersion = "nyawa.compaction-baseline/v1"

type BaselineObservation struct {
	Trace        Trace    `json:"trace"`
	Result       Result   `json:"result"`
	SessionID    string   `json:"session_id,omitempty"`
	InputTokens  *int     `json:"input_tokens,omitempty"`
	OutputTokens *int     `json:"output_tokens,omitempty"`
	CostUSD      *float64 `json:"cost_usd,omitempty"`
}

type BaselineConditionSummary struct {
	Condition             string        `json:"condition"`
	Runs                  int           `json:"runs"`
	Compactions           int           `json:"compactions"`
	ToolAttempts          int           `json:"tool_attempts"`
	ToolErrors            int           `json:"tool_errors"`
	RepeatedToolOutputs   int           `json:"repeated_tool_outputs"`
	RecallAfterCompaction RecallMetrics `json:"recall_after_compaction"`
}

type BaselineReport struct {
	SchemaVersion             string                     `json:"schema_version"`
	ReportVersion             string                     `json:"report_version"`
	WindowStart               string                     `json:"window_start"`
	WindowEnd                 string                     `json:"window_end"`
	WindowDays                int                        `json:"window_days"`
	Observations              int                        `json:"observations"`
	UniqueTasks               int                        `json:"unique_tasks"`
	UniqueRuns                int                        `json:"unique_runs"`
	SessionsWithCompaction    int                        `json:"sessions_with_compaction"`
	SessionsWithoutCompaction int                        `json:"sessions_without_compaction"`
	ToolAttempts              int                        `json:"tool_attempts"`
	ToolErrors                int                        `json:"tool_errors"`
	RepeatedToolOutputs       int                        `json:"repeated_tool_outputs"`
	CancelledToolOutputs      int                        `json:"cancelled_tool_outputs"`
	UnknownToolOutcomes       int                        `json:"unknown_tool_outcomes"`
	InputTokens               int                        `json:"input_tokens"`
	OutputTokens              int                        `json:"output_tokens"`
	TokenObservations         int                        `json:"token_observations"`
	CostUSD                   *float64                   `json:"cost_usd"`
	CostObservations          int                        `json:"cost_observations"`
	Conditions                []BaselineConditionSummary `json:"conditions"`
	Warnings                  []string                   `json:"warnings"`
}

// BuildBaselineReport consumes immutable trace/result observations. It does not
// query live state, infer missing token/cost values, or claim that a window is
// collected when the caller supplies no observations.
func BuildBaselineReport(observations []BaselineObservation, windowStart, windowEnd string) (BaselineReport, error) {
	if strings.TrimSpace(windowStart) == "" || strings.TrimSpace(windowEnd) == "" {
		return BaselineReport{}, errors.New("window_start and window_end are required")
	}
	if len(observations) == 0 {
		return BaselineReport{}, errors.New("no observation data: baseline is not collected")
	}
	traces := make([]Trace, len(observations))
	for i, obs := range observations {
		traces[i] = obs.Trace
		if err := ValidateTrace(obs.Trace); err != nil {
			return BaselineReport{}, fmt.Errorf("observation[%d] trace: %w", i, err)
		}
	}
	if err := ValidateTraces(traces); err != nil {
		return BaselineReport{}, err
	}
	ordered := append([]BaselineObservation(nil), observations...)
	sort.Slice(ordered, func(i, j int) bool { return traceLess(ordered[i].Trace, ordered[j].Trace) })
	report := BaselineReport{SchemaVersion: BaselineReportSchemaVersion, ReportVersion: EvaluatorVersion, WindowStart: windowStart, WindowEnd: windowEnd, WindowDays: 7, Observations: len(ordered), Warnings: []string{}, Conditions: []BaselineConditionSummary{}}
	tasks := map[string]bool{}
	runs := map[string]bool{}
	sessions := map[string]bool{}
	sessionComp := map[string]bool{}
	byCondition := map[string]*BaselineConditionSummary{}
	var cost float64
	for _, obs := range ordered {
		trace := obs.Trace
		result := obs.Result
		tasks[trace.TaskID] = true
		runs[trace.RunID] = true
		sid := obs.SessionID
		if sid == "" {
			sid = trace.SessionID
		}
		if sid == "" {
			sid = trace.RunID
		}
		sessions[sid] = true
		if len(trace.Compactions) > 0 {
			sessionComp[sid] = true
		}
		report.ToolAttempts += result.ToolAttempts
		report.ToolErrors += result.ToolErrors
		report.RepeatedToolOutputs += result.RepeatedToolOutputs
		report.CancelledToolOutputs += result.CancelledToolOutputs
		report.UnknownToolOutcomes += result.UnknownToolOutcomes
		if obs.InputTokens != nil || obs.OutputTokens != nil {
			report.TokenObservations++
		}
		if obs.InputTokens != nil {
			report.InputTokens += *obs.InputTokens
		}
		if obs.OutputTokens != nil {
			report.OutputTokens += *obs.OutputTokens
		}
		if obs.CostUSD != nil {
			report.CostObservations++
			cost += *obs.CostUSD
		}
		condition := byCondition[trace.Condition]
		if condition == nil {
			condition = &BaselineConditionSummary{Condition: trace.Condition}
			byCondition[trace.Condition] = condition
		}
		condition.Runs++
		condition.Compactions += len(trace.Compactions)
		condition.ToolAttempts += result.ToolAttempts
		condition.ToolErrors += result.ToolErrors
		condition.RepeatedToolOutputs += result.RepeatedToolOutputs
		condition.RecallAfterCompaction = mergeRecallMetrics(condition.RecallAfterCompaction, result.RecallAfterCompaction)
		for _, warning := range result.Warnings {
			report.Warnings = append(report.Warnings, traceIdentity(trace)+": "+warning)
		}
	}
	report.UniqueTasks = len(tasks)
	report.UniqueRuns = len(runs)
	report.SessionsWithCompaction = len(sessionComp)
	report.SessionsWithoutCompaction = len(sessions) - len(sessionComp)
	if report.SessionsWithoutCompaction < 0 {
		report.SessionsWithoutCompaction = 0
	}
	if report.CostObservations > 0 {
		report.CostUSD = &cost
	}
	for _, condition := range byCondition {
		report.Conditions = append(report.Conditions, *condition)
	}
	sort.Slice(report.Conditions, func(i, j int) bool { return report.Conditions[i].Condition < report.Conditions[j].Condition })
	sort.Strings(report.Warnings)
	return report, nil
}

func mergeRecallMetrics(a, b RecallMetrics) RecallMetrics {
	a1, a3, a5, mrr := a.RecallAt1ContributingProbes, a.RecallAt3ContributingProbes, a.RecallAt5ContributingProbes, a.MRRAt5ContributingProbes
	a.ValidRecallProbes += b.ValidRecallProbes
	a.UnobservableRecallProbes += b.UnobservableRecallProbes
	a.SupersededTargetProbes += b.SupersededTargetProbes
	a.RecallAt1ContributingProbes += b.RecallAt1ContributingProbes
	a.RecallAt3ContributingProbes += b.RecallAt3ContributingProbes
	a.RecallAt5ContributingProbes += b.RecallAt5ContributingProbes
	a.MRRAt5ContributingProbes += b.MRRAt5ContributingProbes
	a.RecallAt1 = metricWeightedRate(a.RecallAt1, a1, b.RecallAt1, b.RecallAt1ContributingProbes)
	a.RecallAt3 = metricWeightedRate(a.RecallAt3, a3, b.RecallAt3, b.RecallAt3ContributingProbes)
	a.RecallAt5 = metricWeightedRate(a.RecallAt5, a5, b.RecallAt5, b.RecallAt5ContributingProbes)
	a.MRRAt5 = metricWeightedRate(a.MRRAt5, mrr, b.MRRAt5, b.MRRAt5ContributingProbes)
	return a
}
func metricWeightedRate(a *float64, an int, b *float64, bn int) *float64 {
	sum := 0.0
	den := 0
	if a != nil {
		sum += *a * float64(an)
		den += an
	}
	if b != nil {
		sum += *b * float64(bn)
		den += bn
	}
	if den == 0 {
		return nil
	}
	v := sum / float64(den)
	return &v
}
func MarshalBaselineJSON(report BaselineReport) ([]byte, error) { return json.Marshal(report) }

func RenderBaselineMarkdown(report BaselineReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Nyawa compaction baseline\n\n- Window: `%s` to `%s` (%d-day observation format)\n- Observations: %d\n- Unique tasks/runs: %d/%d\n- Sessions with/without compaction: %d/%d\n\n", report.WindowStart, report.WindowEnd, report.WindowDays, report.Observations, report.UniqueTasks, report.UniqueRuns, report.SessionsWithCompaction, report.SessionsWithoutCompaction)
	b.WriteString("## Aggregate\n\n| Metric | Value |\n|---|---:|\n")
	fmt.Fprintf(&b, "| Tool attempts | %d |\n| Tool errors | %d |\n| Repeated tool outputs | %d |\n| Cancelled outcomes | %d |\n| Unknown outcomes | %d |\n| Input tokens | %d (%d observations) |\n| Output tokens | %d |\n", report.ToolAttempts, report.ToolErrors, report.RepeatedToolOutputs, report.CancelledToolOutputs, report.UnknownToolOutcomes, report.InputTokens, report.TokenObservations, report.OutputTokens)
	if report.CostUSD != nil {
		fmt.Fprintf(&b, "| Cost USD | %.6f (%d observations) |\n", *report.CostUSD, report.CostObservations)
	} else {
		b.WriteString("| Cost USD | unavailable |\n")
	}
	b.WriteString("\n## Conditions\n\n| Condition | Runs | Compactions | Attempts | Errors | Repeats |\n|---|---:|---:|---:|---:|---:|\n")
	for _, condition := range report.Conditions {
		fmt.Fprintf(&b, "| %s | %d | %d | %d | %d | %d |\n", condition.Condition, condition.Runs, condition.Compactions, condition.ToolAttempts, condition.ToolErrors, condition.RepeatedToolOutputs)
	}
	if len(report.Warnings) > 0 {
		b.WriteString("\n## Warnings\n\n")
		for _, warning := range report.Warnings {
			fmt.Fprintf(&b, "- %s\n", warning)
		}
	}
	return b.String()
}

// DecodeBaselineObservations reads a strict JSON array of immutable observations.
func DecodeBaselineObservations(data []byte) ([]BaselineObservation, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var observations []BaselineObservation
	if err := decoder.Decode(&observations); err != nil {
		return nil, err
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, errors.New("observations must contain exactly one JSON value")
	}
	return observations, nil
}

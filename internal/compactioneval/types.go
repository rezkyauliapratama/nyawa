// Package compactioneval scores immutable compaction-evaluation traces without
// touching a database, network, clock, model, or other runtime dependency.
package compactioneval

// TraceSchemaVersion is the only trace schema accepted by this evaluator.
const TraceSchemaVersion = "nyawa.compaction-eval/v1"

// ResultSchemaVersion is the schema emitted by this evaluator.
const ResultSchemaVersion = "nyawa.compaction-eval-result/v1"

// EvaluatorVersion identifies the metric semantics implemented by this package.
const EvaluatorVersion = "v1"

// DefaultFingerprintPolicyID is the strict byte-fingerprint policy described by
// the compaction-evaluator specification.
const DefaultFingerprintPolicyID = "tool-output-bytes-v1"

// Condition labels the intervention used for a task run.
const (
	ConditionNoCompaction            = "no_compaction"
	ConditionSummaryOnly             = "summary_only"
	ConditionSummaryPlusResidualPack = "summary_plus_residual_pack"
	// ConditionSummaryPlusBaseline is the read-only P2 comparison condition.
	// It does not imply residual-pack activation or runtime behavior.
	ConditionSummaryPlusBaseline = "summary_plus_baseline"
	ConditionRetrievalOnly       = "retrieval_only"
)

// Outcome is the trace producer's provider-native classification of a tool
// attempt. The evaluator does not infer it from tool text.
type Outcome string

const (
	OutcomeSuccess   Outcome = "success"
	OutcomeFailure   Outcome = "failure"
	OutcomeToolError Outcome = "tool_error"
	OutcomeTimeout   Outcome = "timeout"
	OutcomeCancelled Outcome = "cancelled"
	OutcomeUnknown   Outcome = "unknown"
)

// RecallSurface identifies the retrieval surface that produced a probe's
// ordered result IDs.
type RecallSurface string

const (
	RecallSurfaceNyawa        RecallSurface = "nyawa_recall"
	RecallSurfaceResidualPack RecallSurface = "residual_pack_lookup"
)

// RecallTargetStatus records whether a probe's target can participate in the
// tested active retrieval surface. An omitted status means active for backwards
// compatibility with the proposal's initial probe example.
type RecallTargetStatus string

const (
	RecallTargetActive       RecallTargetStatus = "active"
	RecallTargetUnobservable RecallTargetStatus = "unobservable"
	RecallTargetSuperseded   RecallTargetStatus = "superseded"
)

// Trace is the immutable JSON artifact consumed by the pure scorer.
type Trace struct {
	SchemaVersion string        `json:"schema_version"`
	SuiteID       string        `json:"suite_id"`
	TaskID        string        `json:"task_id"`
	RunID         string        `json:"run_id"`
	Condition     string        `json:"condition"`
	Environment   Environment   `json:"environment"`
	Compactions   []Compaction  `json:"compactions"`
	ToolEvents    []ToolEvent   `json:"tool_events"`
	RecallProbes  []RecallProbe `json:"recall_probes"`
	// SessionID is additive collector metadata. Existing offline traces may omit it.
	SessionID string `json:"session_id,omitempty"`
}

// Environment records the fixed evaluation context needed to compare traces.
type Environment struct {
	NyawaRevision       string `json:"nyawa_revision"`
	ActorRevision       string `json:"actor_revision"`
	SummaryMode         string `json:"summary_mode"`
	FingerprintPolicyID string `json:"fingerprint_policy_id"`
}

// Compaction records one completed compaction boundary. EventIndex is the first
// event after that completed boundary.
type Compaction struct {
	CheckpointID       string  `json:"checkpoint_id"`
	EventIndex         int     `json:"event_index"`
	SourceSHA256       string  `json:"source_sha256"`
	SummarySHA256      string  `json:"summary_sha256"`
	ResidualPackSHA256 *string `json:"residual_pack_sha256"`
	Namespace          string  `json:"namespace"`
	// The following fields are additive lifecycle attribution emitted by the passive collector.
	BeforeSessionID string `json:"before_session_id,omitempty"`
	AfterSessionID  string `json:"after_session_id,omitempty"`
	Timestamp       string `json:"timestamp,omitempty"`
}

// ToolEvent is one tool attempt. Optional pointer fields retain JSON null when
// the trace producer could not provide a value.
type ToolEvent struct {
	EventIndex      int     `json:"event_index"`
	ToolName        string  `json:"tool_name"`
	ToolInputSHA256 *string `json:"tool_input_sha256"`
	OutputSHA256    *string `json:"output_sha256"`
	Outcome         Outcome `json:"outcome"`
	ErrorClass      *string `json:"error_class"`
	IsPlanning      bool    `json:"is_planning"`
	IsPolling       bool    `json:"is_polling"`
	CheckpointID    *string `json:"checkpoint_id"`
	// These fields are additive telemetry metadata. They are not used by v1 scoring.
	ToolCallID        string   `json:"tool_call_id,omitempty"`
	Timestamp         string   `json:"timestamp,omitempty"`
	RecallQuery       *string  `json:"recall_query,omitempty"`
	ReturnedMemoryIDs []string `json:"returned_memory_ids,omitempty"`
}

// RecallProbe records one already-executed retrieval request and its ordered
// result IDs. ResultIDs is optional because the proposal's original wire
// example predates a response field; absent or empty results are scored as a
// miss for an otherwise valid probe.
type RecallProbe struct {
	ProbeID                string             `json:"probe_id"`
	CheckpointID           string             `json:"checkpoint_id"`
	Query                  string             `json:"query"`
	Namespace              string             `json:"namespace"`
	Limit                  int                `json:"limit"`
	MinScore               float64            `json:"min_score"`
	ExcludeTypes           []string           `json:"exclude_types"`
	TargetMemoryIDs        []string           `json:"target_memory_ids"`
	TargetResidualEntryIDs []string           `json:"target_residual_entry_ids"`
	Surface                RecallSurface      `json:"surface"`
	TargetStatus           RecallTargetStatus `json:"target_status,omitempty"`
	ResultIDs              []string           `json:"result_ids,omitempty"`
	// Additive producer metadata; the scorer continues to use Query and ResultIDs.
	Timestamp  string `json:"timestamp,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// Result is the machine-readable output of Score or ScoreTraces. All rates are
// nil (and therefore JSON null) when their denominator has no contributors.
type Result struct {
	SchemaVersion                               string              `json:"schema_version"`
	EvaluatorVersion                            string              `json:"evaluator_version"`
	Condition                                   string              `json:"condition"`
	FingerprintPolicyID                         string              `json:"fingerprint_policy_id"`
	Runs                                        int                 `json:"runs"`
	RepeatedToolOutputs                         int                 `json:"repeated_tool_outputs"`
	EligibleToolOutputs                         int                 `json:"eligible_tool_outputs"`
	RepeatedToolOutputRateMicro                 *float64            `json:"repeated_tool_output_rate_micro"`
	RepeatedToolOutputRateMacro                 *float64            `json:"repeated_tool_output_rate_macro"`
	RepeatedToolOutputRateMacroContributingRuns int                 `json:"repeated_tool_output_rate_macro_contributing_runs"`
	PostCompactionRepeatedToolOutputs           int                 `json:"post_compaction_repeated_tool_outputs"`
	ToolErrors                                  int                 `json:"tool_errors"`
	ToolAttempts                                int                 `json:"tool_attempts"`
	ToolErrorRateMicro                          *float64            `json:"tool_error_rate_micro"`
	ToolErrorRateMacro                          *float64            `json:"tool_error_rate_macro"`
	ToolErrorRateMacroContributingRuns          int                 `json:"tool_error_rate_macro_contributing_runs"`
	PostCompactionToolErrors                    int                 `json:"post_compaction_tool_errors"`
	NonplanningNonpollingToolErrors             int                 `json:"nonplanning_nonpolling_tool_errors"`
	CancelledToolOutputs                        int                 `json:"cancelled_tool_outputs"`
	UnknownToolOutcomes                         int                 `json:"unknown_tool_outcomes"`
	CheckpointMetrics                           []CheckpointMetrics `json:"checkpoint_metrics"`
	RepeatGroups                                []RepeatGroup       `json:"repeat_groups"`
	RetryDiagnostic                             RetryDiagnostic     `json:"retry_diagnostic"`
	RecallAfterCompaction                       RecallMetrics       `json:"recall_after_compaction"`
	ResidualPackLookupAfterCompaction           RecallMetrics       `json:"residual_pack_lookup_after_compaction"`
	Warnings                                    []string            `json:"warnings"`
}

// CheckpointMetrics is the post-compaction breakdown attributed by event index
// to one completed boundary.
type CheckpointMetrics struct {
	SuiteID                         string `json:"suite_id"`
	TaskID                          string `json:"task_id"`
	RunID                           string `json:"run_id"`
	CheckpointID                    string `json:"checkpoint_id"`
	BoundaryEventIndex              int    `json:"boundary_event_index"`
	ToolAttempts                    int    `json:"tool_attempts"`
	EligibleToolOutputs             int    `json:"eligible_tool_outputs"`
	RepeatedToolOutputs             int    `json:"repeated_tool_outputs"`
	ToolErrors                      int    `json:"tool_errors"`
	NonplanningNonpollingToolErrors int    `json:"nonplanning_nonpolling_tool_errors"`
	CancelledToolOutputs            int    `json:"cancelled_tool_outputs"`
	UnknownToolOutcomes             int    `json:"unknown_tool_outcomes"`
}

// RepeatGroup is an auditable primary repeat group. Counts are calculated over
// the full task run and are never reset at a compaction boundary.
type RepeatGroup struct {
	SuiteID                string                        `json:"suite_id"`
	TaskID                 string                        `json:"task_id"`
	RunID                  string                        `json:"run_id"`
	ToolName               string                        `json:"tool_name"`
	OutputSHA256           string                        `json:"output_sha256"`
	Count                  int                           `json:"count"`
	FirstEventIndex        int                           `json:"first_event_index"`
	RepeatContribution     int                           `json:"repeat_contribution"`
	CheckpointAttributions []RepeatCheckpointAttribution `json:"checkpoint_attributions"`
}

// RepeatCheckpointAttribution breaks a repeat group down by the boundary that
// owned each eligible event. A nil CheckpointID represents pre-compaction work.
type RepeatCheckpointAttribution struct {
	CheckpointID       *string `json:"checkpoint_id"`
	EventCount         int     `json:"event_count"`
	RepeatContribution int     `json:"repeat_contribution"`
}

// RetryDiagnostic is a secondary, non-primary grouping that includes the input
// fingerprint so callers can distinguish repeated calls from repeated outputs.
type RetryDiagnostic struct {
	RepeatedIdenticalCalls int                    `json:"repeated_identical_calls"`
	Groups                 []RetryDiagnosticGroup `json:"groups"`
}

// RetryDiagnosticGroup groups eligible events by tool name, input fingerprint,
// and output fingerprint.
type RetryDiagnosticGroup struct {
	SuiteID           string  `json:"suite_id"`
	TaskID            string  `json:"task_id"`
	RunID             string  `json:"run_id"`
	ToolName          string  `json:"tool_name"`
	ToolInputSHA256   *string `json:"tool_input_sha256"`
	OutputSHA256      string  `json:"output_sha256"`
	Count             int     `json:"count"`
	FirstEventIndex   int     `json:"first_event_index"`
	RetryContribution int     `json:"retry_contribution"`
}

// RecallMetrics reports one retrieval surface. RecallAt* fields are micro
// rates; their macro counterparts average non-null per-run rates.
type RecallMetrics struct {
	ValidRecallProbes              int      `json:"valid_recall_probes"`
	UnobservableRecallProbes       int      `json:"unobservable_recall_probes"`
	SupersededTargetProbes         int      `json:"superseded_target_probes"`
	RecallAt1                      *float64 `json:"recall_at_1"`
	RecallAt1ContributingProbes    int      `json:"recall_at_1_contributing_probes"`
	RecallAt1Macro                 *float64 `json:"recall_at_1_macro"`
	RecallAt1MacroContributingRuns int      `json:"recall_at_1_macro_contributing_runs"`
	RecallAt3                      *float64 `json:"recall_at_3"`
	RecallAt3ContributingProbes    int      `json:"recall_at_3_contributing_probes"`
	RecallAt3Macro                 *float64 `json:"recall_at_3_macro"`
	RecallAt3MacroContributingRuns int      `json:"recall_at_3_macro_contributing_runs"`
	RecallAt5                      *float64 `json:"recall_at_5"`
	RecallAt5ContributingProbes    int      `json:"recall_at_5_contributing_probes"`
	RecallAt5Macro                 *float64 `json:"recall_at_5_macro"`
	RecallAt5MacroContributingRuns int      `json:"recall_at_5_macro_contributing_runs"`
	MRRAt5                         *float64 `json:"mrr_at_5"`
	MRRAt5ContributingProbes       int      `json:"mrr_at_5_contributing_probes"`
	MRRAt5Macro                    *float64 `json:"mrr_at_5_macro"`
	MRRAt5MacroContributingRuns    int      `json:"mrr_at_5_macro_contributing_runs"`
}

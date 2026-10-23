package compactioneval

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// ValidationError identifies one invalid field in a trace or trace set.
type ValidationError struct {
	Path    string
	Message string
}

func (e ValidationError) Error() string {
	if e.Path == "" {
		return e.Message
	}
	return e.Path + ": " + e.Message
}

// ValidationErrors holds every validation problem found in one pass.
type ValidationErrors []ValidationError

func (es ValidationErrors) Error() string {
	parts := make([]string, len(es))
	for i, e := range es {
		parts[i] = e.Error()
	}
	return strings.Join(parts, "; ")
}

func (es ValidationErrors) asError() error {
	if len(es) == 0 {
		return nil
	}
	return es
}

func (es *ValidationErrors) add(path, message string) {
	*es = append(*es, ValidationError{Path: path, Message: message})
}

// DecodeTraceJSON strictly decodes one immutable trace JSON object. It rejects
// unknown fields, missing required fields, null required values, trailing JSON,
// and semantically invalid traces.
func DecodeTraceJSON(data []byte) (Trace, error) {
	if err := validateTraceJSONShape(data); err != nil {
		return Trace{}, err
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	var trace Trace
	if err := decoder.Decode(&trace); err != nil {
		return Trace{}, fmt.Errorf("decode trace JSON: %w", err)
	}

	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		if err == nil {
			return Trace{}, ValidationErrors{{Path: "$", Message: "must contain exactly one JSON value"}}
		}
		return Trace{}, fmt.Errorf("decode trailing trace JSON: %w", err)
	}

	if err := ValidateTrace(trace); err != nil {
		return Trace{}, err
	}
	return trace, nil
}

// ValidateTrace checks the trace rules that do not require a database, server,
// network, or other mutable dependency.
func ValidateTrace(trace Trace) error {
	var problems ValidationErrors

	if trace.SchemaVersion != TraceSchemaVersion {
		problems.add("$.schema_version", fmt.Sprintf("must equal %q", TraceSchemaVersion))
	}
	validateNonEmpty(&problems, "$.suite_id", trace.SuiteID)
	validateNonEmpty(&problems, "$.task_id", trace.TaskID)
	validateNonEmpty(&problems, "$.run_id", trace.RunID)
	if !validCondition(trace.Condition) {
		problems.add("$.condition", "must be one of no_compaction, summary_only, summary_plus_baseline, summary_plus_residual_pack, retrieval_only")
	}
	validateNonEmpty(&problems, "$.environment.nyawa_revision", trace.Environment.NyawaRevision)
	validateNonEmpty(&problems, "$.environment.actor_revision", trace.Environment.ActorRevision)
	validateNonEmpty(&problems, "$.environment.summary_mode", trace.Environment.SummaryMode)
	validateNonEmpty(&problems, "$.environment.fingerprint_policy_id", trace.Environment.FingerprintPolicyID)

	if trace.Compactions == nil {
		problems.add("$.compactions", "must be an array; use [] when there are no completed compactions")
	}
	if trace.ToolEvents == nil {
		problems.add("$.tool_events", "must be an array; use [] when there are no tool events")
	}
	if trace.RecallProbes == nil {
		problems.add("$.recall_probes", "must be an array; use [] when there are no recall probes")
	}
	if trace.Condition == ConditionNoCompaction && len(trace.Compactions) != 0 {
		problems.add("$.compactions", "must be empty for condition no_compaction")
	}

	checkpointByID := make(map[string]Compaction, len(trace.Compactions))
	previousBoundary := -1
	for i, compaction := range trace.Compactions {
		path := fmt.Sprintf("$.compactions[%d]", i)
		validateNonEmpty(&problems, path+".checkpoint_id", compaction.CheckpointID)
		if compaction.EventIndex < 0 {
			problems.add(path+".event_index", "must be non-negative")
		}
		if i > 0 && compaction.EventIndex <= previousBoundary {
			problems.add(path+".event_index", "must be strictly increasing and unique")
		}
		previousBoundary = compaction.EventIndex
		validateDigest(&problems, path+".source_sha256", compaction.SourceSHA256)
		validateDigest(&problems, path+".summary_sha256", compaction.SummarySHA256)
		validateNonEmpty(&problems, path+".namespace", compaction.Namespace)

		if trace.Condition == ConditionSummaryPlusResidualPack {
			if !validCanonicalCheckpointID(compaction.CheckpointID) {
				problems.add(path+".checkpoint_id", "must be ccp_v1_ followed by a lowercase SHA-256 digest for summary_plus_residual_pack")
			}
			if compaction.ResidualPackSHA256 == nil {
				problems.add(path+".residual_pack_sha256", "is required for summary_plus_residual_pack")
			} else {
				validateDigest(&problems, path+".residual_pack_sha256", *compaction.ResidualPackSHA256)
			}
		} else if compaction.ResidualPackSHA256 != nil {
			problems.add(path+".residual_pack_sha256", "must be null unless condition is summary_plus_residual_pack")
		}

		if _, exists := checkpointByID[compaction.CheckpointID]; exists && compaction.CheckpointID != "" {
			problems.add(path+".checkpoint_id", "must be unique within a run")
		}
		checkpointByID[compaction.CheckpointID] = compaction
	}

	previousEvent := -1
	for i, event := range trace.ToolEvents {
		path := fmt.Sprintf("$.tool_events[%d]", i)
		if event.EventIndex < 0 {
			problems.add(path+".event_index", "must be non-negative")
		}
		if i > 0 && event.EventIndex <= previousEvent {
			problems.add(path+".event_index", "must be strictly increasing and unique")
		}
		previousEvent = event.EventIndex

		validateNonEmpty(&problems, path+".tool_name", event.ToolName)
		if event.ToolInputSHA256 != nil {
			validateDigest(&problems, path+".tool_input_sha256", *event.ToolInputSHA256)
		}
		if event.OutputSHA256 != nil {
			validateDigest(&problems, path+".output_sha256", *event.OutputSHA256)
		}
		if !validOutcome(event.Outcome) {
			problems.add(path+".outcome", "must be one of success, failure, tool_error, timeout, cancelled, unknown")
		}
		if event.ErrorClass != nil {
			validateNonEmpty(&problems, path+".error_class", *event.ErrorClass)
		}
		if event.CheckpointID != nil {
			validateNonEmpty(&problems, path+".checkpoint_id", *event.CheckpointID)
			expected := latestCompactionAt(trace.Compactions, event.EventIndex)
			if expected == nil {
				problems.add(path+".checkpoint_id", "must be null before the first completed checkpoint")
			} else if *event.CheckpointID != expected.CheckpointID {
				problems.add(path+".checkpoint_id", "must equal the most recent completed checkpoint at this event index")
			}
		}
	}

	probeIDs := make(map[string]struct{}, len(trace.RecallProbes))
	for i, probe := range trace.RecallProbes {
		path := fmt.Sprintf("$.recall_probes[%d]", i)
		validateNonEmpty(&problems, path+".probe_id", probe.ProbeID)
		if _, exists := probeIDs[probe.ProbeID]; exists && probe.ProbeID != "" {
			problems.add(path+".probe_id", "must be unique within a run")
		}
		probeIDs[probe.ProbeID] = struct{}{}

		validateNonEmpty(&problems, path+".checkpoint_id", probe.CheckpointID)
		compaction, checkpointExists := checkpointByID[probe.CheckpointID]
		if !checkpointExists {
			problems.add(path+".checkpoint_id", "must reference a completed checkpoint in this run")
		}
		validateNonEmpty(&problems, path+".query", probe.Query)
		validateNonEmpty(&problems, path+".namespace", probe.Namespace)
		if probe.Limit < 1 {
			problems.add(path+".limit", "must be greater than zero")
		}
		if math.IsNaN(probe.MinScore) || math.IsInf(probe.MinScore, 0) {
			problems.add(path+".min_score", "must be finite")
		}
		if probe.ExcludeTypes == nil {
			problems.add(path+".exclude_types", "must be an array; use [] when no types are excluded")
		}
		validateStringSlice(&problems, path+".exclude_types", probe.ExcludeTypes)
		if probe.TargetMemoryIDs == nil {
			problems.add(path+".target_memory_ids", "must be an array; use [] when unused")
		}
		if probe.TargetResidualEntryIDs == nil {
			problems.add(path+".target_residual_entry_ids", "must be an array; use [] when unused")
		}
		validateStringSlice(&problems, path+".target_memory_ids", probe.TargetMemoryIDs)
		validateStringSlice(&problems, path+".target_residual_entry_ids", probe.TargetResidualEntryIDs)
		validateStringSlice(&problems, path+".result_ids", probe.ResultIDs)
		if probe.ResultIDs != nil && probe.Limit >= 0 && len(probe.ResultIDs) > probe.Limit {
			problems.add(path+".result_ids", "must not contain more IDs than limit")
		}
		if !validRecallSurface(probe.Surface) {
			problems.add(path+".surface", "must be nyawa_recall or residual_pack_lookup")
		}
		if !validRecallTargetStatus(probe.TargetStatus) {
			problems.add(path+".target_status", "must be active, unobservable, or superseded when present")
		}

		switch probe.Surface {
		case RecallSurfaceNyawa:
			if len(probe.TargetMemoryIDs) == 0 {
				problems.add(path+".target_memory_ids", "must contain at least one target for nyawa_recall")
			}
			if len(probe.TargetResidualEntryIDs) != 0 {
				problems.add(path+".target_residual_entry_ids", "must be empty for nyawa_recall")
			}
		case RecallSurfaceResidualPack:
			if len(probe.TargetResidualEntryIDs) == 0 {
				problems.add(path+".target_residual_entry_ids", "must contain at least one target for residual_pack_lookup")
			}
			if len(probe.TargetMemoryIDs) != 0 {
				problems.add(path+".target_memory_ids", "must be empty for residual_pack_lookup")
			}
			if checkpointExists && compaction.ResidualPackSHA256 == nil {
				problems.add(path+".checkpoint_id", "must reference a checkpoint with a residual pack for residual_pack_lookup")
			}
		}
	}

	return problems.asError()
}

// ValidateTraces validates a collection and rejects duplicate task/run/condition
// records before an aggregate result can silently double-count them.
func ValidateTraces(traces []Trace) error {
	var problems ValidationErrors
	if len(traces) == 0 {
		problems.add("$", "at least one trace is required")
		return problems
	}

	for i, trace := range traces {
		if err := ValidateTrace(trace); err != nil {
			problems.add(fmt.Sprintf("traces[%d]", i), err.Error())
		}
	}

	type runIdentityKey struct {
		taskID    string
		runID     string
		condition string
	}
	seen := make(map[runIdentityKey]int, len(traces))
	for i, trace := range traces {
		if trace.TaskID == "" || trace.RunID == "" || trace.Condition == "" {
			continue
		}
		key := runIdentityKey{taskID: trace.TaskID, runID: trace.RunID, condition: trace.Condition}
		if first, exists := seen[key]; exists {
			problems.add(fmt.Sprintf("traces[%d]", i), fmt.Sprintf("duplicates traces[%d] for task_id/run_id/condition", first))
			continue
		}
		seen[key] = i
	}

	return problems.asError()
}

func validateTraceJSONShape(data []byte) error {
	root, err := jsonObject(data, "$")
	if err != nil {
		return err
	}

	var problems ValidationErrors
	rejectUnknownFields(&problems, root, "$", []string{"schema_version", "suite_id", "task_id", "run_id", "condition", "environment", "compactions", "tool_events", "recall_probes", "session_id"})
	requireFields(&problems, root, "$",
		[]string{"schema_version", "suite_id", "task_id", "run_id", "condition", "environment", "compactions", "tool_events", "recall_probes"},
		nil,
	)

	if raw, ok := root["environment"]; ok && !isJSONNull(raw) {
		environment, objectErr := jsonObject(raw, "$.environment")
		if objectErr != nil {
			appendValidationError(&problems, objectErr)
		} else {
			rejectUnknownFields(&problems, environment, "$.environment", []string{"nyawa_revision", "actor_revision", "summary_mode", "fingerprint_policy_id"})
			requireFields(&problems, environment, "$.environment",
				[]string{"nyawa_revision", "actor_revision", "summary_mode", "fingerprint_policy_id"}, nil)
		}
	}

	validateCompactionJSONShape(&problems, root["compactions"])
	validateToolEventJSONShape(&problems, root["tool_events"])
	validateRecallProbeJSONShape(&problems, root["recall_probes"])
	return problems.asError()
}

func validateCompactionJSONShape(problems *ValidationErrors, raw json.RawMessage) {
	if raw == nil || isJSONNull(raw) {
		return
	}
	items, err := jsonArray(raw, "$.compactions")
	if err != nil {
		appendValidationError(problems, err)
		return
	}
	for i, item := range items {
		path := fmt.Sprintf("$.compactions[%d]", i)
		object, objectErr := jsonObject(item, path)
		if objectErr != nil {
			appendValidationError(problems, objectErr)
			continue
		}
		rejectUnknownFields(problems, object, path, []string{"checkpoint_id", "event_index", "source_sha256", "summary_sha256", "residual_pack_sha256", "namespace", "before_session_id", "after_session_id", "timestamp"})
		requireFields(problems, object, path,
			[]string{"checkpoint_id", "event_index", "source_sha256", "summary_sha256", "namespace"},
			[]string{"residual_pack_sha256"},
		)
	}
}

func validateToolEventJSONShape(problems *ValidationErrors, raw json.RawMessage) {
	if raw == nil || isJSONNull(raw) {
		return
	}
	items, err := jsonArray(raw, "$.tool_events")
	if err != nil {
		appendValidationError(problems, err)
		return
	}
	for i, item := range items {
		path := fmt.Sprintf("$.tool_events[%d]", i)
		object, objectErr := jsonObject(item, path)
		if objectErr != nil {
			appendValidationError(problems, objectErr)
			continue
		}
		rejectUnknownFields(problems, object, path, []string{"event_index", "tool_name", "tool_input_sha256", "output_sha256", "outcome", "error_class", "is_planning", "is_polling", "checkpoint_id", "tool_call_id", "timestamp", "recall_query", "returned_memory_ids"})
		requireFields(problems, object, path,
			[]string{"event_index", "tool_name", "outcome", "is_planning", "is_polling"},
			[]string{"output_sha256"},
		)
	}
}

func validateRecallProbeJSONShape(problems *ValidationErrors, raw json.RawMessage) {
	if raw == nil || isJSONNull(raw) {
		return
	}
	items, err := jsonArray(raw, "$.recall_probes")
	if err != nil {
		appendValidationError(problems, err)
		return
	}
	for i, item := range items {
		path := fmt.Sprintf("$.recall_probes[%d]", i)
		object, objectErr := jsonObject(item, path)
		if objectErr != nil {
			appendValidationError(problems, objectErr)
			continue
		}
		rejectUnknownFields(problems, object, path, []string{"probe_id", "checkpoint_id", "query", "namespace", "limit", "min_score", "exclude_types", "target_memory_ids", "target_residual_entry_ids", "surface", "target_status", "result_ids", "timestamp", "tool_call_id"})
		requireFields(problems, object, path,
			[]string{"probe_id", "checkpoint_id", "query", "namespace", "limit", "min_score", "exclude_types", "target_memory_ids", "target_residual_entry_ids", "surface"},
			nil,
		)
	}
}

func rejectUnknownFields(problems *ValidationErrors, object map[string]json.RawMessage, path string, allowed []string) {
	allowedFields := make(map[string]struct{}, len(allowed))
	for _, field := range allowed {
		allowedFields[field] = struct{}{}
	}
	unknownFields := make([]string, 0)
	for field := range object {
		if _, allowed := allowedFields[field]; !allowed {
			unknownFields = append(unknownFields, field)
		}
	}
	sort.Strings(unknownFields)
	for _, field := range unknownFields {
		problems.add(path+"."+field, "is not allowed by "+TraceSchemaVersion)
	}
}

func requireFields(problems *ValidationErrors, object map[string]json.RawMessage, path string, requiredNonNull, requiredNullable []string) {
	for _, field := range requiredNonNull {
		raw, exists := object[field]
		if !exists {
			problems.add(path+"."+field, "is required")
			continue
		}
		if isJSONNull(raw) {
			problems.add(path+"."+field, "must not be null")
		}
	}
	for _, field := range requiredNullable {
		if _, exists := object[field]; !exists {
			problems.add(path+"."+field, "is required")
		}
	}
}

func jsonObject(raw []byte, path string) (map[string]json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return nil, ValidationErrors{{Path: path, Message: "must be a JSON object"}}
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &object); err != nil {
		return nil, fmt.Errorf("decode JSON object at %s: %w", path, err)
	}
	return object, nil
}

func jsonArray(raw []byte, path string) ([]json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '[' {
		return nil, ValidationErrors{{Path: path, Message: "must be a JSON array"}}
	}
	var items []json.RawMessage
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return nil, fmt.Errorf("decode JSON array at %s: %w", path, err)
	}
	return items, nil
}

func isJSONNull(raw []byte) bool {
	return bytes.Equal(bytes.TrimSpace(raw), []byte("null"))
}

func appendValidationError(problems *ValidationErrors, err error) {
	if validationErrors, ok := err.(ValidationErrors); ok {
		*problems = append(*problems, validationErrors...)
		return
	}
	problems.add("$", err.Error())
}

func validateNonEmpty(problems *ValidationErrors, path, value string) {
	if value == "" {
		problems.add(path, "must not be empty")
	}
}

func validateStringSlice(problems *ValidationErrors, path string, values []string) {
	for i, value := range values {
		validateNonEmpty(problems, fmt.Sprintf("%s[%d]", path, i), value)
	}
}

func validateDigest(problems *ValidationErrors, path, digest string) {
	if !validSHA256Digest(digest) {
		problems.add(path, "must be sha256: followed by 64 lowercase hexadecimal characters")
	}
}

func validSHA256Digest(digest string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(digest, prefix) || len(digest) != len(prefix)+64 {
		return false
	}
	for _, char := range digest[len(prefix):] {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

func validCanonicalCheckpointID(checkpointID string) bool {
	const prefix = "ccp_v1_"
	if !strings.HasPrefix(checkpointID, prefix) || len(checkpointID) != len(prefix)+64 {
		return false
	}
	for _, char := range checkpointID[len(prefix):] {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

func validCondition(condition string) bool {
	switch condition {
	case ConditionNoCompaction, ConditionSummaryOnly, ConditionSummaryPlusBaseline, ConditionSummaryPlusResidualPack, ConditionRetrievalOnly:
		return true
	default:
		return false
	}
}

func validOutcome(outcome Outcome) bool {
	switch outcome {
	case OutcomeSuccess, OutcomeFailure, OutcomeToolError, OutcomeTimeout, OutcomeCancelled, OutcomeUnknown:
		return true
	default:
		return false
	}
}

func validRecallSurface(surface RecallSurface) bool {
	return surface == RecallSurfaceNyawa || surface == RecallSurfaceResidualPack
}

func validRecallTargetStatus(status RecallTargetStatus) bool {
	switch status {
	case "", RecallTargetActive, RecallTargetUnobservable, RecallTargetSuperseded:
		return true
	default:
		return false
	}
}

func latestCompactionAt(compactions []Compaction, eventIndex int) *Compaction {
	var latest *Compaction
	for i := range compactions {
		compaction := &compactions[i]
		if compaction.EventIndex <= eventIndex && (latest == nil || compaction.EventIndex > latest.EventIndex) {
			latest = compaction
		}
	}
	return latest
}

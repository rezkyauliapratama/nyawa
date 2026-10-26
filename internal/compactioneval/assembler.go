package compactioneval

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
)

// AssembleJSONL folds the passive collector's event JSONL into one canonical
// immutable Trace. It ignores unknown collector fields and never copies payload
// fields; only hashes, identifiers, redacted recall metadata, and status fields
// are admitted to the result.
func AssembleJSONL(data []byte) (Trace, error) {
	return assembleJSONL(bytes.NewReader(data))
}

// AssembleJSONLFile reads an event stream from a caller-selected file. It does
// not open a database, network connection, or production runtime.
func AssembleJSONLFile(path string) (Trace, error) {
	if path == "" || path == "-" {
		return Trace{}, fmt.Errorf("trace JSONL path must be an explicit file")
	}
	file, err := os.Open(path)
	if err != nil {
		return Trace{}, fmt.Errorf("open trace JSONL: %w", err)
	}
	defer file.Close()
	return assembleJSONL(file)
}

type collectorRecord struct {
	SchemaVersion string
	RecordType    string
	RecordSchema  string
	SuiteID       string
	TaskID        string
	RunID         string
	Condition     string
	Environment   Environment
	SessionID     string
	EventIndex    int
	Timestamp     string

	ToolName        string
	ToolCallID      string
	ToolInputSHA256 *string
	OutputSHA256    *string
	Outcome         Outcome
	ErrorClass      *string
	IsPlanning      bool
	IsPolling       bool
	CheckpointID    *string
	RecallQuery     *string
	ReturnedIDs     []string

	BeforeSessionID string
	AfterSessionID  string
	SourceSHA256    string
	SummarySHA256   string
	Completed       bool
	Reason          string
	MessagesSHA256  string
}

type assembleRun struct {
	key         string
	suiteID     string
	taskID      string
	runID       string
	condition   string
	environment Environment
	sessionID   string
	records     []collectorRecord
	prepares    []collectorRecord
	compactions []collectorRecord
}

func assembleJSONL(reader io.Reader) (Trace, error) {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 8*1024*1024)
	var records []collectorRecord
	line := 0
	for scanner.Scan() {
		line++
		if len(bytes.TrimSpace(scanner.Bytes())) == 0 {
			continue
		}
		record, err := decodeCollectorRecord(scanner.Bytes())
		if err != nil {
			return Trace{}, fmt.Errorf("trace JSONL line %d: %w", line, err)
		}
		records = append(records, record)
	}
	if err := scanner.Err(); err != nil {
		return Trace{}, fmt.Errorf("read trace JSONL: %w", err)
	}
	if len(records) == 0 {
		return Trace{}, fmt.Errorf("trace JSONL contains no records")
	}

	// The collector writes one run per stream in normal use. Rejecting mixed runs
	// avoids silently merging unrelated sessions while still allowing input order
	// to be arbitrary for deterministic offline replay.
	groups := make(map[string]*assembleRun)
	for _, record := range records {
		if record.SchemaVersion != TraceSchemaVersion {
			return Trace{}, fmt.Errorf("unsupported schema_version %q", record.SchemaVersion)
		}
		if record.RecordSchema != "" && record.RecordSchema != "collector-event-v1" {
			return Trace{}, fmt.Errorf("unsupported record_schema %q", record.RecordSchema)
		}
		if record.SuiteID == "" || record.TaskID == "" || record.RunID == "" || record.Condition == "" {
			return Trace{}, fmt.Errorf("collector record is missing run identity")
		}
		key := record.SuiteID + "\x00" + record.TaskID + "\x00" + record.RunID + "\x00" + record.Condition
		run := groups[key]
		if run == nil {
			run = &assembleRun{key: key, suiteID: record.SuiteID, taskID: record.TaskID, runID: record.RunID, condition: record.Condition, environment: record.Environment, sessionID: record.SessionID}
			groups[key] = run
		} else if run.environment != record.Environment || (run.sessionID != "" && record.SessionID != "" && run.sessionID != record.SessionID) {
			return Trace{}, fmt.Errorf("collector run %s has conflicting environment or session metadata", key)
		}
		run.records = append(run.records, record)
		if record.RecordType == "compaction_prepare" {
			run.prepares = append(run.prepares, record)
		}
		if record.RecordType == "compaction" && record.Completed {
			run.compactions = append(run.compactions, record)
		}
	}
	if len(groups) != 1 {
		return Trace{}, fmt.Errorf("expected exactly one run in event stream, found %d; assemble each run separately", len(groups))
	}
	var run *assembleRun
	for _, value := range groups {
		run = value
	}
	sort.Slice(run.records, func(i, j int) bool { return recordLess(run.records[i], run.records[j]) })
	sort.Slice(run.prepares, func(i, j int) bool { return recordLess(run.prepares[i], run.prepares[j]) })
	sort.Slice(run.compactions, func(i, j int) bool { return recordLess(run.compactions[i], run.compactions[j]) })

	trace := Trace{SchemaVersion: TraceSchemaVersion, SuiteID: run.suiteID, TaskID: run.taskID, RunID: run.runID, Condition: run.condition, Environment: run.environment, SessionID: run.sessionID, Compactions: []Compaction{}, ToolEvents: []ToolEvent{}, RecallProbes: []RecallProbe{}}
	for _, record := range run.compactions {
		if record.CheckpointID == nil || *record.CheckpointID == "" {
			return Trace{}, fmt.Errorf("completed compaction at event %d has no checkpoint_id", record.EventIndex)
		}
		prepare := latestPrepareBefore(run.prepares, record.EventIndex)
		source := record.SourceSHA256
		if source == "" && prepare != nil {
			source = prepare.MessagesSHA256
		}
		if source == "" {
			return Trace{}, fmt.Errorf("compaction %q has no source/messages hash", *record.CheckpointID)
		}
		summary := record.SummarySHA256
		if summary == "" {
			return Trace{}, fmt.Errorf("compaction %q has no summary_sha256; collector must hash the captured summary before assembly", *record.CheckpointID)
		}
		boundary := record.EventIndex + 1
		residual := (*string)(nil)
		trace.Compactions = append(trace.Compactions, Compaction{CheckpointID: *record.CheckpointID, EventIndex: boundary, SourceSHA256: source, SummarySHA256: summary, ResidualPackSHA256: residual, Namespace: "context", BeforeSessionID: record.BeforeSessionID, AfterSessionID: record.AfterSessionID, Timestamp: record.Timestamp})
	}
	for _, record := range run.records {
		if record.RecordType != "tool_event" {
			continue
		}
		event := ToolEvent{EventIndex: record.EventIndex, ToolName: record.ToolName, ToolInputSHA256: record.ToolInputSHA256, OutputSHA256: record.OutputSHA256, Outcome: record.Outcome, ErrorClass: record.ErrorClass, IsPlanning: record.IsPlanning, IsPolling: record.IsPolling, CheckpointID: record.CheckpointID, ToolCallID: record.ToolCallID, Timestamp: record.Timestamp, RecallQuery: record.RecallQuery, ReturnedMemoryIDs: append([]string(nil), record.ReturnedIDs...)}
		trace.ToolEvents = append(trace.ToolEvents, event)
	}
	if err := ValidateTrace(trace); err != nil {
		return Trace{}, fmt.Errorf("assembled trace validation: %w", err)
	}
	return trace, nil
}

func decodeCollectorRecord(data []byte) (collectorRecord, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return collectorRecord{}, err
	}
	getString := func(name string) string { var v string; _ = json.Unmarshal(raw[name], &v); return v }
	getBool := func(name string) bool { var v bool; _ = json.Unmarshal(raw[name], &v); return v }
	getInt := func(name string) int { var v int; _ = json.Unmarshal(raw[name], &v); return v }
	getPtrString := func(name string) *string {
		value, ok := raw[name]
		if !ok || bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return nil
		}
		var v string
		if json.Unmarshal(value, &v) != nil {
			return nil
		}
		return &v
	}
	var env Environment
	_ = json.Unmarshal(raw["environment"], &env)
	var ids []string
	_ = json.Unmarshal(raw["returned_memory_ids"], &ids)
	if ids == nil {
		ids = []string{}
	}
	return collectorRecord{SchemaVersion: getString("schema_version"), RecordType: getString("record_type"), RecordSchema: getString("record_schema"), SuiteID: getString("suite_id"), TaskID: getString("task_id"), RunID: getString("run_id"), Condition: getString("condition"), Environment: env, SessionID: getString("session_id"), EventIndex: getInt("event_index"), Timestamp: getString("timestamp"), ToolName: getString("tool_name"), ToolCallID: getString("tool_call_id"), ToolInputSHA256: getPtrString("tool_input_sha256"), OutputSHA256: getPtrString("output_sha256"), Outcome: Outcome(getString("outcome")), ErrorClass: getPtrString("error_class"), IsPlanning: getBool("is_planning"), IsPolling: getBool("is_polling"), CheckpointID: getPtrString("checkpoint_id"), RecallQuery: getPtrString("recall_query"), ReturnedIDs: ids, BeforeSessionID: getString("before_session_id"), AfterSessionID: getString("after_session_id"), SourceSHA256: getString("source_sha256"), SummarySHA256: getString("summary_sha256"), Completed: getBool("completed"), Reason: getString("reason"), MessagesSHA256: getString("messages_sha256")}, nil
}

func recordLess(a, b collectorRecord) bool {
	if a.EventIndex != b.EventIndex {
		return a.EventIndex < b.EventIndex
	}
	if a.RecordType != b.RecordType {
		return a.RecordType < b.RecordType
	}
	return a.Timestamp < b.Timestamp
}
func latestPrepareBefore(prepares []collectorRecord, index int) *collectorRecord {
	var found *collectorRecord
	for i := range prepares {
		if prepares[i].EventIndex < index && (found == nil || prepares[i].EventIndex > found.EventIndex) {
			found = &prepares[i]
		}
	}
	return found
}

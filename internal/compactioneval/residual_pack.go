package compactioneval

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

const ResidualPackSchemaVersion = 1

type ResidualPolicy struct {
	Name              string `json:"name"`
	Version           int    `json:"version"`
	MaxItems          int    `json:"max_items"`
	MaxExcerptBytes   int    `json:"max_excerpt_bytes"`
	MaxPackBytes      int    `json:"max_pack_bytes"`
	RedactionPolicyID string `json:"redaction_policy_id"`
}

func DefaultResidualPolicy() ResidualPolicy {
	return ResidualPolicy{Name: "nyawa.residual-pack", Version: 1, MaxItems: 16, MaxExcerptBytes: 768, MaxPackBytes: 16384, RedactionPolicyID: "deterministic-secret-redaction-v1"}
}

type ResidualSourceMessage struct {
	Index   int    `json:"index"`
	Role    string `json:"role"`
	Content string `json:"content"`
	Omitted bool   `json:"omitted"`
}
type ResidualToolEvent struct {
	EventIndex         int     `json:"event_index"`
	SourceMessageIndex int     `json:"source_message_index"`
	ToolName           string  `json:"tool_name"`
	ToolCallID         *string `json:"tool_call_id"`
	ToolInputSHA256    *string `json:"tool_input_sha256"`
	Outcome            Outcome `json:"outcome"`
	ErrorClass         *string `json:"error_class"`
	Retain             *bool   `json:"retain"`
}
type ResidualHint struct {
	SourceMessageIndex int    `json:"source_message_index"`
	Ordinal            int    `json:"ordinal"`
	Kind               string `json:"kind"`
	Retain             bool   `json:"retain"`
}
type ResidualPackInput struct {
	SessionKey                   string                  `json:"session_key"`
	OldSessionID                 string                  `json:"old_session_id"`
	NewSessionID                 string                  `json:"new_session_id"`
	ParentCheckpointID           *string                 `json:"parent_checkpoint_id"`
	Summary                      string                  `json:"summary"`
	CompactedThroughMessageIndex int                     `json:"compacted_through_message_index"`
	Messages                     []ResidualSourceMessage `json:"messages"`
	ToolEvents                   []ResidualToolEvent     `json:"tool_events"`
	Hints                        []ResidualHint          `json:"hints"`
	MetadataCoverage             string                  `json:"metadata_coverage"`
	Policy                       ResidualPolicy          `json:"policy"`
}

type ResidualPackItem struct {
	EntryID            string  `json:"entry_id"`
	Ordinal            int     `json:"ordinal"`
	SourceMessageIndex int     `json:"source_message_index"`
	Kind               string  `json:"kind"`
	ToolName           string  `json:"tool_name"`
	ToolCallID         *string `json:"tool_call_id"`
	ToolInputSHA256    *string `json:"tool_input_sha256"`
	Outcome            Outcome `json:"outcome"`
	ContentSHA256      string  `json:"content_sha256"`
	Excerpt            string  `json:"excerpt"`
	ExcerptTruncated   bool    `json:"excerpt_truncated"`
	SelectionReason    string  `json:"selection_reason"`
}
type residualPackSource struct {
	SourceSHA256                 string  `json:"source_sha256"`
	SummarySHA256                string  `json:"summary_sha256"`
	CompactedThroughMessageIndex int     `json:"compacted_through_message_index"`
	ParentCheckpointID           *string `json:"parent_checkpoint_id"`
}
type ResidualPackDocument struct {
	SchemaVersion    int                `json:"schema_version"`
	CheckpointID     string             `json:"checkpoint_id"`
	MetadataCoverage string             `json:"metadata_coverage"`
	Policy           ResidualPolicy     `json:"policy"`
	Source           residualPackSource `json:"source"`
	Items            []ResidualPackItem `json:"items"`
}
type ResidualPackResult struct {
	Mode          string               `json:"mode"`
	Enabled       bool                 `json:"enabled"`
	Disabled      bool                 `json:"disabled"`
	CheckpointID  string               `json:"checkpoint_id,omitempty"`
	PackSHA256    string               `json:"pack_sha256,omitempty"`
	CanonicalJSON []byte               `json:"-"`
	Pack          ResidualPackDocument `json:"pack"`
}
type ResidualBuildOptions struct {
	Enabled bool
	DryRun  bool
}

// BuildResidualPack is a pure shadow builder. It never writes a database,
// changes a message, injects a prompt, or activates runtime compaction.
func BuildResidualPack(input ResidualPackInput) (ResidualPackResult, error) {
	return BuildResidualPackWithOptions(input, ResidualBuildOptions{Enabled: true, DryRun: true})
}
func BuildResidualPackWithOptions(input ResidualPackInput, options ResidualBuildOptions) (ResidualPackResult, error) {
	if !options.Enabled {
		return ResidualPackResult{Mode: "disabled", Enabled: false, Disabled: true, Pack: ResidualPackDocument{SchemaVersion: ResidualPackSchemaVersion, Items: []ResidualPackItem{}}}, nil
	}
	policy := input.Policy
	if policy.Name == "" {
		policy = DefaultResidualPolicy()
	}
	if err := validateResidualInput(&input, &policy); err != nil {
		return ResidualPackResult{}, err
	}
	input.SessionKey = normalizeResidualText(input.SessionKey)
	input.OldSessionID = normalizeResidualText(input.OldSessionID)
	input.NewSessionID = normalizeResidualText(input.NewSessionID)
	input.Summary = normalizeResidualText(input.Summary)
	if input.ParentCheckpointID != nil {
		input.ParentCheckpointID = cloneString(input.ParentCheckpointID)
	}
	messages := make([]ResidualSourceMessage, len(input.Messages))
	copy(messages, input.Messages)
	for i := range messages {
		messages[i].Content = normalizeResidualText(messages[i].Content)
		messages[i].Role = normalizeResidualText(messages[i].Role)
	}
	sourceEnvelope := struct {
		SchemaVersion int                     `json:"schema_version"`
		Messages      []ResidualSourceMessage `json:"messages"`
		ToolEvents    []ResidualToolEvent     `json:"tool_events"`
		Hints         []ResidualHint          `json:"hints"`
	}{1, messages, append([]ResidualToolEvent(nil), input.ToolEvents...), append([]ResidualHint(nil), input.Hints...)}
	for i := range sourceEnvelope.ToolEvents {
		sourceEnvelope.ToolEvents[i].ToolName = normalizeResidualText(sourceEnvelope.ToolEvents[i].ToolName)
		sourceEnvelope.ToolEvents[i].ToolCallID = cloneString(sourceEnvelope.ToolEvents[i].ToolCallID)
		sourceEnvelope.ToolEvents[i].ErrorClass = cloneString(sourceEnvelope.ToolEvents[i].ErrorClass)
	}
	sort.Slice(sourceEnvelope.ToolEvents, func(i, j int) bool {
		return sourceEnvelope.ToolEvents[i].EventIndex < sourceEnvelope.ToolEvents[j].EventIndex
	})
	sort.Slice(sourceEnvelope.Hints, func(i, j int) bool {
		if sourceEnvelope.Hints[i].SourceMessageIndex != sourceEnvelope.Hints[j].SourceMessageIndex {
			return sourceEnvelope.Hints[i].SourceMessageIndex < sourceEnvelope.Hints[j].SourceMessageIndex
		}
		return sourceEnvelope.Hints[i].Ordinal < sourceEnvelope.Hints[j].Ordinal
	})
	sourceBytes, _ := json.Marshal(sourceEnvelope)
	sourceHash := digest(sourceBytes)
	summaryHash := digest([]byte(normalizeResidualText(input.Summary)))
	policyBytes, _ := json.Marshal(policy)
	policyHash := digest(policyBytes)
	identity := struct {
		SchemaVersion      int     `json:"schema_version"`
		SessionKey         string  `json:"session_key"`
		OldSessionID       string  `json:"old_session_id"`
		NewSessionID       string  `json:"new_session_id"`
		ParentCheckpointID *string `json:"parent_checkpoint_id"`
		SourceSHA256       string  `json:"source_sha256"`
		SummarySHA256      string  `json:"summary_sha256"`
		PolicySHA256       string  `json:"policy_sha256"`
	}{1, input.SessionKey, input.OldSessionID, input.NewSessionID, input.ParentCheckpointID, sourceHash, summaryHash, policyHash}
	identityBytes, _ := json.Marshal(identity)
	checkpointID := "ccp_v1_" + hexDigest(identityBytes)
	candidates := residualCandidates(messages, input.ToolEvents, input.Hints, policy)
	selected := make([]ResidualPackItem, 0, policy.MaxItems)
	seenSuccess := map[string]bool{}
	for _, candidate := range candidates {
		if candidate.kind == "tool_result" {
			key := candidate.item.ToolName + "\x00" + ptrValue(candidate.item.ToolInputSHA256) + "\x00" + candidate.item.ContentSHA256
			if seenSuccess[key] {
				continue
			}
			seenSuccess[key] = true
		}
		selected = append(selected, candidate.item)
		if len(selected) >= policy.MaxItems {
			break
		}
	}
	sort.Slice(selected, func(i, j int) bool {
		if selected[i].SourceMessageIndex != selected[j].SourceMessageIndex {
			return selected[i].SourceMessageIndex < selected[j].SourceMessageIndex
		}
		return selected[i].EntryID < selected[j].EntryID
	})
	for i := range selected {
		selected[i].Ordinal = i
	}
	doc := ResidualPackDocument{SchemaVersion: 1, CheckpointID: checkpointID, MetadataCoverage: input.MetadataCoverage, Policy: policy, Source: residualPackSource{SourceSHA256: sourceHash, SummarySHA256: summaryHash, CompactedThroughMessageIndex: input.CompactedThroughMessageIndex, ParentCheckpointID: input.ParentCheckpointID}, Items: selected}
	encoded, err := json.Marshal(doc)
	if err != nil {
		return ResidualPackResult{}, err
	}
	for len(encoded) > policy.MaxPackBytes && len(selected) > 0 {
		removeIndex := -1
		for i := len(candidates) - 1; i >= 0; i-- {
			for j := range selected {
				if selected[j].EntryID == candidates[i].item.EntryID {
					removeIndex = j
					break
				}
			}
			if removeIndex >= 0 {
				break
			}
		}
		if removeIndex < 0 {
			removeIndex = len(selected) - 1
		}
		selected = append(selected[:removeIndex], selected[removeIndex+1:]...)
		sort.Slice(selected, func(i, j int) bool {
			if selected[i].SourceMessageIndex != selected[j].SourceMessageIndex {
				return selected[i].SourceMessageIndex < selected[j].SourceMessageIndex
			}
			return selected[i].EntryID < selected[j].EntryID
		})
		for i := range selected {
			selected[i].Ordinal = i
		}
		doc.Items = selected
		encoded, err = json.Marshal(doc)
		if err != nil {
			return ResidualPackResult{}, err
		}
	}
	if len(encoded) > policy.MaxPackBytes {
		return ResidualPackResult{}, fmt.Errorf("fixed residual pack metadata exceeds max_pack_bytes=%d", policy.MaxPackBytes)
	}
	return ResidualPackResult{Mode: "shadow", Enabled: true, Disabled: false, CheckpointID: checkpointID, PackSHA256: digest(encoded), CanonicalJSON: encoded, Pack: doc}, nil
}

type residualCandidate struct {
	item       ResidualPackItem
	priority   int
	eventIndex int
	kind       string
}

func residualCandidates(messages []ResidualSourceMessage, events []ResidualToolEvent, hints []ResidualHint, policy ResidualPolicy) []residualCandidate {
	byIndex := map[int]ResidualSourceMessage{}
	for _, m := range messages {
		byIndex[m.Index] = m
	}
	out := []residualCandidate{}
	for _, event := range events {
		m, ok := byIndex[event.SourceMessageIndex]
		if !ok || !m.Omitted {
			continue
		}
		kind := "tool_result"
		priority := 3
		reason := "explicit_tool_metadata"
		if event.Outcome == OutcomeFailure || event.Outcome == OutcomeToolError || event.Outcome == OutcomeTimeout {
			kind = "tool_error"
			priority = 1
			reason = "explicit_tool_outcome"
		}
		item := makeResidualItem(m, event.ToolName, event.ToolCallID, event.ToolInputSHA256, event.Outcome, kind, reason, policy)
		out = append(out, residualCandidate{item, priority, event.EventIndex, kind})
	}
	for _, hint := range hints {
		m, ok := byIndex[hint.SourceMessageIndex]
		if !ok || !m.Omitted || !hint.Retain {
			continue
		}
		kind := hint.Kind
		item := makeResidualItem(m, "", nil, nil, OutcomeUnknown, kind, "explicit_retained_hint", policy)
		priority := 2
		if kind == "carry_forward" {
			priority = 4
		}
		out = append(out, residualCandidate{item, priority, hint.Ordinal, kind})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].priority != out[j].priority {
			return out[i].priority < out[j].priority
		}
		if out[i].item.SourceMessageIndex != out[j].item.SourceMessageIndex {
			return out[i].item.SourceMessageIndex > out[j].item.SourceMessageIndex
		}
		if out[i].eventIndex != out[j].eventIndex {
			return out[i].eventIndex > out[j].eventIndex
		}
		if out[i].kind != out[j].kind {
			return out[i].kind < out[j].kind
		}
		return out[i].item.ContentSHA256 < out[j].item.ContentSHA256
	})
	return out
}
func makeResidualItem(message ResidualSourceMessage, tool string, call, input *string, outcome Outcome, kind, reason string, policy ResidualPolicy) ResidualPackItem {
	content := normalizeResidualText(message.Content)
	excerpt, truncated := boundedRedactedExcerpt(content, policy.MaxExcerptBytes)
	item := ResidualPackItem{SourceMessageIndex: message.Index, Kind: kind, ToolName: normalizeResidualText(tool), ToolCallID: cloneString(call), ToolInputSHA256: cloneString(input), Outcome: outcome, ContentSHA256: digest([]byte(content)), Excerpt: excerpt, ExcerptTruncated: truncated, SelectionReason: reason}
	id := struct {
		SourceMessageIndex int     `json:"source_message_index"`
		Kind               string  `json:"kind"`
		ToolName           string  `json:"tool_name"`
		ToolCallID         *string `json:"tool_call_id"`
		ToolInputSHA256    *string `json:"tool_input_sha256"`
		Outcome            Outcome `json:"outcome"`
		ContentSHA256      string  `json:"content_sha256"`
		Excerpt            string  `json:"excerpt"`
		ExcerptTruncated   bool    `json:"excerpt_truncated"`
		SelectionReason    string  `json:"selection_reason"`
	}{item.SourceMessageIndex, item.Kind, item.ToolName, item.ToolCallID, item.ToolInputSHA256, item.Outcome, item.ContentSHA256, item.Excerpt, item.ExcerptTruncated, item.SelectionReason}
	data, _ := json.Marshal(id)
	item.EntryID = "rpi_v1_" + hexDigest(data)
	return item
}
func validateResidualInput(input *ResidualPackInput, policy *ResidualPolicy) error {
	if input.SessionKey == "" || input.OldSessionID == "" || input.NewSessionID == "" {
		return errors.New("residual input requires session_key and session IDs")
	}
	if input.MetadataCoverage != "none" && input.MetadataCoverage != "partial" && input.MetadataCoverage != "complete" {
		return errors.New("metadata_coverage must be none, partial, or complete")
	}
	if policy.Version < 1 || policy.MaxItems < 0 || policy.MaxExcerptBytes < 0 || policy.MaxPackBytes <= 0 {
		return errors.New("invalid residual policy bounds")
	}
	if input.Messages == nil {
		return errors.New("messages must be an array")
	}
	seenMsg := map[int]bool{}
	roles := map[int]string{}
	for _, m := range input.Messages {
		if m.Index < 0 || seenMsg[m.Index] {
			return errors.New("message indexes must be non-negative and unique")
		}
		seenMsg[m.Index] = true
		roles[m.Index] = strings.ToLower(normalizeResidualText(m.Role))
	}
	seenEvents := map[int]bool{}
	for _, event := range input.ToolEvents {
		if event.EventIndex < 0 || seenEvents[event.EventIndex] {
			return errors.New("tool event indexes must be non-negative and unique")
		}
		seenEvents[event.EventIndex] = true
		if roles[event.SourceMessageIndex] != "tool" {
			return fmt.Errorf("tool event %d references non-tool or unknown source message", event.EventIndex)
		}
		if event.ToolName == "" {
			return fmt.Errorf("tool event %d has empty tool_name", event.EventIndex)
		}
		if !validOutcome(event.Outcome) {
			return fmt.Errorf("tool event %d has invalid outcome", event.EventIndex)
		}
		if event.ToolInputSHA256 != nil && !validSHA256Digest(*event.ToolInputSHA256) {
			return fmt.Errorf("tool event %d has invalid tool_input_sha256", event.EventIndex)
		}
	}
	for _, hint := range input.Hints {
		if roles[hint.SourceMessageIndex] == "" {
			return errors.New("hint references unknown source message")
		}
		switch hint.Kind {
		case "decision", "constraint", "pending_action", "carry_forward":
		default:
			return fmt.Errorf("invalid residual hint kind %q", hint.Kind)
		}
	}
	return nil
}

var residualSecretPatterns = []*regexp.Regexp{regexp.MustCompile(`(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+`), regexp.MustCompile(`(?i)\b(?:api[_-]?key|access[_-]?key|secret|password|passwd|token|authorization)\b\s*[:=]\s*[^\s,;]+`), regexp.MustCompile(`\bAKIA[0-9A-Z]{16}\b`)}

func normalizeResidualText(value string) string {
	value = strings.ToValidUTF8(value, "�")
	value = strings.ReplaceAll(value, "\r\n", "\n")
	return strings.ReplaceAll(value, "\r", "\n")
}
func boundedRedactedExcerpt(value string, maxBytes int) (string, bool) {
	redacted := value
	for _, pattern := range residualSecretPatterns {
		redacted = pattern.ReplaceAllString(redacted, "[REDACTED]")
	}
	if maxBytes <= 0 {
		return "", len(redacted) > 0
	}
	if len([]byte(redacted)) <= maxBytes {
		return redacted, false
	}
	bytes := []byte(redacted)
	cut := maxBytes
	for cut > 0 && !utf8.RuneStart(bytes[cut]) {
		cut--
	}
	return string(bytes[:cut]), true
}
func digest(data []byte) string {
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:])
}
func hexDigest(data []byte) string { sum := sha256.Sum256(data); return hex.EncodeToString(sum[:]) }
func ptrValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}
func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	v := *value
	return &v
}

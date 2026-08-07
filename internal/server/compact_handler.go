package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/rezkyauliapratama/nyawa/internal/compact"
	"github.com/rezkyauliapratama/nyawa/internal/types"
)

const (
	defaultPreserveRecentN = 30
	defaultSegmentSize     = 40
	defaultSegmentOverlap  = 5
	defaultPruneMaxBytes   = 8192
	compactNamespace       = "context"
)

type compactRequest struct {
	Messages        []map[string]interface{} `json:"messages"`
	FocusTopic      string                   `json:"focus_topic,omitempty"`
	PreserveRecentN int                      `json:"preserve_recent_n,omitempty"`
	SegmentSize     int                      `json:"segment_size,omitempty"`
	SegmentOverlap  int                      `json:"segment_overlap,omitempty"`
	PruneMaxBytes   int                      `json:"prune_max_bytes,omitempty"`
	DryRun          bool                     `json:"dry_run,omitempty"`
	SessionKey      string                   `json:"session_key,omitempty"`
	OldSessionID    string                   `json:"old_session_id,omitempty"`
	NewSessionID    string                   `json:"new_session_id,omitempty"`
}

// handleCompact implements POST /v1/compact.
// Pipeline: parse -> Tier-1 prune -> segment -> summarize (LLM w/ deterministic
// fallback) -> store summary + compaction log -> return compact block + stats.
func (s *Server) handleCompact(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	var req compactRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON: " + err.Error()})
		return
	}
	if len(req.Messages) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "messages required"})
		return
	}

	preserveRecentN := req.PreserveRecentN
	if preserveRecentN <= 0 {
		preserveRecentN = defaultPreserveRecentN
	}
	segmentSize := req.SegmentSize
	if segmentSize <= 0 {
		segmentSize = defaultSegmentSize
	}
	segmentOverlap := req.SegmentOverlap
	if segmentOverlap < 0 {
		segmentOverlap = defaultSegmentOverlap
	}
	pruneMaxBytes := req.PruneMaxBytes
	if pruneMaxBytes <= 0 {
		pruneMaxBytes = defaultPruneMaxBytes
	}

	// Parse messages to internal format.
	rawMsgs := make([]map[string]interface{}, len(req.Messages))
	for i, m := range req.Messages {
		rawMsgs[i] = m
	}
	msgs := compact.FromOpenAI(rawMsgs)
	inputTokens := compact.TokensIn(msgs)

	// Tier-1: deterministic prune of oversized tool results.
	prunedMsgs, pruned := compact.PruneToolResults(msgs, pruneMaxBytes)

	// Segment (tail excluded, kept verbatim).
	var tail []compact.Message
	segments := compact.Segment(prunedMsgs, segmentSize, segmentOverlap, preserveRecentN)
	if len(prunedMsgs) > preserveRecentN {
		tail = prunedMsgs[len(prunedMsgs)-preserveRecentN:]
	} else {
		tail = nil
	}

	// No-op guard: if the transcript is too short to segment, return the
	// original messages unchanged so compaction never destroys context.
	if len(segments) == 0 {
		writeJSON(w, http.StatusOK, map[string]any{
			"summary_block": compact.ToOpenAIList(msgs),
			"memory_ids":    []string{},
			"stats": compact.CompactionStats{
				InputMessages:  len(msgs),
				OutputMessages: len(msgs),
				InputTokens:    inputTokens,
				OutputTokens:   inputTokens,
				ReductionPct:   0,
				Segments:       0,
			},
			"noop": true,
		})
		return
	}

	// Summarize each segment (LLM with deterministic fallback).
	summaries := compact.SummarizeSegments(segments, req.FocusTopic)
	summaryText := strings.Join(summaries, "\n\n---\n\n")

	// Build output block: boundary + summary + tail verbatim + continuation.
	summaryBlock := compact.BuildSummaryBlock(summaryText, req.SessionKey)
	outputMsgs := append([]compact.Message{}, summaryBlock...)
	if len(tail) > 0 {
		outputMsgs = append(outputMsgs, tail...)
	}
	outputTokens := compact.TokensIn(outputMsgs)

	reductionPct := 0.0
	if inputTokens > 0 {
		reductionPct = float64(inputTokens-outputTokens) / float64(inputTokens) * 100
		if reductionPct < 0 {
			reductionPct = 0
		}
	}

	stats := compact.CompactionStats{
		InputMessages:     len(msgs),
		OutputMessages:    len(outputMsgs),
		InputTokens:       inputTokens,
		OutputTokens:      outputTokens,
		ReductionPct:      reductionPct,
		Segments:          len(segments),
		PrunedToolResults: pruned,
	}

	memoryIDs := []string{}
	if !req.DryRun && s.store != nil {
		// Store summary memory.
		summaryID := fmt.Sprintf("summary_%d", time.Now().UnixNano())
		summaryMem := &types.Memory{
			ID:        summaryID,
			Content:   summaryText,
			Type:      types.TypeContext,
			Namespace: compactNamespace,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}
		if err := s.store.InsertMemory(summaryMem); err == nil {
			memoryIDs = append(memoryIDs, summaryID)
		}
		// Record compaction log.
		sessionKey := req.SessionKey
		if sessionKey == "" {
			sessionKey = req.OldSessionID
		}
		if sessionKey != "" {
			_ = s.store.LogCompaction(sessionKey, req.OldSessionID, req.NewSessionID, summaryID)
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"summary_block": compact.ToOpenAIList(outputMsgs),
		"memory_ids":    memoryIDs,
		"stats":         stats,
	})
}

package types

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

type MemoryType string

const (
	TypeDecision     MemoryType = "decision"
	TypeInsight      MemoryType = "insight"
	TypeProcedure    MemoryType = "procedure"
	TypeFact         MemoryType = "fact"
	TypePreference   MemoryType = "preference"
	TypeContext      MemoryType = "context"
	TypeNote         MemoryType = "note"
	TypeEvent        MemoryType = "event"
	TypeReference    MemoryType = "reference"
	TypeRule         MemoryType = "rule"
	TypeConversation MemoryType = "conversation"
)

// RetrievalFactor is the per-type multiplier recall ranking uses to lift durable,
// high-signal memories (rules, decisions, preferences) above ephemeral chatter.
// It is separate from Weight (which the legacy ranking still consumes) so the
// retrieval behaviour can change without rewriting on-disk type semantics.
//
// conversation is deliberately 0.0: raw transcripts are noise for retrieval and
// must never be boosted just for existing (the task's explicit requirement).
func (t MemoryType) RetrievalFactor() float64 {
	switch t {
	case TypeRule, TypeDecision:
		return 1.0
	case TypePreference:
		return 0.9
	case TypeProcedure:
		return 0.8
	case TypeFact, TypeInsight:
		return 0.7
	case TypeContext, TypeEvent:
		return 0.5
	case TypeConversation:
		return 0.0
	case TypeNote, TypeReference:
		return 0.3
	default:
		// memory, snapshot, handoff, incident, unknown: neutral-low, so bulk
		// imported noise does not outrank curated memories.
		return 0.3
	}
}

func (t MemoryType) Weight() float64 {
	switch t {
	case TypeDecision: return 1.0
	case TypeInsight: return 0.9
	case TypeProcedure: return 0.8
	case TypeFact: return 0.7
	case TypePreference: return 0.6
	case TypeContext: return 0.5
	case TypeNote: return 0.4
	case TypeEvent: return 0.4
	case TypeReference: return 0.3
	default: return 0.4
	}
}

func (t MemoryType) DecayHours() float64 {
	switch t {
	case TypeDecision: return 720
	case TypeInsight: return 1440
	case TypeProcedure: return 2160
	case TypeFact: return 4320
	case TypePreference: return 720
	case TypeContext: return 336
	case TypeNote: return 168
	case TypeEvent: return 2160
	case TypeReference: return 8760
	case TypeRule: return 2160
	default: return 168
	}
}

type Memory struct {
	ID          string     `json:"id"`
	Content     string     `json:"content"`
	Type        MemoryType `json:"type"`
	Namespace   string     `json:"namespace"`
	Importance  float64    `json:"importance"`
	AccessCount int        `json:"access_count"`
	Pinned      bool       `json:"pinned"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
	SupersededAt *time.Time `json:"superseded_at,omitempty"`
	Vector      []float32  `json:"-"`
	EdgeCount   int        `json:"edge_count,omitempty"`
}

type MemoryResult struct {
	Memory
	Score           float64 `json:"score"`
	RRFScore        float64 `json:"rrf_score"`
	TemporalBoost   float64 `json:"temporal_boost"`
	ImportanceBoost float64 `json:"importance_boost"`
	TypeBoost       float64 `json:"type_boost"`
	AccessBoost     float64 `json:"access_boost"`
	LengthBoost     float64 `json:"length_boost"`
	Rank            int     `json:"rank"`
}

func (r *MemoryResult) Reset() {
	r.ID = ""
	r.Content = ""
	r.Type = ""
	r.Namespace = ""
	r.Importance = 0
	r.AccessCount = 0
	r.Pinned = false
	r.CreatedAt = time.Time{}
	r.UpdatedAt = time.Time{}
	r.SupersededAt = nil
	r.Vector = r.Vector[:0]
	r.EdgeCount = 0
	r.Score = 0
	r.RRFScore = 0
	r.TemporalBoost = 0
	r.ImportanceBoost = 0
	r.TypeBoost = 0
	r.AccessBoost = 0
	r.LengthBoost = 0
	r.Rank = 0
}

type StoreQuery struct {
	QueryText    string     `json:"query"`
	Namespace    string     `json:"namespace,omitempty"`
	Limit        int        `json:"limit,omitempty"`
	TimeTravel   *time.Time `json:"time_travel,omitempty"`
	MinScore     float64    `json:"min_score,omitempty"`
	ExcludeTypes []string   `json:"exclude_types,omitempty"`
}

// CacheKey identifies a fully-resolved query. Every field that changes the
// result set must take part, otherwise a filtered query can be served from the
// cache entry of an unfiltered one (and vice versa).
func (q StoreQuery) CacheKey() string {
	var b strings.Builder
	b.WriteString(q.QueryText)
	b.WriteByte(0)
	b.WriteString(q.Namespace)
	b.WriteByte(0)
	b.WriteString(strconv.Itoa(q.Limit))
	if q.MinScore > 0 {
		b.WriteByte(0)
		b.WriteString(strconv.FormatFloat(q.MinScore, 'g', -1, 64))
	}
	if len(q.ExcludeTypes) > 0 {
		b.WriteByte(0)
		b.WriteString(strings.Join(q.ExcludeTypes, ","))
	}
	if q.TimeTravel != nil {
		b.WriteByte(0)
		b.WriteString(q.TimeTravel.UTC().Format(time.RFC3339Nano))
	}
	return b.String()
}

const DefaultQueryLimit = 20

type Vector []float32

func (v Vector) MarshalJSON() ([]byte, error) {
	vals := make([]float64, len(v))
	for i, f := range v {
		vals[i] = float64(f)
	}
	return json.Marshal(vals)
}

func (v *Vector) UnmarshalJSON(b []byte) error {
	var vals []float64
	if err := json.Unmarshal(b, &vals); err != nil {
		return err
	}
	*v = make(Vector, len(vals))
	for i, f := range vals {
		(*v)[i] = float32(f)
	}
	return nil
}

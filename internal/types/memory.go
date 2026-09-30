package types

import (
	"encoding/json"
	"strconv"
	"strings"
	"time"
)

type MemoryType string

const (
	TypeDecision   MemoryType = "decision"
	TypeInsight    MemoryType = "insight"
	TypeProcedure  MemoryType = "procedure"
	TypeFact       MemoryType = "fact"
	TypePreference MemoryType = "preference"
	TypeContext    MemoryType = "context"
	TypeNote       MemoryType = "note"
	TypeEvent      MemoryType = "event"
	TypeReference  MemoryType = "reference"
)

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
	Score          float64 `json:"score"`
	RRFScore       float64 `json:"rrf_score"`
	TemporalBoost  float64 `json:"temporal_boost"`
	ImportanceBoost float64 `json:"importance_boost"`
	Rank           int     `json:"rank"`
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

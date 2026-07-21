package dream

import (
	"database/sql"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
	"github.com/rezkyauliapratama/nyawa/internal/index"
)

type Engine struct {
	db, hnswPath, interval *sql.DB, string, time.Duration
	hnsw     *index.HNSW
	mu       sync.Mutex
	running  bool
	lastRun  time.Time
	stats    Stats
}

type Stats struct {
	LastRun, Evicted, Contradictions, Deduped, LinksCreated, Prioritized, SnapshotsCreated string, int, int, int, int, int, int
}

type Config struct {
	Interval, StaleDays, StaleMinAccess int
	ImportanceThreshold, DedupThreshold float64
}

func New(db *sql.DB, hnsw *index.HNSW, hnswPath string) *Engine {
	return &Engine{db: db, hnsw: hnsw, hnswPath: hnswPath}
}

func DefaultConfig() Config {
	return Config{StaleDays: 90, StaleMinAccess: 2, ImportanceThreshold: 0.3, DedupThreshold: 0.92}
}

func (e *Engine) Run(cfg Config) Stats {
	e.mu.Lock(); defer e.mu.Unlock()
	start := time.Now(); log.Print("Dream Cycle starting")
	s := Stats{}
	s.Evicted = e.phaseEvict(cfg)
	s.Contradictions = e.phaseContradiction()
	s.Deduped = e.phaseDedup(cfg)
	s.LinksCreated = e.phaseLink()
	s.Prioritized = e.phasePriority()
	s.SnapshotsCreated = e.phaseSnapshot()
	s.LastRun = time.Now().UTC().Format(time.RFC3339)
	e.stats = s
	log.Printf("Dream Cycle done (%v): ev=%d ct=%d de=%d lk=%d pr=%d sn=%d", time.Since(start), s.Evicted, s.Contradictions, s.Deduped, s.LinksCreated, s.Prioritized, s.SnapshotsCreated)
	return s
}

func (e *Engine) phaseEvict(cfg Config) int {
	cut := time.Now().AddDate(0, 0, -cfg.StaleDays).Format(time.RFC3339)
	r, _ := e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE superseded_at IS NULL AND created_at<? AND access_count<? AND importance<? AND pinned=0`, cut, cfg.StaleMinAccess, cfg.ImportanceThreshold)
	if r == nil { return 0 }
	n, _ := r.RowsAffected()
	if n > 0 { log.Printf("Dream evicted %d", n) }
	return int(n)
}

func (e *Engine) phaseContradiction() int {
	pairs := [][2]string{{"suka","tidak suka"},{"like","dislike"},{"prefer","avoid"},{"favorite","hate"},{"recommend","not recommend"},{"enable","disable"},{"use","don't use"}}
	found := 0
	for _, p := range pairs {
		ra, _ := e.db.Query(`SELECT id FROM memories WHERE content LIKE ? AND superseded_at IS NULL LIMIT 20`, "%"+p[0]+"%")
		if ra == nil { continue }
		var aids []string
		for ra.Next() { var id string; ra.Scan(&id); aids = append(aids, id) }
		ra.Close()
		rb, _ := e.db.Query(`SELECT id FROM memories WHERE content LIKE ? AND superseded_at IS NULL LIMIT 20`, "%"+p[1]+"%")
		if rb == nil { continue }
		var bids []string
		for rb.Next() { var id string; rb.Scan(&id); bids = append(bids, id) }
		rb.Close()
		for _, a := range aids { for _, b := range bids { if a != b { e.db.Exec(`UPDATE memories SET importance=MAX(0.3,importance*0.9) WHERE id IN(?,?)`, a, b); found++ }; if found >= 10 { break } }; if found >= 10 { break } }
		if found >= 10 { break }
	}
	if found > 0 { log.Printf("Dream contradictions: %d", found) }
	return found
}

func (e *Engine) phaseDedup(cfg Config) int {
	if e.hnsw == nil || e.hnsw.Size() < 2 { return 0 }
	r, _ := e.db.Query(`SELECT id,content FROM memories WHERE superseded_at IS NULL ORDER BY created_at ASC`)
	if r == nil { return 0 }
	type mc struct{ id, content string }
	var ms []mc
	for r.Next() { var m mc; r.Scan(&m.id, &m.content); ms = append(ms, m) }
	r.Close()
	if len(ms) < 2 { return 0 }
	d := 0
	for i := 0; i < len(ms)-1 && d < 5; i++ {
		wa := strings.Fields(strings.ToLower(ms[i].content))
		if len(wa) < 3 { continue }
		for j := i + 1; j < len(ms) && d < 5; j++ {
			wb := strings.Fields(strings.ToLower(ms[j].content))
			ov := float64(0)
			if len(wa) > 0 { s := map[string]bool{}; for _, w := range wb { s[w] = true }; m := 0; for _, w := range wa { if s[w] { m++ } }; ov = float64(m) / float64(len(wa)) }
			if ov > cfg.DedupThreshold {
				e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`, ms[j].id)
				d++
			}
		}
	}
	if d > 0 { log.Printf("Dream dedup: %d", d) }
	return d
}

func (e *Engine) phaseLink() int {
	r, _ := e.db.Query(`SELECT e1.entity_id, e2.entity_id, COUNT(*) as c FROM entity_edges e1 JOIN entity_edges e2 ON e1.memory_id=e2.memory_id AND e1.entity_id<e2.entity_id GROUP BY e1.entity_id,e2.entity_id HAVING c>=2 LIMIT 50`)
	if r == nil { return 0 }
	type p struct{ e1, e2, c int }
	var ps []p
	for r.Next() { var x p; r.Scan(&x.e1, &x.e2, &x.c); ps = append(ps, x) }
	r.Close()
	for _, x := range ps { e.db.Exec(`UPDATE entity_nodes SET access_count=access_count+1 WHERE id IN(?,?)`, x.e1, x.e2) }
	if len(ps) > 0 { log.Printf("Dream linked %d", len(ps)) }
	return len(ps)
}

func (e *Engine) phasePriority() int {
	r1, _ := e.db.Exec(`UPDATE memories SET importance=LEAST(1.0,importance+0.05) WHERE superseded_at IS NULL AND access_count>5 AND pinned=0`)
	r2, _ := e.db.Exec(`UPDATE memories SET importance=GREATEST(0.1,importance-0.02) WHERE superseded_at IS NULL AND access_count=0 AND created_at<datetime('now','-7 days') AND pinned=0`)
	n1, n2 := int64(0), int64(0)
	if r1 != nil { n1, _ = r1.RowsAffected() }
	if r2 != nil { n2, _ = r2.RowsAffected() }
	return int(n1 + n2)
}

func (e *Engine) phaseSnapshot() int {
	cut := time.Now().AddDate(0, 0, -30).Format(time.RFC3339)
	r, _ := e.db.Query(`SELECT id,content FROM memories WHERE superseded_at IS NULL AND created_at<? AND pinned=0 ORDER BY importance ASC LIMIT 5`, cut)
	if r == nil { return 0 }
	type oc struct{ id, content string }
	var old []oc
	for r.Next() { var o oc; r.Scan(&o.id, &o.content); old = append(old, o) }
	r.Close()
	if len(old) < 2 { return 0 }
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("📦 Snapshot of %d old memories:\n", len(old)))
	for _, o := range old { sb.WriteString(fmt.Sprintf("• %s\n", truncate(o.content, 120))); e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`, o.id) }
	sid := fmt.Sprintf("snap_%d", time.Now().UnixNano())
	e.db.Exec(`INSERT INTO memories(id,content,mem_type,namespace,importance,pinned,created_at,updated_at) VALUES(?,?,'snapshot','default',0.5,0,datetime('now'),datetime('now'))`, sid, sb.String())
	log.Printf("Dream snapshot: %s", sid)
	return 1
}

func truncate(s string, max int) string { if len(s) <= max { return s }; return s[:max-3]+"..." }

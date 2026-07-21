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
	db       *sql.DB
	hnsw     *index.HNSW
	hnswPath string
	mu       sync.Mutex
	running  bool
	interval time.Duration
	lastRun  time.Time
	stats    Stats
}

type Stats struct {
	LastRun          string `json:"last_run"`
	Evicted          int    `json:"evicted"`
	Contradictions   int    `json:"contradictions"`
	Deduped          int    `json:"deduped"`
	LinksCreated     int    `json:"links_created"`
	Prioritized      int    `json:"prioritized"`
	SnapshotsCreated int    `json:"snapshots_created"`
}

func New(db *sql.DB, hnsw *index.HNSW, hnswPath string) *Engine {
	return &Engine{db: db, hnsw: hnsw, hnswPath: hnswPath, interval: 1 * time.Hour}
}

type Config struct {
	Interval            time.Duration
	StaleDays           int
	StaleMinAccess      int
	ImportanceThreshold float64
	DedupThreshold      float64
}

func DefaultConfig() Config {
	return Config{Interval: 1 * time.Hour, StaleDays: 90, StaleMinAccess: 2, ImportanceThreshold: 0.3, DedupThreshold: 0.92}
}

func (e *Engine) Start(cfg Config) {
	e.mu.Lock()
	if e.running { e.mu.Unlock(); return }
	e.running = true
	e.interval = cfg.Interval
	e.mu.Unlock()
	go e.loop(cfg)
	log.Printf("Dream Cycle started (interval=%v)", e.interval)
}

func (e *Engine) Stop() { e.mu.Lock(); e.running = false; e.mu.Unlock(); log.Print("Dream Cycle stopped") }

func (e *Engine) Run(cfg Config) Stats {
	e.mu.Lock()
	defer e.mu.Unlock()
	start := time.Now()
	log.Print("Dream Cycle starting")
	s := Stats{}
	s.Evicted = e.phaseEvict(cfg)
	s.Contradictions = e.phaseContradiction()
	s.Deduped = e.phaseDedup(cfg)
	s.LinksCreated = e.phaseLink()
	s.Prioritized = e.phasePriority()
	s.SnapshotsCreated = e.phaseSnapshot()
	s.LastRun = time.Now().UTC().Format(time.RFC3339)
	e.lastRun = time.Now()
	e.stats = s
	log.Printf("Dream Cycle done in %v: ev=%d ct=%d de=%d lk=%d pr=%d sn=%d",
		time.Since(start), s.Evicted, s.Contradictions, s.Deduped, s.LinksCreated, s.Prioritized, s.SnapshotsCreated)
	return s
}

func (e *Engine) Stats() Stats { e.mu.Lock(); defer e.mu.Unlock(); return e.stats }
func (e *Engine) Running() bool { e.mu.Lock(); defer e.mu.Unlock(); return e.running }

func (e *Engine) loop(cfg Config) {
	for {
		time.Sleep(e.interval)
		e.mu.Lock()
		running := e.running
		e.mu.Unlock()
		if !running { return }
		e.Run(cfg)
	}
}

func (e *Engine) phaseEvict(cfg Config) int {
	cutoff := time.Now().AddDate(0, 0, -cfg.StaleDays).Format(time.RFC3339)
	res, err := e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE superseded_at IS NULL AND created_at<? AND access_count<? AND importance<? AND pinned=0`,
		cutoff, cfg.StaleMinAccess, cfg.ImportanceThreshold)
	if err != nil { return 0 }
	n, _ := res.RowsAffected()
	if n > 0 { log.Printf("Dream evicted %d stale", n) }
	return int(n)
}

var contradictPairs = [][2]string{
	{"suka", "tidak suka"}, {"like", "dislike"},
	{"prefer", "avoid"}, {"favorite", "hate"},
	{"recommend", "not recommend"}, {"enable", "disable"},
	{"use", "don't use"}, {"pakai", "jangan pakai"},
}

func (e *Engine) phaseContradiction() int {
	found := 0
	for _, pair := range contradictPairs {
		rows, err := e.db.Query(`SELECT id FROM memories WHERE content LIKE ? AND superseded_at IS NULL LIMIT 20`, "%"+pair[0]+"%")
		if err != nil { continue }
		var aids []string
		for rows.Next() { var id string; rows.Scan(&id); aids = append(aids, id) }
		rows.Close()
		if len(aids) == 0 { continue }

		rows, err = e.db.Query(`SELECT id FROM memories WHERE content LIKE ? AND superseded_at IS NULL LIMIT 20`, "%"+pair[1]+"%")
		if err != nil { continue }
		var bids []string
		for rows.Next() { var id string; rows.Scan(&id); bids = append(bids, id) }
		rows.Close()
		if len(bids) == 0 { continue }

		for _, a := range aids {
			for _, b := range bids {
				if a == b { continue }
				e.db.Exec(`UPDATE memories SET importance=MAX(0.3,importance*0.9) WHERE id IN(?,?)`, a, b)
				found++
				if found >= 20 { break }
			}
			if found >= 20 { break }
		}
		if found >= 20 { break }
	}
	if found > 0 { log.Printf("Dream found %d contradictions", found) }
	return found
}

func (e *Engine) phaseDedup(cfg Config) int {
	if e.hnsw == nil || e.hnsw.Size() < 2 { return 0 }

	rows, err := e.db.Query(`SELECT id, content FROM memories WHERE superseded_at IS NULL ORDER BY created_at ASC`)
	if err != nil { return 0 }
	defer rows.Close()

	type mc struct{ id, content string }
	var mems []mc
	for rows.Next() { var m mc; rows.Scan(&m.id, &m.content); mems = append(mems, m) }
	if len(mems) < 2 { return 0 }

	deduped := 0
	for i := 0; i < len(mems)-1 && deduped < 10; i++ {
		if mems[i].content == "" { continue }
		wordsA := strings.Fields(strings.ToLower(mems[i].content))
		if len(wordsA) < 3 { continue }

		for j := i + 1; j < len(mems) && deduped < 10; j++ {
			if mems[j].content == "" { continue }
			wordsB := strings.Fields(strings.ToLower(mems[j].content))
			overlap := countOverlap(wordsA, wordsB, len(wordsA))
			if overlap > cfg.DedupThreshold {
				e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`,
					mems[j].id)
				e.hnsw.Delete(mems[j].id)
				e.db.Exec(`UPDATE memories SET edge_count=edge_count+1 WHERE id=?`, mems[i].id)
				deduped++
				log.Printf("Dream dedup: %s -> %s (%.0f%%)", mems[j].id[:20], mems[i].id[:20], overlap*100)
			}
		}
	}
	return deduped
}

func countOverlap(a, b []string, totalA int) float64 {
	if totalA == 0 { return 0 }
	set := make(map[string]bool, len(b))
	for _, w := range b { set[w] = true }
	match := 0
	for _, w := range a { if set[w] { match++ } }
	return float64(match) / float64(totalA)
}

func (e *Engine) phaseLink() int {
	rows, err := e.db.Query(`SELECT e1.entity_id, e2.entity_id, COUNT(*) as c FROM entity_edges e1 JOIN entity_edges e2 ON e1.memory_id=e2.memory_id AND e1.entity_id<e2.entity_id GROUP BY e1.entity_id,e2.entity_id HAVING c>=2 ORDER BY c DESC LIMIT 50`)
	if err != nil { return 0 }
	type pair struct{ e1, e2, cnt int }
	var pairs []pair
	for rows.Next() { var p pair; rows.Scan(&p.e1, &p.e2, &p.cnt); pairs = append(pairs, p) }
	rows.Close()

	linked := 0
	for _, p := range pairs {
		e.db.Exec(`UPDATE entity_nodes SET access_count=access_count+1 WHERE id IN(?,?)`, p.e1, p.e2)
		linked++
	}
	if linked > 0 { log.Printf("Dream linked %d entity pairs", linked) }
	return linked
}

func (e *Engine) phasePriority() int {
	r1, _ := e.db.Exec(`UPDATE memories SET importance=LEAST(1.0,importance+0.05) WHERE superseded_at IS NULL AND access_count>5 AND pinned=0`)
	r2, _ := e.db.Exec(`UPDATE memories SET importance=GREATEST(0.1,importance-0.02) WHERE superseded_at IS NULL AND access_count=0 AND created_at<datetime('now','-7 days') AND pinned=0`)
	var n1, n2 int64
	if r1 != nil { n1, _ = r1.RowsAffected() }
	if r2 != nil { n2, _ = r2.RowsAffected() }
	if n1+n2 > 0 { log.Printf("Dream priority: %d boosted, %d decayed", n1, n2) }
	return int(n1 + n2)
}

func (e *Engine) phaseSnapshot() int {
	cutoff := time.Now().AddDate(0, 0, -30).Format(time.RFC3339)
	rows, err := e.db.Query(`SELECT id,content FROM memories WHERE superseded_at IS NULL AND created_at<? AND pinned=0 ORDER BY importance ASC LIMIT 5`, cutoff)
	if err != nil { return 0 }
	type oc struct{ id, content string }
	var old []oc
	for rows.Next() { var o oc; rows.Scan(&o.id, &o.content); old = append(old, o) }
	rows.Close()

	if len(old) < 2 { return 0 }

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Snapshot of %d old memories:\\n", len(old)))
	for _, o := range old {
		sb.WriteString(fmt.Sprintf("* %s\\n", truncate(o.content, 120)))
		e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`, o.id)
	}

	snapID := fmt.Sprintf("snap_%d", time.Now().UnixNano())
	_, err = e.db.Exec(`INSERT INTO memories(id,content,mem_type,namespace,importance,pinned,created_at,updated_at) VALUES(?,?,'snapshot','default',0.5,0,datetime('now'),datetime('now'))`,
		snapID, sb.String())
	if err != nil { return 0 }
	log.Printf("Dream snapshot: %s (%d consolidated)", snapID, len(old))
	return 1
}

func truncate(s string, max int) string {
	if len(s) <= max { return s }
	return s[:max-3] + "..."
}

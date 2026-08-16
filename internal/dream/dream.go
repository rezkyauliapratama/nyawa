package dream

import (
	"database/sql"
	"fmt"
	"log"
	"math/rand"
	"strings"
	"sync"
	"time"
	"github.com/rezkyauliapratama/nyawa/internal/graph"
	"github.com/rezkyauliapratama/nyawa/internal/index"
)

type Engine struct {
	db         *sql.DB
	hnsw       *index.HNSW
	hnswPath   string
	graphStore *graph.Store
	mu         sync.Mutex
	running    bool
	interval   time.Duration
	lastRun    time.Time
	stats      Stats
}

type Stats struct {
	LastRun          string `json:"last_run"`
	Evicted          int    `json:"evicted"`
	Contradictions   int    `json:"contradictions"`
	Deduped          int    `json:"deduped"`
	LinksCreated     int    `json:"links_created"`
	Prioritized      int    `json:"prioritized"`
	SnapshotsCreated int    `json:"snapshots_created"`
	GraphBuilt       int    `json:"graph_built"`
	GraphNodes       int    `json:"graph_nodes"`
	GraphEdges       int    `json:"graph_edges"`
}

func New(db *sql.DB, hnsw *index.HNSW, hnswPath string) *Engine {
	return &Engine{db: db, hnsw: hnsw, hnswPath: hnswPath, interval: 1 * time.Hour}
}

func (e *Engine) SetGraphStore(gs *graph.Store) { e.graphStore = gs }

type Config struct {
	Interval            time.Duration
	StaleDays           int
	StaleMinAccess      int
	ImportanceThreshold float64
	DedupThreshold      float64
}

func DefaultConfig() Config {
	return Config{Interval: 2 * time.Hour, StaleDays: 90, StaleMinAccess: 2, ImportanceThreshold: 0.3, DedupThreshold: 0.92}
}

func (e *Engine) Start(cfg Config) {
	e.mu.Lock()
	if e.running { e.mu.Unlock(); return }
	e.running = true; e.interval = cfg.Interval
	e.mu.Unlock()
	go e.loop(cfg)
	log.Printf("Dream Cycle started (interval=%v)", e.interval)
}

func (e *Engine) Stop() { e.mu.Lock(); e.running = false; e.mu.Unlock(); log.Print("Dream Cycle stopped") }

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
	s.GraphBuilt, s.GraphNodes, s.GraphEdges = e.phaseGraphBuild()
	s.LastRun = time.Now().UTC().Format(time.RFC3339)
	e.lastRun = time.Now(); e.stats = s
	log.Printf("Dream Cycle done in %v: ev=%d ct=%d de=%d lk=%d pr=%d sn=%d gb=%d(nd=%d ed=%d)",
		time.Since(start), s.Evicted, s.Contradictions, s.Deduped, s.LinksCreated, s.Prioritized, s.SnapshotsCreated,
		s.GraphBuilt, s.GraphNodes, s.GraphEdges)
	return s
}

func (e *Engine) Stats() Stats { e.mu.Lock(); defer e.mu.Unlock(); return e.stats }
func (e *Engine) Running() bool { e.mu.Lock(); defer e.mu.Unlock(); return e.running }

func (e *Engine) loop(cfg Config) {
	// Random phase offset (0-15m) on first sleep so Dream Cycle doesn't always
	// collide with the 30-minute trading cron's write window (was the cause of
	// recurring "database is locked" store failures).
	time.Sleep(time.Duration(rand.Int63n(int64(15 * time.Minute))))
	for {
		time.Sleep(e.interval)
		e.mu.Lock()
		running := e.running
		e.mu.Unlock()
		if !running { return }
		e.Run(cfg)
		// Dream Cycle is the heaviest writer; checkpoint the WAL right after it
		// finishes so the WAL never balloons (a 200MB WAL caused every write to
		// slow down and "database is locked" errors on concurrent stores).
		if _, err := e.db.Exec(`PRAGMA wal_checkpoint(TRUNCATE)`); err != nil {
			log.Printf("Dream WAL checkpoint: %v", err)
		} else {
			log.Printf("Dream WAL checkpoint: truncated")
		}
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
				if _, err := e.db.Exec(`UPDATE memories SET importance=MAX(0.3,importance*0.9) WHERE id IN(?,?)`, a, b); err != nil {
					log.Printf("Dream contradiction: importance update failed: %v", err)
					continue
				}
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

	// Precompute token sets once per memory. Computing strings.Fields inside
	// the O(n^2) loop is wasteful: for n memories it re-tokenizes ~n^2/2
	// times. With 3k+ memories that dominates runtime and memory.
	words := make([][]string, len(mems))
	wordSets := make([]map[string]bool, len(mems))
	for i := range mems {
		w := strings.Fields(strings.ToLower(mems[i].content))
		if len(w) < 3 { continue }
		words[i] = w
		set := make(map[string]bool, len(w))
		for _, tok := range w { set[tok] = true }
		wordSets[i] = set
	}

	deduped := 0
	for i := 0; i < len(mems)-1 && deduped < 10; i++ {
		if words[i] == nil { continue }
		wordsA := words[i]
		for j := i + 1; j < len(mems) && deduped < 10; j++ {
			if words[j] == nil { continue }
			// Quick pre-check: overlap can't exceed min(lenA, lenB)
			setB := wordSets[j]
			minLen := len(wordsA)
			if len(setB) < minLen { minLen = len(setB) }
			if float64(minLen)/float64(len(wordsA)) < cfg.DedupThreshold { continue }
			overlap := countOverlapSet(wordsA, setB, len(wordsA))
			if overlap > cfg.DedupThreshold {
				if _, err := e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`, mems[j].id); err != nil {
					log.Printf("Dream dedup: skip %s (write busy: %v)", truncate(mems[j].id, 20), err)
					continue
				}
				e.hnsw.Delete(mems[j].id)
				if _, err := e.db.Exec(`UPDATE memories SET edge_count=edge_count+1 WHERE id=?`, mems[i].id); err != nil {
					log.Printf("Dream dedup: edge_count update failed: %v", err)
				}
				deduped++
				log.Printf("Dream dedup: %s → %s (%.0f%%)", truncate(mems[j].id, 20), truncate(mems[i].id, 20), overlap*100)
			}
		}
	}
	return deduped
}

// countOverlapSet returns |a ∩ setB| / totalA without allocating a new map
// per call. setB is the precomputed token set of memory B.
func countOverlapSet(a []string, setB map[string]bool, totalA int) float64 {
	if totalA == 0 { return 0 }
	match := 0
	for _, w := range a { if setB[w] { match++ } }
	return float64(match) / float64(totalA)
}

func (e *Engine) phaseLink() int {
	rows, err := e.db.Query(`SELECT e1.entity_id, e2.entity_id, COUNT(*) as c FROM entity_edges e1 JOIN entity_edges e2 ON e1.memory_id=e2.memory_id AND e1.entity_id<e2.entity_id GROUP BY e1.entity_id,e2.entity_id HAVING c>=2 ORDER BY c DESC LIMIT 50`)
	if err != nil { return 0 }
	type pair struct{ a, b int }
	var pairs []pair
	for rows.Next() {
		var p pair
		var cnt int
		rows.Scan(&p.a, &p.b, &cnt)
		pairs = append(pairs, p)
	}
	rows.Close()
	linked := 0
	for _, p := range pairs {
		// Exec AFTER rows are closed: with SetMaxOpenConns(1) the SELECT pins
		// the single connection, and an Exec inside rows.Next() would wait for
		// that same connection forever (deadlock).
		e.db.Exec(`UPDATE entity_nodes SET access_count=access_count+1 WHERE id IN(?,?)`, p.a, p.b)
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
	var old []struct{ id, content string }
	for rows.Next() { var o struct{ id, content string }; rows.Scan(&o.id, &o.content); old = append(old, o) }
	rows.Close()
	if len(old) < 2 { return 0 }
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("📦 Snapshot of %d old memories:\n", len(old)))
	for _, o := range old {
		sb.WriteString(fmt.Sprintf("• %s\n", truncate(o.content, 120)))
		e.db.Exec(`UPDATE memories SET superseded_at=datetime('now') WHERE id=?`, o.id)
	}
	snapID := fmt.Sprintf("snap_%d", time.Now().UnixNano())
	e.db.Exec(`INSERT INTO memories(id,content,mem_type,namespace,importance,pinned,created_at,updated_at) VALUES(?,?,'snapshot','default',0.5,0,datetime('now'),datetime('now'))`,
		snapID, sb.String())
	log.Printf("Dream snapshot: %s (%d consolidated)", snapID, len(old))
	return 1
}

func (e *Engine) phaseGraphBuild() (built, nodes, edges int) {
	if e.graphStore == nil {
		return 0, 0, 0
	}
	// Re-extract entities with the current classifier dictionary so new
	// aliases (DeepSeek, AWS Bedrock, MCP, ...) backfill older memories,
	// then rebuild co-occurrence edges from the enriched graph.
	re, err := e.graphStore.ReextractEntities()
	if err != nil {
		log.Printf("Dream re-extract error: %v", err)
	} else if re.MemoriesScanned > 0 {
		log.Printf("Dream re-extract: scanned=%d entities=%d edges=%d",
			re.MemoriesScanned, re.EntitiesAdded, re.EdgesAdded)
	}
	stats, err := e.graphStore.RebuildGraph()
	if err != nil {
		log.Printf("Dream graph build error: %v", err)
		return 0, 0, 0
	}
	log.Printf("Dream graph build: scanned=%d pairs=%d updated=%d pruned=%d nodes=%d edges=%d avgdeg=%.2f",
		stats.MemoriesScanned, stats.PairsCounted, stats.EdgesUpdated, stats.EdgesPruned,
		stats.NodesTotal, stats.EdgesTotal, stats.AvgDegree)
	return stats.EdgesUpdated, stats.NodesTotal, stats.EdgesTotal
}

func truncate(s string, max int) string {
	if len(s) <= max { return s }
	return s[:max-3] + "..."
}

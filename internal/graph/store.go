package graph

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
	"github.com/rezkyauliapratama/nyawa/internal/extract"
)

type Store struct{ db *sql.DB }

func NewStore(db *sql.DB) (*Store, error) {
	s := &Store{db: db}
	_, err := s.db.Exec(`
	CREATE TABLE IF NOT EXISTS entity_nodes (
		id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
		category TEXT NOT NULL DEFAULT 'unknown', created_at TEXT DEFAULT (datetime('now')),
		access_count INTEGER NOT NULL DEFAULT 0
	);
	CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_nodes(name);
	CREATE TABLE IF NOT EXISTS entity_edges (
		id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL,
		entity_id INTEGER NOT NULL, weight REAL DEFAULT 1.0,
		created_at TEXT DEFAULT (datetime('now')),
		UNIQUE(memory_id, entity_id)
	);
	CREATE INDEX IF NOT EXISTS idx_edge_memory ON entity_edges(memory_id);
	CREATE INDEX IF NOT EXISTS idx_edge_entity ON entity_edges(entity_id);
	`)
	return s, err
}

type Entity struct{ ID int; Name, Category string }
type RelatedMemory struct{ MemoryID, EntityName string; Score float64 }

func (s *Store) InsertMemoryEntities(memoryID string, entities extract.Entities) (int, error) {
	type item struct{ name, cat string }
	var items []item
	for _, e := range entities.Tech { items = append(items, item{e, "tech"}) }
	for _, e := range entities.People { items = append(items, item{e, "person"}) }
	for _, e := range entities.URLs {
		if len(e) > 64 { e = e[:64] }
		items = append(items, item{e, "url"})
	}
	if len(items) == 0 { return 0, nil }
	count := 0
	for _, it := range items {
		s.db.Exec(`INSERT INTO entity_nodes(name,category) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET access_count=access_count+1`, strings.TrimSpace(it.name), it.cat)
		var eid int
		s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name=?`, strings.TrimSpace(it.name)).Scan(&eid)
		if eid == 0 { continue }
		if _, err := s.db.Exec(`INSERT OR IGNORE INTO entity_edges(memory_id,entity_id,weight,created_at) VALUES(?,?,?,?)`, memoryID, eid, 1.0, time.Now().UTC().Format(time.RFC3339)); err == nil { count++ }
	}
	return count, nil
}

func (s *Store) FindRelatedMemories(memoryID string, limit int) ([]RelatedMemory, error) {
	rows, err := s.db.Query(`SELECT DISTINCT e2.memory_id, en.name, COUNT(*) as w FROM entity_edges e1 JOIN entity_edges e2 ON e1.entity_id=e2.entity_id AND e2.memory_id!=e1.memory_id JOIN entity_nodes en ON e1.entity_id=en.id WHERE e1.memory_id=? GROUP BY e2.memory_id ORDER BY w DESC LIMIT ?`, memoryID, limit)
	if err != nil { return nil, err }
	defer rows.Close()
	var r []RelatedMemory
	for rows.Next() {
		var rm RelatedMemory; var w int
		if rows.Scan(&rm.MemoryID, &rm.EntityName, &w) == nil { rm.Score = float64(w) * 0.1; r = append(r, rm) }
	}
	return r, nil
}

func (s *Store) GetEntitiesForMemory(memoryID string) ([]Entity, error) {
	rows, err := s.db.Query(`SELECT en.id,en.name,en.category FROM entity_edges ee JOIN entity_nodes en ON ee.entity_id=en.id WHERE ee.memory_id=? ORDER BY en.name`, memoryID)
	if err != nil { return nil, err }
	defer rows.Close()
	var e []Entity
	for rows.Next() { var en Entity; if rows.Scan(&en.ID, &en.Name, &en.Category) == nil { e = append(e, en) } }
	return e, nil
}

func (s *Store) SearchByEntityName(entityName string, limit int) ([]string, error) {
	rows, err := s.db.Query(`SELECT ee.memory_id FROM entity_edges ee JOIN entity_nodes en ON ee.entity_id=en.id WHERE en.name LIKE ? ORDER BY ee.weight DESC LIMIT ?`, "%"+entityName+"%", limit)
	if err != nil { return nil, err }
	defer rows.Close()
	var ids []string
	for rows.Next() { var id string; rows.Scan(&id); ids = append(ids, id) }
	return ids, nil
}

func (s *Store) Stats() (map[string]any, error) {
	var n, e int
	s.db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&n)
	s.db.QueryRow(`SELECT COUNT(*) FROM entity_edges`).Scan(&e)
	return map[string]any{"entity_nodes": n, "entity_edges": e}, nil
}

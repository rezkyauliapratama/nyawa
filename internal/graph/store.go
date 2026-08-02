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
	if err := s.migrate(); err != nil { return nil, fmt.Errorf("graph migrate: %w", err) }
	return s, nil
}

func (s *Store) migrate() error {
	_, err := s.db.Exec(`
	CREATE TABLE IF NOT EXISTS entity_nodes (
		id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
		category TEXT NOT NULL DEFAULT 'unknown', created_at TEXT NOT NULL DEFAULT (datetime('now')),
		access_count INTEGER NOT NULL DEFAULT 0);
	CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_nodes(name);
	CREATE TABLE IF NOT EXISTS entity_edges (
		id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL, entity_id INTEGER NOT NULL,
		weight REAL NOT NULL DEFAULT 1.0, created_at TEXT NOT NULL DEFAULT (datetime('now')),
		FOREIGN KEY (entity_id) REFERENCES entity_nodes(id), UNIQUE(memory_id, entity_id));
	CREATE INDEX IF NOT EXISTS idx_edge_memory ON entity_edges(memory_id);
	CREATE INDEX IF NOT EXISTS idx_edge_entity ON entity_edges(entity_id);
	CREATE TABLE IF NOT EXISTS entity_entity_edges (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		source_id INTEGER NOT NULL REFERENCES entity_nodes(id),
		target_id INTEGER NOT NULL REFERENCES entity_nodes(id),
		rel_type TEXT NOT NULL DEFAULT 'related_to',
		weight REAL NOT NULL DEFAULT 1.0,
		created_at TEXT NOT NULL DEFAULT (datetime('now')),
		UNIQUE(source_id, target_id, rel_type));
	CREATE INDEX IF NOT EXISTS idx_ee_source ON entity_entity_edges(source_id);
	CREATE INDEX IF NOT EXISTS idx_ee_target ON entity_entity_edges(target_id);
	CREATE INDEX IF NOT EXISTS idx_ee_type ON entity_entity_edges(rel_type);
	CREATE TABLE IF NOT EXISTS entity_pair_counts (
		source_id INTEGER NOT NULL,
		target_id INTEGER NOT NULL,
		count INTEGER NOT NULL DEFAULT 1,
		UNIQUE(source_id, target_id));
	CREATE INDEX IF NOT EXISTS idx_epc_source ON entity_pair_counts(source_id);
	CREATE INDEX IF NOT EXISTS idx_epc_target ON entity_pair_counts(target_id);`)
	return err
}

type Entity struct{ ID int; Name string; Category string }

func (s *Store) InsertMemoryEntities(memoryID string, entities extract.Entities) (int, error) {
	var names, categories []string
	for _, e := range entities.Tech { names = append(names, e); categories = append(categories, "tech") }
	for _, e := range entities.People { names = append(names, e); categories = append(categories, "person") }
	for _, e := range entities.URLs {
		if len(e) > 64 { e = e[:64] }
		names = append(names, e); categories = append(categories, "url")
	}
	if len(names) == 0 { return 0, nil }
	count := 0
	entityIDs := make(map[int]bool)
	for i, name := range names {
		if name == "" { continue }
		name = strings.TrimSpace(name)
		s.db.Exec(`INSERT INTO entity_nodes (name, category) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET access_count = access_count + 1`, name, categories[i])
		var entityID int
		s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ?`, name).Scan(&entityID)
		if entityID == 0 { continue }
		entityIDs[entityID] = true
		if _, err := s.db.Exec(`INSERT OR IGNORE INTO entity_edges (memory_id, entity_id, weight, created_at) VALUES (?, ?, ?, ?)`, memoryID, entityID, 1.0, time.Now().UTC().Format(time.RFC3339)); err == nil { count++ }
	}
	if len(entityIDs) >= 2 {
		ids := make([]int, 0, len(entityIDs))
		for id := range entityIDs { ids = append(ids, id) }
		for i := 0; i < len(ids); i++ {
			for j := i + 1; j < len(ids); j++ {
				sid, tid := ids[i], ids[j]
				if sid > tid { sid, tid = tid, sid }
				if _, err := s.db.Exec(`INSERT INTO entity_pair_counts (source_id, target_id, count) VALUES (?, ?, 1) ON CONFLICT(source_id, target_id) DO UPDATE SET count = count + 1`, sid, tid); err != nil {
					continue
				}
				var pairCount int
				if err := s.db.QueryRow(`SELECT count FROM entity_pair_counts WHERE source_id = ? AND target_id = ?`, sid, tid).Scan(&pairCount); err != nil {
					continue
				}
				if pairCount >= 2 {
					s.db.Exec(`INSERT INTO entity_entity_edges (source_id, target_id, rel_type, weight) VALUES (?, ?, 'related_to', ?) ON CONFLICT(source_id, target_id, rel_type) DO UPDATE SET weight = ?`, sid, tid, float64(pairCount), float64(pairCount))
				}
			}
		}
	}
	return count, nil
}

type RelatedMemory struct{ MemoryID, EntityName string; Score float64 }

func (s *Store) FindRelatedMemories(memoryID string, limit int) ([]RelatedMemory, error) {
	rows, err := s.db.Query(`SELECT DISTINCT e2.memory_id, en.name, COUNT(*) as weight FROM entity_edges e1 JOIN entity_edges e2 ON e1.entity_id = e2.entity_id AND e2.memory_id != e1.memory_id JOIN entity_nodes en ON e1.entity_id = en.id WHERE e1.memory_id = ? GROUP BY e2.memory_id ORDER BY weight DESC LIMIT ?`, memoryID, limit)
	if err != nil { return nil, fmt.Errorf("related: %w", err) }
	defer rows.Close()
	var results []RelatedMemory
	for rows.Next() {
		var rm RelatedMemory; var weight int
		if err := rows.Scan(&rm.MemoryID, &rm.EntityName, &weight); err != nil { continue }
		rm.Score = float64(weight) * 0.1; results = append(results, rm)
	}
	return results, nil
}

func (s *Store) GetEntitiesForMemory(memoryID string) ([]Entity, error) {
	rows, err := s.db.Query(`SELECT en.id, en.name, en.category FROM entity_edges ee JOIN entity_nodes en ON ee.entity_id = en.id WHERE ee.memory_id = ? ORDER BY en.name`, memoryID)
	if err != nil { return nil, err }
	defer rows.Close()
	var entities []Entity
	for rows.Next() { var e Entity; rows.Scan(&e.ID, &e.Name, &e.Category); entities = append(entities, e) }
	return entities, nil
}

func (s *Store) SearchByEntityName(entityName string, limit int) ([]string, error) {
	rows, err := s.db.Query(`SELECT ee.memory_id FROM entity_edges ee JOIN entity_nodes en ON ee.entity_id = en.id WHERE en.name LIKE ? ORDER BY ee.weight DESC LIMIT ?`, "%"+entityName+"%", limit)
	if err != nil { return nil, err }
	defer rows.Close()
	var ids []string
	for rows.Next() { var id string; rows.Scan(&id); ids = append(ids, id) }
	return ids, nil
}

func (s *Store) Stats() (map[string]any, error) {
	var nodes, edges, eeEdges int
	s.db.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&nodes)
	s.db.QueryRow(`SELECT COUNT(*) FROM entity_edges`).Scan(&edges)
	s.db.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&eeEdges)
	return map[string]any{"entity_nodes": nodes, "entity_edges": edges, "entity_entity_edges": eeEdges}, nil
}

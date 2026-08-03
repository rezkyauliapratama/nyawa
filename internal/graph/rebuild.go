package graph

import (
	"fmt"
)

// RebuildStats reports what the batch graph rebuild did.
type RebuildStats struct {
	MemoriesScanned int
	PairsCounted    int
	EdgesUpdated    int
	EdgesPruned     int
	NodesTotal      int
	EdgesTotal      int
	AvgDegree       float64
}

// RebuildGraph rebuilds all co-occurrence related_to edges from entity_pair_counts,
// preserves typed edges (works_at, uses, located_in, part_of), prunes stale
// related_to edges, and returns detailed stats. The operation is transaction-safe.
func (s *Store) RebuildGraph() (RebuildStats, error) {
	var stats RebuildStats

	tx, err := s.db.Begin()
	if err != nil {
		return stats, fmt.Errorf("rebuild begin tx: %w", err)
	}
	defer tx.Rollback()

	// Snapshot how many related_to edges exist before the rebuild (for EdgesPruned)
	tx.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges WHERE rel_type = 'related_to'`).Scan(&stats.EdgesPruned)

	// Step 1: clear entity_pair_counts so we rebuild from scratch
	if _, err := tx.Exec(`DELETE FROM entity_pair_counts`); err != nil {
		return stats, fmt.Errorf("rebuild delete pair_counts: %w", err)
	}

	// Step 2: delete only related_to edges (preserve typed edges)
	if _, err := tx.Exec(`DELETE FROM entity_entity_edges WHERE rel_type = 'related_to'`); err != nil {
		return stats, fmt.Errorf("rebuild delete related_to: %w", err)
	}

	// Step 3: count distinct memories that have entities
	tx.QueryRow(`SELECT COUNT(DISTINCT memory_id) FROM entity_edges`).Scan(&stats.MemoriesScanned)

	// Step 4: rebuild pair counts from all memories
	rows, err := tx.Query(`SELECT memory_id, entity_id FROM entity_edges ORDER BY memory_id, entity_id`)
	if err != nil {
		return stats, fmt.Errorf("rebuild query entity_edges: %w", err)
	}

	memEntities := make(map[string][]int)
	for rows.Next() {
		var memID string
		var entityID int
		if err := rows.Scan(&memID, &entityID); err != nil {
			rows.Close()
			return stats, fmt.Errorf("rebuild scan: %w", err)
		}
		memEntities[memID] = append(memEntities[memID], entityID)
	}
	rows.Close()

	// Build pair counts
	pairs := 0
	for _, ids := range memEntities {
		if len(ids) < 2 {
			continue
		}
		for i := 0; i < len(ids); i++ {
			for j := i + 1; j < len(ids); j++ {
				sid, tid := ids[i], ids[j]
				if sid > tid {
					sid, tid = tid, sid
				}
				if _, err := tx.Exec(
					`INSERT INTO entity_pair_counts (source_id, target_id, count) VALUES (?, ?, 1)
					 ON CONFLICT(source_id, target_id) DO UPDATE SET count = count + 1`,
					sid, tid,
				); err != nil {
					continue
				}
				pairs++
			}
		}
	}
	stats.PairsCounted = pairs

	// Step 5: promote pairs with count >= 2 into entity_entity_edges as related_to
	res, err := tx.Exec(
		`INSERT INTO entity_entity_edges (source_id, target_id, rel_type, weight)
		 SELECT source_id, target_id, 'related_to', CAST(count AS REAL)
		 FROM entity_pair_counts
		 WHERE count >= 2
		 ON CONFLICT(source_id, target_id, rel_type) DO UPDATE SET weight = excluded.weight`,
	)
	if err != nil {
		return stats, fmt.Errorf("rebuild promote pairs: %w", err)
	}
	if res != nil {
		n, _ := res.RowsAffected()
		stats.EdgesUpdated = int(n)
	}

	// Step 6: compute final stats
	tx.QueryRow(`SELECT COUNT(*) FROM entity_nodes`).Scan(&stats.NodesTotal)
	tx.QueryRow(`SELECT COUNT(*) FROM entity_entity_edges`).Scan(&stats.EdgesTotal)

	if stats.NodesTotal > 0 {
		stats.AvgDegree = 2.0 * float64(stats.EdgesTotal) / float64(stats.NodesTotal)
	}

	if err := tx.Commit(); err != nil {
		return stats, fmt.Errorf("rebuild commit: %w", err)
	}

	return stats, nil
}

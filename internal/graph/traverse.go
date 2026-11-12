package graph

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const decay = 0.5

func memoryEdgeQuery(entityID int, namespace string) (string, []any) {
	if namespace == "" {
		return `SELECT memory_id, weight FROM entity_edges WHERE entity_id = ?`, []any{entityID}
	}
	return `SELECT e.memory_id, e.weight FROM entity_edges e JOIN memories m ON m.id = e.memory_id WHERE e.entity_id = ? AND m.namespace = ? AND m.superseded_at IS NULL`, []any{entityID, namespace}
}

type TraversalResult struct {
	MemoryID   string   `json:"memory_id"`
	Score      float64  `json:"score"`
	PathLength int      `json:"path_length"`
	Entities   []string `json:"entities"`
}

func (s *Store) Traverse(seedNames []string, depth, limit int) ([]TraversalResult, error) {
	return s.traverse(seedNames, depth, limit, "")
}

// TraverseScoped applies an exact namespace filter while collecting each
// entity's memory edges, before scores are sorted and limited. Entity-to-entity
// relations remain global, so this scopes returned memories, not graph topology
// or tenant authorization.
func (s *Store) TraverseScoped(seedNames []string, depth, limit int, namespace string) ([]TraversalResult, error) {
	if namespace == "" {
		return s.Traverse(seedNames, depth, limit)
	}
	return s.traverse(seedNames, depth, limit, namespace)
}

func (s *Store) traverse(seedNames []string, depth, limit int, namespace string) ([]TraversalResult, error) {
	if depth <= 0 {
		depth = 2
	}
	if limit <= 0 {
		limit = 10
	}

	// Step 1: resolve seed entity IDs
	seedIDs := make(map[int]string)
	for _, name := range seedNames {
		var id int
		if err := s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, name).Scan(&id); err == nil && id > 0 {
			seedIDs[id] = name
		}
	}
	if len(seedIDs) == 0 {
		return nil, nil
	}

	visitedEntities := make(map[int]bool)
	for id := range seedIDs {
		visitedEntities[id] = true
	}
	visitedMemories := make(map[string]*TraversalResult)

	// Step 2: collect seed's own memories (hop 0, path_length = 1)
	for id := range seedIDs {
		query, args := memoryEdgeQuery(id, namespace)
		rows, err := s.db.Query(query, args...)
		if err != nil {
			if namespace != "" {
				return nil, fmt.Errorf("traverse scoped seed memories: %w", err)
			}
			continue
		}
		for rows.Next() {
			var memID string
			var w float64
			if err := rows.Scan(&memID, &w); err != nil {
				if namespace != "" {
					rows.Close()
					return nil, fmt.Errorf("scan scoped seed memory: %w", err)
				}
				continue
			}
			contrib := w * math.Pow(decay, 0) / 1.0
			if existing, ok := visitedMemories[memID]; ok {
				existing.Score += contrib
				if 0 < existing.PathLength {
					existing.PathLength = 0
				}
			} else {
				visitedMemories[memID] = &TraversalResult{
					MemoryID:   memID,
					Score:      contrib,
					PathLength: 0,
					Entities:   []string{seedIDs[id]},
				}
			}
		}
		if err := rows.Err(); err != nil && namespace != "" {
			rows.Close()
			return nil, fmt.Errorf("iterate scoped seed memories: %w", err)
		}
		rows.Close()
	}

	// Step 3-5: BFS expansion
	frontier := make([]int, 0, len(seedIDs))
	for id := range seedIDs {
		frontier = append(frontier, id)
	}

	for hop := 1; hop <= depth; hop++ {
		if len(frontier) == 0 {
			break
		}

		frontierSet := make(map[int]bool, len(frontier))
		for _, fid := range frontier {
			frontierSet[fid] = true
		}

		// Build IN clause with correct number of placeholders
		placeholders := make([]string, len(frontier))
		for i := range placeholders {
			placeholders[i] = "?"
		}
		ph := strings.Join(placeholders, ",")

		query := fmt.Sprintf(`SELECT source_id, target_id, rel_type, weight FROM entity_entity_edges WHERE source_id IN (%s) OR target_id IN (%s)`, ph, ph)

		allArgs := make([]interface{}, 0, len(frontier)*2)
		for _, fid := range frontier {
			allArgs = append(allArgs, fid)
		}
		for _, fid := range frontier {
			allArgs = append(allArgs, fid)
		}

		rows, err := s.db.Query(query, allArgs...)
		if err != nil {
			return nil, fmt.Errorf("traverse: %w", err)
		}

		nextFrontierSet := make(map[int]bool)
		type edgeDisc struct {
			entityID int
			weight   float64
		}
		discoveries := make([]edgeDisc, 0)

		for rows.Next() {
			var sourceID, targetID int
			var relType string
			var weight float64
			if err := rows.Scan(&sourceID, &targetID, &relType, &weight); err != nil {
				continue
			}
			_ = relType

			if frontierSet[sourceID] && !visitedEntities[targetID] {
				discoveries = append(discoveries, edgeDisc{targetID, weight})
				nextFrontierSet[targetID] = true
			}
			if frontierSet[targetID] && !visitedEntities[sourceID] {
				discoveries = append(discoveries, edgeDisc{sourceID, weight})
				nextFrontierSet[sourceID] = true
			}
		}
		rows.Close()

		for _, d := range discoveries {
			if visitedEntities[d.entityID] {
				continue
			}
			visitedEntities[d.entityID] = true

			var entityName string
			s.db.QueryRow(`SELECT name FROM entity_nodes WHERE id = ?`, d.entityID).Scan(&entityName)

			query, args := memoryEdgeQuery(d.entityID, namespace)
			memRows, err := s.db.Query(query, args...)
			if err != nil {
				if namespace != "" {
					return nil, fmt.Errorf("traverse scoped related memories: %w", err)
				}
				continue
			}
			for memRows.Next() {
				var memID string
				var edgeWeight float64
				if err := memRows.Scan(&memID, &edgeWeight); err != nil {
					if namespace != "" {
						memRows.Close()
						return nil, fmt.Errorf("scan scoped related memory: %w", err)
					}
					continue
				}

				pathLength := hop + 1
				contrib := edgeWeight * math.Pow(decay, float64(hop)) / float64(pathLength)

				if existing, ok := visitedMemories[memID]; ok {
					existing.Score += contrib
					if hop < existing.PathLength {
						existing.PathLength = hop
						existing.Entities = []string{entityName}
					}
				} else {
					visitedMemories[memID] = &TraversalResult{
						MemoryID:   memID,
						Score:      contrib,
						PathLength: hop,
						Entities:   []string{entityName},
					}
				}
			}
			if err := memRows.Err(); err != nil && namespace != "" {
				memRows.Close()
				return nil, fmt.Errorf("iterate scoped related memories: %w", err)
			}
			memRows.Close()
		}

		frontier = make([]int, 0, len(nextFrontierSet))
		for id := range nextFrontierSet {
			frontier = append(frontier, id)
		}
	}

	results := make([]TraversalResult, 0, len(visitedMemories))
	for _, r := range visitedMemories {
		results = append(results, *r)
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].Score != results[j].Score {
			return results[i].Score > results[j].Score
		}
		return results[i].MemoryID < results[j].MemoryID
	})

	if len(results) > limit {
		results = results[:limit]
	}

	return results, nil
}

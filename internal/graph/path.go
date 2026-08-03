package graph

import (
	"fmt"
)

// PathHop represents a single directed hop in a graph path.
type PathHop struct {
	From    string  `json:"from"`
	To      string  `json:"to"`
	RelType string  `json:"rel_type"`
	Weight  float64 `json:"weight"`
}

type rawEdge struct {
	sid, tid int
	relType  string
	weight   float64
}

type parentInfo struct {
	prevID int
	edge   PathHop
}

// FindPath attempts to find a shortest path between sourceName and targetName
// using BFS over entity_entity_edges (both directions). Returns nil, nil if
// either entity is unknown or no path exists within maxDepth hops.
func (s *Store) FindPath(sourceName, targetName string, maxDepth int) ([]PathHop, error) {
	if maxDepth <= 0 {
		maxDepth = 4
	}

	// Resolve source and target IDs
	var sourceID, targetID int
	if err := s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, sourceName).Scan(&sourceID); err != nil || sourceID == 0 {
		return nil, nil
	}
	if err := s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, targetName).Scan(&targetID); err != nil || targetID == 0 {
		return nil, nil
	}
	if sourceID == targetID {
		return nil, nil
	}

	// BFS with parent tracking.
	visited := map[int]bool{sourceID: true}
	parent := make(map[int]parentInfo)
	frontier := []int{sourceID}

	for depth := 1; depth <= maxDepth; depth++ {
		if len(frontier) == 0 {
			break
		}

		frontierSet := make(map[int]bool, len(frontier))
		for _, fid := range frontier {
			frontierSet[fid] = true
		}

		// Build placeholders for IN clause
		placeholders := make([]string, len(frontier))
		for i := range placeholders {
			placeholders[i] = "?"
		}
		ph := joinPlaceholders(placeholders)

		// Query edges in both directions
		query := fmt.Sprintf(
			`SELECT source_id, target_id, rel_type, weight FROM entity_entity_edges WHERE source_id IN (%s) OR target_id IN (%s)`,
			ph, ph,
		)
		allArgs := make([]interface{}, 0, len(frontier)*2)
		for _, fid := range frontier {
			allArgs = append(allArgs, fid)
		}
		for _, fid := range frontier {
			allArgs = append(allArgs, fid)
		}

		rows, err := s.db.Query(query, allArgs...)
		if err != nil {
			return nil, fmt.Errorf("findpath: %w", err)
		}

		// Step 1: collect raw edges while rows are open
		var rawEdges []rawEdge
		for rows.Next() {
			var e rawEdge
			if err := rows.Scan(&e.sid, &e.tid, &e.relType, &e.weight); err != nil {
				continue
			}
			rawEdges = append(rawEdges, e)
		}
		rows.Close()

		// Step 2: process edges with closed rows (safe to query entity names)
		var nextFrontier []int
		found := false
		for _, e := range rawEdges {
			if found {
				break
			}
			// Try both directions: frontier -> new entity
			for _, dir := range [][2]int{{e.sid, e.tid}, {e.tid, e.sid}} {
				fromID, toID := dir[0], dir[1]
				if !frontierSet[fromID] || visited[toID] {
					continue
				}
				visited[toID] = true
				var fromName, toName string
				s.db.QueryRow(`SELECT name FROM entity_nodes WHERE id = ?`, fromID).Scan(&fromName)
				s.db.QueryRow(`SELECT name FROM entity_nodes WHERE id = ?`, toID).Scan(&toName)
				parent[toID] = parentInfo{
					prevID: fromID,
					edge: PathHop{
						From:    fromName,
						To:      toName,
						RelType: e.relType,
						Weight:  e.weight,
					},
				}
				nextFrontier = append(nextFrontier, toID)
				if toID == targetID {
					found = true
					break
				}
			}
		}
		if found {
			return reconstructPath(parent, sourceID, targetID), nil
		}

		frontier = nextFrontier
	}

	return nil, nil
}

func reconstructPath(parent map[int]parentInfo, sourceID, targetID int) []PathHop {
	var hops []PathHop
	cur := targetID
	for cur != sourceID {
		info, ok := parent[cur]
		if !ok {
			return nil
		}
		hops = append([]PathHop{info.edge}, hops...)
		cur = info.prevID
	}
	return hops
}

func joinPlaceholders(phs []string) string {
	if len(phs) == 0 {
		return ""
	}
	s := phs[0]
	for _, p := range phs[1:] {
		s += "," + p
	}
	return s
}
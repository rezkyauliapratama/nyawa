package graph

import (
	"regexp"
	"strings"
)

// RelationPattern maps a regex to a typed edge between entity categories.
type RelationPattern struct {
	RelType   string         // edge type: works_at, uses, located_in, part_of
	SourceCat string         // expected source entity category ("" = any)
	TargetCat string         // expected target entity category ("" = any)
	Re        *regexp.Regexp // compiled regex with 2 capture groups: (source, target)
}

var relationPatterns = []RelationPattern{
	// works_at: person -> org (Indonesian + English)
	{
		RelType:   "works_at",
		SourceCat: "person",
		TargetCat: "org",
		Re: regexp.MustCompile(
			`(?i)\b([\w][\w .\-]+)\s+(?:bekerja di|kerja di|bergabung dengan|joined|works at)\s+([\w][\w .\-]+)(?:\.|,|;|\s*$|\s+dan\s|\s+di\s|\s+untuk\s|\s+dengan\s|\s+pada\s|\s+yang\s|\s+sejak\s)`,
		),
	},
	// uses: entity -> tech (Indonesian + English)
	{
		RelType:   "uses",
		SourceCat: "",
		TargetCat: "tech",
		Re: regexp.MustCompile(
			`(?i)\b([\w][\w .\-]+)\s+(?:menggunakan|pakai|memakai|uses)\s+([\w][\w .\-]+)(?:\.|,|;|\s*$|\s+dan\s|\s+di\s|\s+untuk\s|\s+dengan\s|\s+pada\s|\s+yang\s|\s+sebagai\s)`,
		),
	},
	// located_in: entity -> place (Indonesian + English)
	{
		RelType:   "located_in",
		SourceCat: "",
		TargetCat: "place",
		Re: regexp.MustCompile(
			`(?i)\b([\w][\w .\-]+)\s+(?:berlokasi di|located in)\s+([\w][\w .\-]+)(?:\.|,|;|\s*$|\s+dan\s|\s+di\s|\s+dengan\s|\s+yang\s)`,
		),
	},
	// part_of: child -> group (Indonesian + English)
	{
		RelType:   "part_of",
		SourceCat: "",
		TargetCat: "group",
		Re: regexp.MustCompile(
			`(?i)\b([\w][\w .\-]+)\s+(?:bagian dari|part of)\s+([\w][\w .\-]+)(?:\.|,|;|\s*$|\s+dan\s|\s+di\s|\s+dengan\s|\s+yang\s)`,
		),
	},
}

// InferTypedEdges scans content for relation patterns and inserts typed edges
// between existing entity nodes. Entities must already exist in entity_nodes
// (InsertMemoryEntities should be called first). Never returns an error that
// breaks the caller: DB errors are logged and skipped.
func (s *Store) InferTypedEdges(memoryID string, content string) error {
	s.mu.Lock()
	s.inferTotal++
	s.mu.Unlock()

	matched := false
	for _, p := range relationPatterns {
		matches := p.Re.FindAllStringSubmatch(content, -1)
		for _, match := range matches {
			if len(match) < 3 {
				continue
			}
			sourceName := strings.TrimSpace(match[1])
			targetNamesStr := strings.TrimSpace(match[2])

			// Split targets on "dan", "and", "," to handle multi-target patterns
			targetNames := splitTargets(targetNamesStr)

			for _, targetName := range targetNames {
				targetName = strings.TrimSpace(targetName)
				if targetName == "" || sourceName == "" {
					continue
				}

				// Resolve entity names to IDs; auto-register missing entities
				// so typed inference works even when the classifier extractor
				// does not recognize the entity category.
				sourceID, err := s.resolveOrCreateEntity(sourceName, p.SourceCat)
				if err != nil {
					continue
				}
				targetID, err := s.resolveOrCreateEntity(targetName, p.TargetCat)
				if err != nil {
					continue
				}
				if sourceID == 0 || targetID == 0 || sourceID == targetID {
					continue
				}

				// Insert typed edge with UPSERT
				if _, err := s.db.Exec(
					`INSERT INTO entity_entity_edges (source_id, target_id, rel_type, weight)
					 VALUES (?, ?, ?, 1.0)
					 ON CONFLICT(source_id, target_id, rel_type) DO UPDATE SET weight = weight + 1`,
					sourceID, targetID, p.RelType,
				); err != nil {
					continue
				}
				matched = true
			}
		}
	}

	if matched {
		s.mu.Lock()
		s.inferMatched++
		s.mu.Unlock()
	}

	return nil
}

// splitTargets splits a compound target string on "dan", "and", or ",".
func splitTargets(s string) []string {
	s = strings.ReplaceAll(s, " dan ", "\x00")
	s = strings.ReplaceAll(s, " and ", "\x00")
	s = strings.ReplaceAll(s, ", ", "\x00")
	s = strings.ReplaceAll(s, ",", "\x00")
	parts := strings.Split(s, "\x00")
	var result []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			result = append(result, p)
		}
	}
	if len(result) == 0 {
		return []string{s}
	}
	return result
}

// resolveOrCreateEntity returns the entity node ID for name, auto-registering
// the node (with category) if it does not exist yet. Returns 0 on error.
func (s *Store) resolveOrCreateEntity(name string, category string) (int, error) {
	if category == "" {
		category = "unknown"
	}
	var id int
	err := s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, name).Scan(&id)
	if err == nil {
		return id, nil
	}
	if _, err := s.db.Exec(`INSERT INTO entity_nodes (name, category) VALUES (?, ?)`, name, category); err != nil {
		return 0, err
	}
	if err := s.db.QueryRow(`SELECT id FROM entity_nodes WHERE name = ? COLLATE NOCASE`, name).Scan(&id); err != nil {
		return 0, err
	}
	return id, nil
}

package compactioneval

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"unicode"

	_ "github.com/mattn/go-sqlite3"
)

const RecallRunnerVersion = "v1"

type RecallRunnerConfig struct {
	DBPath           string            `json:"db_path"`
	Snapshot         bool              `json:"snapshot"`
	Environment      map[string]string `json:"environment,omitempty"`
	DefaultNamespace string            `json:"default_namespace,omitempty"`
}

type RecallProbeSpec struct {
	ProbeID         string   `json:"probe_id"`
	Query           string   `json:"query"`
	Namespace       string   `json:"namespace"`
	Limit           int      `json:"limit"`
	MinScore        float64  `json:"min_score"`
	ExcludeTypes    []string `json:"exclude_types"`
	TargetMemoryIDs []string `json:"target_memory_ids"`
}

type RecallProbeCondition struct {
	Condition string            `json:"condition"`
	Probes    []RecallProbeSpec `json:"probes"`
}

type RecallProbeSuite struct {
	SchemaVersion string                 `json:"schema_version"`
	SuiteID       string                 `json:"suite_id"`
	Environment   map[string]string      `json:"environment,omitempty"`
	Conditions    []RecallProbeCondition `json:"conditions"`
}

type RecallProbeResult struct {
	ProbeID         string   `json:"probe_id"`
	Condition       string   `json:"condition"`
	Query           string   `json:"query"`
	Namespace       string   `json:"namespace"`
	Limit           int      `json:"limit"`
	MinScore        float64  `json:"min_score"`
	ExcludeTypes    []string `json:"exclude_types"`
	TargetMemoryIDs []string `json:"target_memory_ids"`
	TargetStatus    string   `json:"target_status"`
	ResultIDs       []string `json:"result_ids"`
	Error           string   `json:"error,omitempty"`
}

type RecallProbeRun struct {
	SchemaVersion string                  `json:"schema_version"`
	RunnerVersion string                  `json:"runner_version"`
	DBPath        string                  `json:"db_path"`
	DBSHA256      string                  `json:"db_sha256"`
	Snapshot      bool                    `json:"snapshot"`
	Environment   map[string]string       `json:"environment"`
	Conditions    []string                `json:"conditions"`
	Probes        []RecallProbeResult     `json:"probes"`
	Metrics       []RecallProbeRunMetrics `json:"metrics"`
}

type RecallProbeRunMetrics struct {
	Condition          string   `json:"condition"`
	TotalProbes        int      `json:"total_probes"`
	ValidProbes        int      `json:"valid_probes"`
	UnobservableProbes int      `json:"unobservable_probes"`
	SupersededProbes   int      `json:"superseded_probes"`
	RecallAt1          *float64 `json:"recall_at_1"`
	RecallAt3          *float64 `json:"recall_at_3"`
	RecallAt5          *float64 `json:"recall_at_5"`
	MRRAt5             *float64 `json:"mrr_at_5"`
}

func ValidateSnapshotDBPath(path string, snapshot bool) (string, error) {
	if !snapshot {
		return "", errors.New("recall probes require snapshot=true; production databases are never a default")
	}
	if path == "" || path == "-" || strings.Contains(path, "?") || strings.HasPrefix(path, "file:") || path == ":memory:" {
		return "", errors.New("DBPath must be an explicit filesystem snapshot path")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("resolve snapshot path: %w", err)
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("stat snapshot path: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", errors.New("snapshot path must be a regular file")
	}
	if isKnownProductionPath(abs) {
		return "", errors.New("refusing known production Nyawa database path")
	}
	return abs, nil
}
func isKnownProductionPath(path string) bool {
	clean := filepath.ToSlash(filepath.Clean(path))
	return strings.HasSuffix(clean, "/.nyawa/memory.db") || strings.HasSuffix(clean, "/.nyawa/memory.sqlite")
}

func RunRecallProbeSuite(config RecallRunnerConfig, suite RecallProbeSuite) (RecallProbeRun, error) {
	path, err := ValidateSnapshotDBPath(config.DBPath, config.Snapshot)
	if err != nil {
		return RecallProbeRun{}, err
	}
	if suite.SchemaVersion != "" && suite.SchemaVersion != TraceSchemaVersion {
		return RecallProbeRun{}, fmt.Errorf("unsupported probe suite schema_version %q", suite.SchemaVersion)
	}
	if suite.SuiteID == "" {
		return RecallProbeRun{}, errors.New("probe suite_id is required")
	}
	if len(suite.Conditions) == 0 {
		return RecallProbeRun{}, errors.New("at least one probe condition is required")
	}
	sha, err := fileSHA256(path)
	if err != nil {
		return RecallProbeRun{}, fmt.Errorf("hash snapshot: %w", err)
	}
	conn, err := sql.Open("sqlite3", "file:"+path+"?mode=ro&_query_only=1")
	if err != nil {
		return RecallProbeRun{}, fmt.Errorf("open read-only snapshot: %w", err)
	}
	defer conn.Close()
	if err := conn.Ping(); err != nil {
		return RecallProbeRun{}, fmt.Errorf("ping read-only snapshot: %w", err)
	}
	if _, err := conn.Exec("PRAGMA query_only=ON"); err != nil {
		return RecallProbeRun{}, fmt.Errorf("enable sqlite query_only: %w", err)
	}
	env := mergeEnvironment(config.Environment, suite.Environment)
	if env == nil {
		env = map[string]string{}
	}
	env["runner_version"] = RecallRunnerVersion
	env["sqlite_mode"] = "ro_query_only"
	env["go_version"] = runtime.Version()
	conditions := make([]string, 0, len(suite.Conditions))
	probes := make([]RecallProbeResult, 0)
	for _, condition := range suite.Conditions {
		if condition.Condition != ConditionSummaryOnly && condition.Condition != ConditionSummaryPlusBaseline {
			return RecallProbeRun{}, fmt.Errorf("unsupported probe condition %q", condition.Condition)
		}
		conditions = append(conditions, condition.Condition)
		for _, spec := range condition.Probes {
			result := RecallProbeResult{ProbeID: spec.ProbeID, Condition: condition.Condition, Query: spec.Query, Namespace: spec.Namespace, Limit: spec.Limit, MinScore: spec.MinScore, ExcludeTypes: append([]string{}, spec.ExcludeTypes...), TargetMemoryIDs: append([]string{}, spec.TargetMemoryIDs...), ResultIDs: []string{}}
			if result.Namespace == "" {
				result.Namespace = config.DefaultNamespace
			}
			if result.ProbeID == "" || result.Query == "" || result.Limit < 1 || result.Limit > 1000 {
				result.Error = "probe_id/query required and limit must be 1..1000"
				probes = append(probes, result)
				continue
			}
			status, statusErr := targetStatus(conn, result.TargetMemoryIDs, result.Namespace)
			result.TargetStatus = status
			if statusErr != nil {
				result.Error = statusErr.Error()
				probes = append(probes, result)
				continue
			}
			if result.MinScore != 0 {
				result.Error = "min_score is recorded but unsupported by the deterministic FTS-only runner; use min_score=0"
				probes = append(probes, result)
				continue
			}
			ids, queryErr := readOnlyFTSSearch(conn, result.Query, result.Namespace, result.ExcludeTypes, result.Limit)
			if queryErr != nil {
				result.Error = queryErr.Error()
			} else {
				result.ResultIDs = ids
			}
			probes = append(probes, result)
		}
	}
	return RecallProbeRun{SchemaVersion: "nyawa.compaction-eval-probe/v1", RunnerVersion: RecallRunnerVersion, DBPath: path, DBSHA256: sha, Snapshot: true, Environment: env, Conditions: conditions, Probes: probes, Metrics: recallProbeMetrics(probes, conditions)}, nil
}
func MarshalRecallProbeRun(run RecallProbeRun) ([]byte, error) { return json.Marshal(run) }
func mergeEnvironment(left, right map[string]string) map[string]string {
	out := make(map[string]string, len(left)+len(right))
	for k, v := range left {
		out[k] = v
	}
	for k, v := range right {
		out[k] = v
	}
	return out
}
func fileSHA256(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
func targetStatus(conn *sql.DB, ids []string, namespace string) (string, error) {
	if len(ids) == 0 {
		return string(RecallTargetUnobservable), nil
	}
	for _, id := range ids {
		var active int
		var superseded sql.NullString
		var ns string
		err := conn.QueryRow("SELECT CASE WHEN superseded_at IS NULL THEN 1 ELSE 0 END, superseded_at, namespace FROM memories WHERE id=?", id).Scan(&active, &superseded, &ns)
		if errors.Is(err, sql.ErrNoRows) {
			return string(RecallTargetUnobservable), nil
		}
		if err != nil {
			return "", err
		}
		if active == 0 {
			return string(RecallTargetSuperseded), nil
		}
		if namespace != "" && ns != namespace {
			return string(RecallTargetUnobservable), nil
		}
	}
	return string(RecallTargetActive), nil
}
func readOnlyFTSSearch(conn *sql.DB, query, namespace string, exclude []string, limit int) ([]string, error) {
	tokens := recallTokens(query)
	if len(tokens) == 0 {
		return []string{}, nil
	}
	parts := make([]string, len(tokens))
	for i, token := range tokens {
		parts[i] = `"` + token + `"`
	}
	where := []string{"memories_fts MATCH ?", "m.superseded_at IS NULL"}
	args := []any{strings.Join(parts, " OR ")}
	if namespace != "" {
		where = append(where, "m.namespace=?")
		args = append(args, namespace)
	}
	for _, typ := range exclude {
		if typ != "" {
			where = append(where, "m.mem_type<>?")
			args = append(args, typ)
		}
	}
	args = append(args, limit)
	rows, err := conn.Query("SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE "+strings.Join(where, " AND ")+" ORDER BY rank, m.id LIMIT ?", args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := make([]string, 0, limit)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}
func recallTokens(query string) []string {
	seen := map[string]bool{}
	tokens := []string{}
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		token := strings.ToLower(b.String())
		b.Reset()
		if len([]rune(token)) < 2 || seen[token] {
			return
		}
		seen[token] = true
		tokens = append(tokens, token)
	}
	for _, r := range query {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	sort.Strings(tokens)
	if len(tokens) > 32 {
		tokens = tokens[:32]
	}
	return tokens
}

type recallMetricAccumulator struct {
	total, valid, unobservable, superseded int
	hits                                   [3]int
	den                                    [3]int
	rr                                     float64
	rrDen                                  int
}

func recallProbeMetrics(probes []RecallProbeResult, order []string) []RecallProbeRunMetrics {
	by := map[string]*recallMetricAccumulator{}
	for _, condition := range order {
		if by[condition] == nil {
			by[condition] = &recallMetricAccumulator{}
		}
	}
	for _, probe := range probes {
		a := by[probe.Condition]
		if a == nil {
			a = &recallMetricAccumulator{}
			by[probe.Condition] = a
		}
		a.total++
		switch probe.TargetStatus {
		case string(RecallTargetUnobservable):
			a.unobservable++
			continue
		case string(RecallTargetSuperseded):
			a.superseded++
			continue
		}
		if probe.Error != "" {
			continue
		}
		a.valid++
		target := map[string]bool{}
		for _, id := range probe.TargetMemoryIDs {
			target[id] = true
		}
		rank := 0
		for i, id := range probe.ResultIDs {
			if target[id] {
				rank = i + 1
				break
			}
		}
		for i, cutoff := range []int{1, 3, 5} {
			if probe.Limit >= cutoff {
				a.den[i]++
				if rank > 0 && rank <= cutoff {
					a.hits[i]++
				}
			}
		}
		if probe.Limit >= 5 {
			a.rrDen++
			if rank > 0 && rank <= 5 {
				a.rr += 1 / float64(rank)
			}
		}
	}
	out := make([]RecallProbeRunMetrics, 0, len(order))
	for _, condition := range order {
		a := by[condition]
		if a == nil {
			continue
		}
		out = append(out, RecallProbeRunMetrics{Condition: condition, TotalProbes: a.total, ValidProbes: a.valid, UnobservableProbes: a.unobservable, SupersededProbes: a.superseded, RecallAt1: metricRatio(a.hits[0], a.den[0]), RecallAt3: metricRatio(a.hits[1], a.den[1]), RecallAt5: metricRatio(a.hits[2], a.den[2]), MRRAt5: metricFloatRatio(a.rr, a.rrDen)})
	}
	return out
}
func metricRatio(n, d int) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}
func metricFloatRatio(n float64, d int) *float64 {
	if d == 0 {
		return nil
	}
	v := n / float64(d)
	return &v
}

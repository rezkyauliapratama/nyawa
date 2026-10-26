package compactioneval

import (
	"database/sql"
	"os"
	"testing"

	_ "github.com/mattn/go-sqlite3"
)

func TestRecallProbeSuiteReadOnlyAndMetrics(t *testing.T) {
	path := t.TempDir() + "/snapshot.db"
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`CREATE TABLE memories(id TEXT PRIMARY KEY,content TEXT NOT NULL,mem_type TEXT NOT NULL,namespace TEXT NOT NULL,superseded_at TEXT); CREATE VIRTUAL TABLE memories_fts USING fts5(content,content='memories',content_rowid='rowid'); INSERT INTO memories VALUES('m1','alpha baseline fact','fact','context',NULL); INSERT INTO memories_fts(rowid,content) SELECT rowid,content FROM memories;`)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	suite := RecallProbeSuite{SchemaVersion: TraceSchemaVersion, SuiteID: "suite", Conditions: []RecallProbeCondition{{Condition: ConditionSummaryOnly, Probes: []RecallProbeSpec{{ProbeID: "p1", Query: "alpha", Namespace: "context", Limit: 5, ExcludeTypes: []string{}, TargetMemoryIDs: []string{"m1"}}}}, {Condition: ConditionSummaryPlusBaseline, Probes: []RecallProbeSpec{{ProbeID: "p2", Query: "alpha", Namespace: "context", Limit: 5, ExcludeTypes: []string{}, TargetMemoryIDs: []string{"m1"}}}}}}
	run, err := RunRecallProbeSuite(RecallRunnerConfig{DBPath: path, Snapshot: true}, suite)
	if err != nil {
		t.Fatal(err)
	}
	if len(run.Metrics) != 2 || run.Metrics[0].RecallAt1 == nil || *run.Metrics[0].RecallAt1 != 1 {
		t.Fatalf("metrics=%+v", run.Metrics)
	}
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Fatal("read-only probe mutated snapshot")
	}
}

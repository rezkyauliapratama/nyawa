package main

import (
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/compactioneval"
)

func TestCompactionEvalScoresBeforeAndAfterFixtures(t *testing.T) {
	beforeOutput, err := invokeCompactionEval(t, "before.json", nil)
	if err != nil {
		t.Fatalf("score before-compaction fixture: %v", err)
	}
	before := decodeCLIResult(t, beforeOutput)
	if before.Condition != compactioneval.ConditionSummaryOnly || before.Runs != 1 {
		t.Fatalf("before result identity = %+v", before)
	}
	if before.RepeatedToolOutputs != 1 || before.PostCompactionRepeatedToolOutputs != 0 {
		t.Errorf("before repeat metrics = (%d, %d), want (1, 0)", before.RepeatedToolOutputs, before.PostCompactionRepeatedToolOutputs)
	}
	if len(before.CheckpointMetrics) != 0 {
		t.Errorf("before checkpoint metrics = %+v, want empty", before.CheckpointMetrics)
	}

	afterOutput, err := invokeCompactionEval(t, "after.json", nil)
	if err != nil {
		t.Fatalf("score after-compaction fixture: %v", err)
	}
	after := decodeCLIResult(t, afterOutput)
	if after.RepeatedToolOutputs != 1 || after.PostCompactionRepeatedToolOutputs != 1 {
		t.Errorf("after repeat metrics = (%d, %d), want (1, 1)", after.RepeatedToolOutputs, after.PostCompactionRepeatedToolOutputs)
	}
	if len(after.CheckpointMetrics) != 1 {
		t.Fatalf("after checkpoint metrics = %+v, want one boundary", after.CheckpointMetrics)
	}
	checkpoint := after.CheckpointMetrics[0]
	if checkpoint.CheckpointID != "legacy-boundary" || checkpoint.BoundaryEventIndex != 10 || checkpoint.ToolAttempts != 1 || checkpoint.RepeatedToolOutputs != 1 {
		t.Errorf("after checkpoint metrics = %+v", checkpoint)
	}
}

func TestCompactionEvalOutputIsDeterministicAndUsesStableKeys(t *testing.T) {
	first, err := invokeCompactionEval(t, "after.json", nil)
	if err != nil {
		t.Fatalf("first score: %v", err)
	}
	second, err := invokeCompactionEval(t, "after.json", nil)
	if err != nil {
		t.Fatalf("second score: %v", err)
	}
	if first != second {
		t.Fatalf("repeated CLI scoring differed:\nfirst=%s\nsecond=%s", first, second)
	}
	if !strings.HasPrefix(first, `{"schema_version":"nyawa.compaction-eval-result/v1","evaluator_version":"v1"`) {
		t.Fatalf("result keys are not in deterministic struct order: %s", first)
	}
}

func TestCompactionEvalAcceptsPathAndStdinWithIdenticalOutput(t *testing.T) {
	pathOutput, err := invokeCompactionEval(t, "after.json", nil)
	if err != nil {
		t.Fatalf("path score: %v", err)
	}
	stdinOutput, err := invokeCompactionEval(t, "-", fixtureBytes(t, "after.json"))
	if err != nil {
		t.Fatalf("stdin score: %v", err)
	}
	if pathOutput != stdinOutput {
		t.Fatalf("path and stdin output differed:\npath=%s\nstdin=%s", pathOutput, stdinOutput)
	}
}

func TestCompactionEvalInvalidTraceReturnsClearError(t *testing.T) {
	output, err := invokeCompactionEval(t, "invalid.json", nil)
	if err == nil {
		t.Fatal("invalid trace returned nil error")
	}
	if output != "" {
		t.Errorf("invalid trace wrote stdout = %q, want empty", output)
	}
	if !strings.Contains(err.Error(), "invalid trace") || !strings.Contains(err.Error(), "outcome") {
		t.Errorf("invalid trace error = %q, want source and validation details", err)
	}
}

func TestCompactionEvalCLIInvalidTraceExitsNonZero(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestCompactionEvalCLIHelperProcess$")
	cmd.Env = append(os.Environ(),
		"NYAWA_COMPACTION_EVAL_HELPER=1",
		"NYAWA_COMPACTION_EVAL_TRACE="+fixturePath(t, "invalid.json"),
	)
	output, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatalf("CLI accepted invalid trace; output=%s", output)
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok || exitErr.ExitCode() == 0 {
		t.Fatalf("CLI error = %v, want non-zero exit", err)
	}
	if !strings.Contains(string(output), "invalid trace") || !strings.Contains(string(output), "outcome") {
		t.Errorf("CLI error output = %q, want clear validation details", output)
	}
}

func TestCompactionEvalCLIHelperProcess(t *testing.T) {
	if os.Getenv("NYAWA_COMPACTION_EVAL_HELPER") != "1" {
		return
	}
	os.Args = []string{"nyawa", "compaction-eval", "score", os.Getenv("NYAWA_COMPACTION_EVAL_TRACE")}
	main()
}

func invokeCompactionEval(t *testing.T, inputName string, stdinData []byte) (string, error) {
	t.Helper()
	originalArgs := os.Args
	originalStdin := os.Stdin
	defer func() {
		os.Args = originalArgs
		os.Stdin = originalStdin
	}()
	os.Args = []string{"nyawa", "compaction-eval", "score", inputName}
	if inputName != "-" {
		os.Args[3] = fixturePath(t, inputName)
	}
	if inputName == "-" {
		file, err := os.CreateTemp(t.TempDir(), "trace-stdin-*.json")
		if err != nil {
			t.Fatalf("create stdin fixture: %v", err)
		}
		if _, err := file.Write(stdinData); err != nil {
			t.Fatalf("write stdin fixture: %v", err)
		}
		if _, err := file.Seek(0, io.SeekStart); err != nil {
			t.Fatalf("rewind stdin fixture: %v", err)
		}
		defer file.Close()
		os.Stdin = file
	}

	var scoreErr error
	output := captureStdout(t, func() { scoreErr = cmdCompactionEval() })
	return output, scoreErr
}

func decodeCLIResult(t *testing.T, output string) compactioneval.Result {
	t.Helper()
	var result compactioneval.Result
	if err := json.Unmarshal([]byte(output), &result); err != nil {
		t.Fatalf("decode CLI result: %v\noutput=%s", err, output)
	}
	return result
}

func fixturePath(t *testing.T, name string) string {
	t.Helper()
	return filepath.Join("testdata", "compaction-eval", name)
}

func fixtureBytes(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(fixturePath(t, name))
	if err != nil {
		t.Fatalf("read fixture %q: %v", name, err)
	}
	return data
}

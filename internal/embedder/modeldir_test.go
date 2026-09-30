package embedder

import (
	"os"
	"path/filepath"
	"testing"
)

func noEnv(string) string { return "" }

func withModelDirs(t *testing.T, dirs ...string) {
	t.Helper()
	old := embedModelDirs
	embedModelDirs = dirs
	t.Cleanup(func() { embedModelDirs = old })
}

func TestResolveModelDirEnvOverrideWins(t *testing.T) {
	withModelDirs(t, "/nonexistent/a", "/nonexistent/b")
	path, source := resolveModelDir(
		func(k string) string {
			if k == "NYAWA_EMBED_MODEL_DIR" {
				return "  /custom/model  "
			}
			return ""
		},
		func(string) (os.FileInfo, error) { return nil, os.ErrNotExist },
	)
	if path != "/custom/model" {
		t.Errorf("path = %q, want /custom/model (env must be trimmed and win)", path)
	}
	if source != "env:NYAWA_EMBED_MODEL_DIR" {
		t.Errorf("source = %q, want env:NYAWA_EMBED_MODEL_DIR", source)
	}
}

func TestResolveModelDirPicksFirstExisting(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "missing")
	present := filepath.Join(dir, "model")
	if err := os.MkdirAll(present, 0o755); err != nil {
		t.Fatal(err)
	}
	withModelDirs(t, missing, present)

	path, source := resolveModelDir(noEnv, os.Stat)
	if path != present {
		t.Errorf("path = %q, want %q", path, present)
	}
	if source != "default" {
		t.Errorf("source = %q, want default", source)
	}
}

func TestResolveModelDirFallsBackToFirstWhenNoneExist(t *testing.T) {
	dir := t.TempDir()
	first, second := filepath.Join(dir, "a"), filepath.Join(dir, "b")
	withModelDirs(t, first, second)

	path, source := resolveModelDir(noEnv, os.Stat)
	if path != first {
		t.Errorf("path = %q, want the first candidate %q", path, first)
	}
	if source != "default(missing)" {
		t.Errorf("source = %q, want default(missing)", source)
	}
}

// The persistent-volume venv must be probed before the image's own venv, since
// only the former survives container recreation.
func TestPythonCandidatesPreferPersistentVenv(t *testing.T) {
	if len(embedPythonCandidates) == 0 {
		t.Fatal("no python candidates configured")
	}
	if got, want := embedPythonCandidates[0], "/opt/data/bge-venv/bin/python3"; got != want {
		t.Errorf("first python candidate = %q, want %q", got, want)
	}
}

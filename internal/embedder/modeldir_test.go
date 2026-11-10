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

// With no explicit override, the durable image probes its bundled interpreter
// candidates before relying on generic PATH lookup.
func TestPythonCandidatesPreferImageInterpreter(t *testing.T) {
	want := []string{"/usr/local/bin/python3", "/usr/bin/python3", "python3"}
	if len(embedPythonCandidates) != len(want) {
		t.Fatalf("python candidates = %v, want %v", embedPythonCandidates, want)
	}
	for i := range want {
		if embedPythonCandidates[i] != want[i] {
			t.Errorf("python candidate %d = %q, want %q", i, embedPythonCandidates[i], want[i])
		}
	}
}

func TestResolvePythonPathAcceptsVerifiedAbsoluteOverride(t *testing.T) {
	const configuredPath = "/custom/python3"
	probed := []string{}
	path, err := resolvePythonPath(
		func(key string) (string, bool) { return configuredPath, key == "NYAWA_EMBED_PYTHON" },
		[]string{"/image/python3"},
		func(candidate string) error {
			probed = append(probed, candidate)
			return nil
		},
	)
	if err != nil {
		t.Fatalf("resolvePythonPath() error = %v", err)
	}
	if path != configuredPath {
		t.Errorf("path = %q, want %q", path, configuredPath)
	}
	if len(probed) != 1 || probed[0] != configuredPath {
		t.Errorf("probed = %v, want only %q", probed, configuredPath)
	}
}

func TestResolvePythonPathRejectsInvalidOverrideWithoutFallback(t *testing.T) {
	for _, configuredPath := range []string{"python3", ""} {
		t.Run("path="+configuredPath, func(t *testing.T) {
			probed := false
			path, err := resolvePythonPath(
				func(string) (string, bool) { return configuredPath, true },
				[]string{"/image/python3"},
				func(string) error { probed = true; return nil },
			)
			if err == nil {
				t.Fatal("resolvePythonPath() error = nil, want invalid-override error")
			}
			if path != "" {
				t.Errorf("path = %q, want empty", path)
			}
			if probed {
				t.Error("invalid relative override must not be probed or fall back")
			}
		})
	}
}

func TestResolvePythonPathProbeFailureDoesNotFallback(t *testing.T) {
	probed := []string{}
	path, err := resolvePythonPath(
		func(string) (string, bool) { return "/custom/python3", true },
		[]string{"/image/python3"},
		func(candidate string) error {
			probed = append(probed, candidate)
			return os.ErrPermission
		},
	)
	if err == nil {
		t.Fatal("resolvePythonPath() error = nil, want probe error")
	}
	if path != "" {
		t.Errorf("path = %q, want empty", path)
	}
	if len(probed) != 1 || probed[0] != "/custom/python3" {
		t.Errorf("probed = %v, want only explicit override", probed)
	}
}

func TestResolvePythonPathUsesDefaultCandidatesWhenUnset(t *testing.T) {
	probed := []string{}
	path, err := resolvePythonPath(
		func(string) (string, bool) { return "", false },
		[]string{"/usr/local/bin/python3", "/usr/bin/python3", "python3"},
		func(candidate string) error {
			probed = append(probed, candidate)
			if candidate == "/usr/bin/python3" {
				return nil
			}
			return os.ErrNotExist
		},
	)
	if err != nil {
		t.Fatalf("resolvePythonPath() error = %v", err)
	}
	if path != "/usr/bin/python3" {
		t.Errorf("path = %q, want /usr/bin/python3", path)
	}
	wantProbed := []string{"/usr/local/bin/python3", "/usr/bin/python3"}
	if len(probed) != len(wantProbed) {
		t.Fatalf("probed = %v, want %v", probed, wantProbed)
	}
	for i := range wantProbed {
		if probed[i] != wantProbed[i] {
			t.Errorf("probe %d = %q, want %q", i, probed[i], wantProbed[i])
		}
	}
}

package embedder

import (
	"os"
	"strings"
	"testing"
)

// This is intentionally a static contract test. The image owns one private
// stdin/stdout child per embedder instance. An explicit interpreter path is
// opt-in and import-checked; runtime transport must remain socket-free.
func TestEmbeddedRuntimeHasSingleProcessContract(t *testing.T) {
	data, err := os.ReadFile("bge_server.py")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	for _, forbidden := range []string{"socket", "serve_mode", "AF_UNIX", "threading"} {
		if strings.Contains(source, forbidden) {
			t.Fatalf("bge_server.py contains forbidden alternate transport %q", forbidden)
		}
	}
	if !strings.Contains(source, "READY dim=") || !strings.Contains(source, "for line in sys.stdin") {
		t.Fatal("bge_server.py must expose deterministic stderr readiness and stdin JSONL")
	}

	goSource, err := os.ReadFile("py_embedder.go")
	if err != nil {
		t.Fatal(err)
	}
	goText := string(goSource)
	for _, forbidden := range []string{"net.Dial", "socketPathFor", "/opt/data/bge-venv", "/opt/hermes/.venv"} {
		if strings.Contains(goText, forbidden) {
			t.Fatalf("py_embedder.go contains forbidden alternate transport or hard-coded host path %q", forbidden)
		}
	}
	if !strings.Contains(goText, `lookup("NYAWA_EMBED_PYTHON")`) || !strings.Contains(goText, "filepath.IsAbs(path)") {
		t.Fatal("py_embedder.go must validate the explicit interpreter override as an absolute path")
	}
	if strings.Count(goText, "exec.Command(pythonPath, scriptPath)") != 1 {
		t.Fatal("exactly one Python child launch is required")
	}
}

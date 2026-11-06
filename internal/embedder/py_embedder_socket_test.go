package embedder

import (
	"os"
	"strings"
	"testing"
)

// This is intentionally a static contract test. The durable image owns one
// private stdin/stdout child; it must not regress to a shared socket or a host
// probe that can race another Nyawa process.
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
	if strings.Contains(goText, "net.Dial") || strings.Contains(goText, "socketPathFor") || strings.Contains(goText, "NYAWA_EMBED_PYTHON") {
		t.Fatal("py_embedder.go must not use socket or host Python override paths")
	}
	if strings.Count(goText, "exec.Command(pythonPath, scriptPath)") != 1 {
		t.Fatal("exactly one Python child launch is required")
	}
}

package main

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/version"
)

// captureStdout runs fn with os.Stdout redirected to a temp file and returns
// everything it printed.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stdout.txt")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create capture file: %v", err)
	}
	orig := os.Stdout
	os.Stdout = f
	defer func() { os.Stdout = orig; f.Close() }()

	fn()

	os.Stdout = orig
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("seek capture file: %v", err)
	}
	data, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("read capture file: %v", err)
	}
	return string(data)
}

// TestUsageBannerReportsCanonicalVersion locks the CLI banner and the `version`
// subcommand output to the single version source, so the stale literal cannot
// come back.
func TestUsageBannerReportsCanonicalVersion(t *testing.T) {
	banner := captureStdout(t, printUsage)
	if !strings.Contains(banner, "v"+version.Version) {
		t.Errorf("usage banner does not contain v%s:\n%s", version.Version, banner)
	}

	cmd := captureStdout(t, func() { os.Args = []string{"nyawa", "version"}; main() })
	if !strings.Contains(cmd, version.String()) {
		t.Errorf("`nyawa version` output = %q, want it to contain %q", cmd, version.String())
	}
}

// TestVersionStringForm proves the display helper keeps the documented shape.
func TestVersionStringForm(t *testing.T) {
	if got, want := version.String(), "nyawa v"+version.Version; got != want {
		t.Errorf("version.String() = %q, want %q", got, want)
	}
}

func TestMCPHTTPListenerConfigurationFailsClosed(t *testing.T) {
	if srv, err := startMCPHTTP("", "", nil, nil, nil); err != nil || srv != nil {
		t.Fatalf("empty port should disable MCP HTTP, got server=%v err=%v", srv, err)
	}
	if _, err := startMCPHTTP("not-a-port", "test-token", nil, nil, nil); err == nil {
		t.Fatal("invalid MCP HTTP port was accepted")
	}
	if _, err := startMCPHTTP("3301", "", nil, nil, nil); err == nil {
		t.Fatal("MCP HTTP listener was configured without a bearer token")
	}
}

func TestMCPHealthHandlerIsMinimalAndUnauthenticated(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rw := httptest.NewRecorder()
	mcpHealthHandler().ServeHTTP(rw, req)
	if rw.Code != http.StatusOK || strings.TrimSpace(rw.Body.String()) != `{"status":"ready"}` {
		t.Fatalf("health response = status %d body %q", rw.Code, rw.Body.String())
	}

	req = httptest.NewRequest(http.MethodPost, "/healthz", nil)
	rw = httptest.NewRecorder()
	mcpHealthHandler().ServeHTTP(rw, req)
	if rw.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /healthz status = %d, want 405", rw.Code)
	}
}

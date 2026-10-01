package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/store"
	"github.com/rezkyauliapratama/nyawa/internal/version"
)

// TestVersionEndpointsReportCanonicalVersion locks every HTTP endpoint that
// reports a version to the single source. They used to hardcode "0.1.0" while
// the binary was v1.2.0.
func TestVersionEndpointsReportCanonicalVersion(t *testing.T) {
	st, err := store.NewStore(filepath.Join(t.TempDir(), "version-endpoints.db"), nil)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	srv := New(st, nil, nil, nil, DefaultServerConfig())

	cases := []struct {
		name    string
		handler http.HandlerFunc
		method  string
		target  string
	}{
		{"status /", srv.handleRoot, http.MethodGet, "/"},
		{"stats /v1/stats", srv.handleStats, http.MethodGet, "/v1/stats"},
		{"health /v1/health", srv.handleHealth, http.MethodGet, "/v1/health"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			tc.handler(rec, httptest.NewRequest(tc.method, tc.target, nil))
			var body map[string]any
			if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
				t.Fatalf("decode %s body %q: %v", tc.target, rec.Body.String(), err)
			}
			got, _ := body["version"].(string)
			if got != version.Version {
				t.Errorf("%s version = %q, want %q", tc.target, got, version.Version)
			}
		})
	}
}

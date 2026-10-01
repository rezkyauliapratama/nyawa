package mcp

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/version"
)

// TestInitializeReportsCanonicalVersion locks the MCP handshake to the single
// version source. Before this guard the handshake reported a hardcoded "0.9.0"
// while the binary was v1.2.0, so every client saw a version that did not
// exist.
func TestInitializeReportsCanonicalVersion(t *testing.T) {
	buf := &bytes.Buffer{}
	s := &Server{writer: json.NewEncoder(buf)}
	s.handleInitialize(jsonRPCRequest{ID: 1})

	var resp struct {
		Result struct {
			ServerInfo struct {
				Name    string `json:"name"`
				Version string `json:"version"`
			} `json:"serverInfo"`
		} `json:"result"`
	}
	if err := json.Unmarshal(buf.Bytes(), &resp); err != nil {
		t.Fatalf("decode initialize response %q: %v", buf.String(), err)
	}
	if resp.Result.ServerInfo.Version != version.Version {
		t.Errorf("serverInfo.version = %q, want %q (raw: %s)",
			resp.Result.ServerInfo.Version, version.Version, buf.String())
	}
	if resp.Result.ServerInfo.Name != "nyawa" {
		t.Errorf("serverInfo.name = %q, want %q", resp.Result.ServerInfo.Name, "nyawa")
	}
}

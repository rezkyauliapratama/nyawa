package mcp

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/store"
)

func newIngressTestStore(t *testing.T) *store.Store {
	t.Helper()
	st, err := store.NewStore(t.TempDir()+"/mcp-security-ingress.db", nil)
	if err != nil {
		t.Fatalf("open temporary SQLite store: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

// Keep the classifier-triggering fixture synthetic and split its marker so a
// test failure or source listing never contains a live-looking credential.
func syntheticMCPClassifiedContent() string {
	return "synthetic ingress fixture " + "s" + "k-" + strings.Repeat("A", 32)
}

func mcpStoredContentCount(t *testing.T, st *store.Store, content string) int {
	t.Helper()
	var count int
	if err := st.GetDB().QueryRow(`SELECT COUNT(*) FROM memories WHERE content = ?`, content).Scan(&count); err != nil {
		t.Fatalf("count stored test fixture: %v", err)
	}
	return count
}

func invokeInProcessStoreTool(t *testing.T, srv *Server, content, id string) []byte {
	t.Helper()
	params, err := json.Marshal(map[string]any{
		"name":      "nyawa_store",
		"arguments": map[string]string{"content": content},
	})
	if err != nil {
		t.Fatalf("marshal tool parameters: %v", err)
	}
	var output bytes.Buffer
	srv.writer = json.NewEncoder(&output)
	srv.handleRequest(jsonRPCRequest{JSONRPC: "2.0", ID: id, Method: "tools/call", Params: params})
	if output.Len() == 0 {
		t.Fatal("in-process MCP handler emitted no JSON-RPC response")
	}
	return output.Bytes()
}

func invokeMCPHTTPStore(t *testing.T, handler http.Handler, content, token string) (int, []byte) {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0",
		"id":      "security-regression",
		"method":  "tools/call",
		"params": map[string]any{
			"name":      "nyawa_store",
			"arguments": map[string]string{"content": content},
		},
	})
	if err != nil {
		t.Fatalf("marshal HTTP MCP request: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec.Code, rec.Body.Bytes()
}

func mcpResponseIndicatesRejection(response []byte) bool {
	var envelope struct {
		Error json.RawMessage `json:"error"`
		Result struct {
			IsError bool `json:"isError"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
	}
	if json.Unmarshal(response, &envelope) != nil {
		return false
	}
	if len(envelope.Error) != 0 && string(envelope.Error) != "null" {
		return true
	}
	if envelope.Result.IsError {
		return true
	}
	for _, item := range envelope.Result.Content {
		var result struct {
			Status string `json:"status"`
			Error  string `json:"error"`
		}
		if json.Unmarshal([]byte(item.Text), &result) == nil && (result.Status == "error" || result.Error != "") {
			return true
		}
	}
	return false
}

func TestMCPStoreIngressSecurityParity(t *testing.T) {
	classified := syntheticMCPClassifiedContent()
	const token = "test-only-mcp-ingress-token"

	t.Run("in-process classified", func(t *testing.T) {
		st := newIngressTestStore(t)
		srv := NewServer(st, nil, nil)
		response := invokeInProcessStoreTool(t, srv, classified, "in-process-classified")
		if !mcpResponseIndicatesRejection(response) {
			t.Error("in-process MCP classified store did not return an error/rejection indication")
		}
		if mcpStoredContentCount(t, st, classified) != 0 {
			t.Error("in-process MCP classified store persisted the fixture")
		}
		if bytes.Contains(response, []byte(classified)) {
			t.Error("in-process MCP classified store echoed the fixture")
		}
	})

	t.Run("authenticated HTTP classified", func(t *testing.T) {
		st := newIngressTestStore(t)
		handler, err := NewHTTPHandler(st, nil, nil, token)
		if err != nil {
			t.Fatalf("construct authenticated MCP HTTP handler: %v", err)
		}
		status, response := invokeMCPHTTPStore(t, handler, classified, token)
		if status != http.StatusOK || !mcpResponseIndicatesRejection(response) {
			t.Errorf("authenticated MCP HTTP classified store: status=%d, JSON-RPC rejection=%t; want HTTP 200 with an error/rejection indication", status, mcpResponseIndicatesRejection(response))
		}
		if mcpStoredContentCount(t, st, classified) != 0 {
			t.Error("authenticated MCP HTTP classified store persisted the fixture")
		}
		if bytes.Contains(response, []byte(classified)) {
			t.Error("authenticated MCP HTTP classified store echoed the fixture")
		}
	})

	t.Run("unauthenticated HTTP isolation", func(t *testing.T) {
		st := newIngressTestStore(t)
		handler, err := NewHTTPHandler(st, nil, nil, token)
		if err != nil {
			t.Fatalf("construct authenticated MCP HTTP handler: %v", err)
		}
		status, response := invokeMCPHTTPStore(t, handler, classified, "")
		if status != http.StatusUnauthorized || mcpStoredContentCount(t, st, classified) != 0 {
			t.Errorf("unauthenticated MCP HTTP store: status=%d; want 401 and no persistence", status)
		}
		if bytes.Contains(response, []byte(classified)) {
			t.Error("unauthenticated MCP HTTP store echoed the fixture")
		}
	})

	// Each benign control owns its own temporary DB and handler, so it proves
	// accepted storage without influencing any classified-input assertion.
	t.Run("benign in-process control", func(t *testing.T) {
		st := newIngressTestStore(t)
		srv := NewServer(st, nil, nil)
		benign := "ordinary in-process MCP regression control"
		_ = invokeInProcessStoreTool(t, srv, benign, "in-process-benign")
		if got := mcpStoredContentCount(t, st, benign); got != 1 {
			t.Errorf("benign in-process MCP control: rows=%d; want one row", got)
		}
	})

	t.Run("benign authenticated HTTP control", func(t *testing.T) {
		st := newIngressTestStore(t)
		handler, err := NewHTTPHandler(st, nil, nil, token)
		if err != nil {
			t.Fatalf("construct authenticated MCP HTTP handler: %v", err)
		}
		benign := "ordinary authenticated MCP HTTP regression control"
		status, _ := invokeMCPHTTPStore(t, handler, benign, token)
		if got := mcpStoredContentCount(t, st, benign); status != http.StatusOK || got != 1 {
			t.Errorf("benign authenticated MCP HTTP control: status=%d, rows=%d; want 200 and one row", status, got)
		}
	})
}

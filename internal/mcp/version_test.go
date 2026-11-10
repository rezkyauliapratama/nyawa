package mcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

func TestHTTPHandlerRequiresNonemptyToken(t *testing.T) {
	for _, token := range []string{"", " 	"} {
		if _, err := NewHTTPHandler(nil, nil, nil, token); err == nil {
			t.Fatalf("NewHTTPHandler accepted an empty/blank bearer token %q", token)
		}
	}
}

func TestHTTPHandlerAuthenticationAndRequestValidation(t *testing.T) {
	handler, err := NewHTTPHandler(nil, nil, nil, "unit-test-token")
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		method string
		token  string
		type_  string
		want   int
	}{
		{name: "missing bearer", method: http.MethodPost, type_: "application/json", want: http.StatusUnauthorized},
		{name: "wrong bearer", method: http.MethodPost, token: "wrong", type_: "application/json", want: http.StatusUnauthorized},
		{name: "method", method: http.MethodGet, token: "unit-test-token", type_: "application/json", want: http.StatusMethodNotAllowed},
		{name: "content type", method: http.MethodPost, token: "unit-test-token", type_: "text/plain", want: http.StatusUnsupportedMediaType},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, "/mcp", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
			req.Header.Set("Content-Type", tt.type_)
			if tt.token != "" {
				req.Header.Set("Authorization", "Bearer "+tt.token)
			}
			rw := httptest.NewRecorder()
			handler.ServeHTTP(rw, req)
			if rw.Code != tt.want {
				t.Fatalf("status = %d, want %d; body: %s", rw.Code, tt.want, rw.Body.String())
			}
		})
	}

	rw := sendHTTPRPC(handler, `{"jsonrpc":"2.0","id":3,"method":"initialize"}`)
	if rw.Code != http.StatusOK {
		t.Fatalf("initialize status = %d, want 200: %s", rw.Code, rw.Body.String())
	}
	if rw.Header().Get("Mcp-Session-Id") != "" {
		t.Fatalf("stateless initialize unexpectedly returned session ID %q", rw.Header().Get("Mcp-Session-Id"))
	}

	rw = sendHTTPRPC(handler, `{"jsonrpc":"2.0","id":4,"method":"not-a-method"}`)
	var rpcResp struct {
		ID    int       `json:"id"`
		Error *rpcError `json:"error"`
	}
	if err := json.Unmarshal(rw.Body.Bytes(), &rpcResp); err != nil {
		t.Fatalf("decode method error: %v; response %s", err, rw.Body.String())
	}
	if rw.Code != http.StatusOK || rpcResp.ID != 4 || rpcResp.Error == nil || rpcResp.Error.Code != -32601 {
		t.Fatalf("unexpected method error response: status=%d body=%s", rw.Code, rw.Body.String())
	}

	rw = sendHTTPRPC(handler, `{"jsonrpc":"2.0","id":5,"method":"ping"}`)
	var pingResp struct {
		ID     int            `json:"id"`
		Result map[string]any `json:"result"`
	}
	if err := json.Unmarshal(rw.Body.Bytes(), &pingResp); err != nil {
		t.Fatalf("decode ping response: %v; response %s", err, rw.Body.String())
	}
	if rw.Code != http.StatusOK || pingResp.ID != 5 || pingResp.Result == nil {
		t.Fatalf("unexpected ping response: status=%d body=%s", rw.Code, rw.Body.String())
	}

	rw = sendHTTPRPC(handler, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
	if rw.Code != http.StatusAccepted || rw.Body.Len() != 0 {
		t.Fatalf("notification response = status %d body %q, want 202 and empty", rw.Code, rw.Body.String())
	}
}

func TestHTTPHandlerConcurrentRequestsKeepIDsIsolated(t *testing.T) {
	handler, err := NewHTTPHandler(nil, nil, nil, "unit-test-token")
	if err != nil {
		t.Fatal(err)
	}

	const count = 24
	var wg sync.WaitGroup
	errs := make(chan error, count)
	for i := 0; i < count; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			body := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"method":"tools/list"}`, id)
			rw := sendHTTPRPC(handler, body)
			var response struct {
				ID     int `json:"id"`
				Result struct {
					Tools []toolDefinition `json:"tools"`
				} `json:"result"`
			}
			if rw.Code != http.StatusOK {
				errs <- fmt.Errorf("request %d status=%d body=%s", id, rw.Code, rw.Body.String())
				return
			}
			if err := json.Unmarshal(rw.Body.Bytes(), &response); err != nil {
				errs <- fmt.Errorf("request %d decode: %w", id, err)
				return
			}
			if response.ID != id || len(response.Result.Tools) == 0 {
				errs <- fmt.Errorf("request %d received response id=%d tools=%d", id, response.ID, len(response.Result.Tools))
			}
		}(i + 1)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func sendHTTPRPC(handler http.Handler, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer unit-test-token")
	rw := httptest.NewRecorder()
	handler.ServeHTTP(rw, req)
	return rw
}

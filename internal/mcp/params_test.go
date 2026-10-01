package mcp

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/rag"
	"github.com/rezkyauliapratama/nyawa/internal/store"
)

// toolArgStructs maps every declared MCP tool that takes arguments to a
// constructor for the arg struct its handler decodes into. It must stay in
// sync with handleToolCall; TestDeclaredToolPropertiesBindToArgStructs fails
// if a tool with declared properties is missing here.
func toolArgStructs() map[string]func() any {
	return map[string]func() any{
		"nyawa_store":           func() any { return &storeArgs{} },
		"nyawa_recall":          func() any { return &recallArgs{} },
		"nyawa_list":            func() any { return &listArgs{} },
		"nyawa_forget":          func() any { return &forgetArgs{} },
		"rag_create_collection": func() any { return &ragCreateCollectionArgs{} },
		"rag_delete_collection": func() any { return &ragDeleteCollectionArgs{} },
		"rag_ingest_file":       func() any { return &ragIngestFileArgs{} },
		"rag_query":             func() any { return &ragQueryArgs{} },
		"nyawa_graph_query":     func() any { return &graphQueryArgs{} },
		"nyawa_graph_entities":  func() any { return &graphEntitiesArgs{} },
		"nyawa_graph_path":      func() any { return &graphPathArgs{} },
		"compact_context":       func() any { return &compactArgs{} },
	}
}

// sampleFor returns a distinctive value for a declared property so that the
// round-tripped JSON can be searched for it without knowing the Go field name.
// The value dtype follows the declared schema type; sending a string for a
// numeric property (or a scalar for an array property) would make
// encoding/json reject the payload outright.
func sampleFor(prop, schemaType string) any {
	switch schemaType {
	case "number", "integer", "boolean":
		return 4242
	case "array":
		return []any{"sample_" + prop}
	}
	return "sample_" + prop
}

// TestDeclaredToolPropertiesBindToArgStructs is the regression guard for the
// class of bug where a snake_case property declared in a tool's InputSchema
// (e.g. file_path, top_k, max_depth) never populates the handler's arg struct
// because encoding/json matches field names case-insensitively but an
// underscore breaks the match when the field has no explicit json tag.
func TestDeclaredToolPropertiesBindToArgStructs(t *testing.T) {
	s := &Server{}
	structs := toolArgStructs()
	for _, tool := range s.tools() {
		if len(tool.InputSchema.Properties) == 0 {
			continue
		}
		newArgs, ok := structs[tool.Name]
		if !ok {
			t.Errorf("tool %q declares properties but no arg struct is registered in toolArgStructs", tool.Name)
			continue
		}
		for prop, schema := range tool.InputSchema.Properties {
			prop, schema := prop, schema
			t.Run(tool.Name+"/"+prop, func(t *testing.T) {
				sample := sampleFor(prop, schema.Type)
				payload, err := json.Marshal(map[string]any{prop: sample})
				if err != nil {
					t.Fatalf("marshal payload: %v", err)
				}
				args := newArgs()
				if err := json.Unmarshal(payload, args); err != nil {
					t.Fatalf("unmarshal %s into %T: %v", payload, args, err)
				}
				out, err := json.Marshal(args)
				if err != nil {
					t.Fatalf("marshal args: %v", err)
				}
				marker, _ := json.Marshal(sample)
				if !strings.Contains(string(out), string(marker)) {
					t.Fatalf("declared property %q did not bind: sent %s, arg struct decoded to %s", prop, payload, out)
				}
			})
		}
	}
}

// TestRAGIngestFileHandlerCanonicalArg proves the end-to-end handler accepts
// the declared "file_path" property instead of failing with "file_path
// required".
func TestRAGIngestFileHandlerCanonicalArg(t *testing.T) {
	srv, buf := newTestServer(t)
	path := writeScratchFile(t)

	args, _ := json.Marshal(map[string]any{"file_path": path, "collection": "scratch_test"})
	srv.handleRAGIngestFile(1, args)

	status, chunks := ingestResultFrom(t, buf)
	if status != "ingested" {
		t.Fatalf("handleRAGIngestFile(status) = %q, want %q (raw: %s)", status, "ingested", buf.String())
	}
	if chunks <= 0 {
		t.Fatalf("handleRAGIngestFile chunk count = %d, want > 0 (raw: %s)", chunks, buf.String())
	}
}

// TestRAGIngestFileHandlerLegacyFilePathAlias proves the pre-v1.2.0
// single-word spelling ("filepath") still works for existing clients.
func TestRAGIngestFileHandlerLegacyFilePathAlias(t *testing.T) {
	srv, buf := newTestServer(t)
	path := writeScratchFile(t)

	args, _ := json.Marshal(map[string]any{"filepath": path, "collection": "scratch_test"})
	srv.handleRAGIngestFile(1, args)

	status, chunks := ingestResultFrom(t, buf)
	if status != "ingested" {
		t.Fatalf("handleRAGIngestFile(legacy filepath) status = %q, want %q (raw: %s)", status, "ingested", buf.String())
	}
	if chunks <= 0 {
		t.Fatalf("handleRAGIngestFile(legacy filepath) chunk count = %d, want > 0 (raw: %s)", chunks, buf.String())
	}
}

// TestRAGIngestFileHandlerMissingPathStillFails keeps the required-field
// validation honest: an empty payload must still be rejected.
func TestRAGIngestFileHandlerMissingPathStillFails(t *testing.T) {
	srv, buf := newTestServer(t)
	srv.handleRAGIngestFile(1, json.RawMessage(`{"collection":"scratch_test"}`))
	if !strings.Contains(buf.String(), "file_path required") {
		t.Fatalf("empty payload should be rejected with file_path required, got: %s", buf.String())
	}
}

// TestLegacyAliasesResolve covers the explicit alias fields added for
// backwards compatibility with the pre-v1.2.0 spellings.
func TestLegacyAliasesResolve(t *testing.T) {
	ingest := ragIngestFileArgs{}
	_ = json.Unmarshal([]byte(`{"filepath":"/tmp/legacy.txt","collection":"c"}`), &ingest)
	ingest.resolve()
	if ingest.FilePath != "/tmp/legacy.txt" {
		t.Errorf("filepath alias not resolved: %+v", ingest)
	}

	query := ragQueryArgs{}
	_ = json.Unmarshal([]byte(`{"query":"q","topK":7}`), &query)
	query.resolve()
	if query.TopK != 7 {
		t.Errorf("topK alias not resolved: %+v", query)
	}

	col := ragCreateCollectionArgs{}
	_ = json.Unmarshal([]byte(`{"name":"n","chunkSize":250}`), &col)
	col.resolve()
	if col.ChunkSize != 250 {
		t.Errorf("chunkSize alias not resolved: %+v", col)
	}

	pathArgs := graphPathArgs{}
	_ = json.Unmarshal([]byte(`{"source":"a","target":"b","maxDepth":6}`), &pathArgs)
	pathArgs.resolve()
	if pathArgs.MaxDepth != 6 {
		t.Errorf("maxDepth alias not resolved: %+v", pathArgs)
	}

	// Canonical spellings keep working and are not overwritten by the alias.
	canonical := ragQueryArgs{}
	_ = json.Unmarshal([]byte(`{"query":"q","top_k":3,"topK":9}`), &canonical)
	canonical.resolve()
	if canonical.TopK != 3 {
		t.Errorf("canonical top_k should win over alias: %+v", canonical)
	}
}

// ─── helpers ──────────────────────────────────────

func newTestServer(t *testing.T) (*Server, *bytes.Buffer) {
	t.Helper()
	st, err := store.NewStore(filepath.Join(t.TempDir(), "mcp-params.db"), nil)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { st.Close() })
	rs := rag.NewRAGStore(st.GetDB(), st.GetHNSW(), st.GetHNSWPath(), nil)
	buf := &bytes.Buffer{}
	return &Server{store: st, ragStore: rs, writer: json.NewEncoder(buf)}, buf
}

func writeScratchFile(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "scratch.md")
	if err := os.WriteFile(path, []byte("# Scratch\n\nNyawa parameter binding regression fixture.\n"), 0o644); err != nil {
		t.Fatalf("write scratch file: %v", err)
	}
	return path
}

// ingestResultFrom decodes the JSON-RPC envelope written by writeToolResult
// and returns the status and chunk count reported by the ingest handler.
func ingestResultFrom(t *testing.T, buf *bytes.Buffer) (string, int) {
	t.Helper()
	var resp struct {
		Result struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"result"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(buf.Bytes(), &resp); err != nil {
		t.Fatalf("decode JSON-RPC response %q: %v", buf.String(), err)
	}
	if resp.Error != nil {
		t.Fatalf("handler returned error: %s", resp.Error.Message)
	}
	if len(resp.Result.Content) == 0 {
		t.Fatalf("no content in response: %s", buf.String())
	}
	// ingestResult in production has no json tags today (Go field names) but we
	// accept either spelling so adding tags later does not break the test.
	raw := map[string]any{}
	if err := json.Unmarshal([]byte(resp.Result.Content[0].Text), &raw); err != nil {
		t.Fatalf("decode tool payload %q: %v", resp.Result.Content[0].Text, err)
	}
	status, _ := raw["status"].(string)
	if status == "" {
		status, _ = raw["Status"].(string)
	}
	chunks := 0
	for _, key := range []string{"chunk_count", "ChunkCount"} {
		if c, ok := raw[key].(float64); ok {
			chunks = int(c)
			break
		}
	}
	return status, chunks
}

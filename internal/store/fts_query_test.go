package store

import (
	"strings"
	"testing"

	"github.com/rezkyauliapratama/nyawa/internal/types"
)

func TestBuildFTSMatchExpr(t *testing.T) {
	cases := []struct {
		in   string
		want []string // expected tokens, in order
	}{
		{"pembersihan docker massal volume crypto-data nyawa-gomod",
			[]string{"pembersihan", "docker", "massal", "volume", "crypto", "data", "nyawa", "gomod"}},
		{"  Hello, WORLD!  ", []string{"hello", "world"}},
		{"a b cc", []string{"cc"}},       // 1-char tokens dropped
		{"dup dup DUP", []string{"dup"}}, // de-duplicated, lower-cased
		{"", nil},
		{"!!! ???", nil},
	}
	for _, tc := range cases {
		expr := buildFTSMatchExpr(tc.in)
		if len(tc.want) == 0 {
			if expr != "" {
				t.Errorf("buildFTSMatchExpr(%q) = %q, want empty", tc.in, expr)
			}
			continue
		}
		got := strings.Split(expr, " OR ")
		if len(got) != len(tc.want) {
			t.Fatalf("buildFTSMatchExpr(%q) = %q, want %d tokens", tc.in, expr, len(tc.want))
		}
		for i, w := range tc.want {
			if got[i] != `"`+w+`"` {
				t.Errorf("token %d = %q, want %q", i, got[i], `"`+w+`"`)
			}
		}
	}
}

// An FTS5 MATCH fed the raw query string is brittle: whitespace means implicit
// AND (matching almost nothing for a multi-word query) and '-' is parsed as
// query syntax. The rewrite must make such queries return the relevant short
// memory again.
func TestFTS5SearchRawQueryRegression(t *testing.T) {
	s, err := NewStore(":memory:", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	mustInsert := func(id, content string, typ types.MemoryType) {
		t.Helper()
		if err := s.InsertMemory(&types.Memory{ID: id, Content: content, Type: typ, Namespace: "test"}); err != nil {
			t.Fatal(err)
		}
	}
	mustInsert("rule", "docker volume prune massal menghapus crypto data milik stack lain", types.TypeRule)
	mustInsert("build", "cara build image docker tanpa cache", types.TypeInsight)
	mustInsert("other", "catatan rapat mingguan tim", types.TypeNote)

	// Raw query with the hyphen: rewrite must not error and must surface the rule.
	ids, err := s.FTS5Search("docker volume crypto-data", 10, "test")
	if err != nil {
		t.Fatalf("FTS5Search returned error: %v", err)
	}
	if len(ids) == 0 {
		t.Fatal("expected results for hyphenated query, got none")
	}
	if ids[0] != "rule" {
		t.Errorf("best hit = %q, want rule (got %v)", ids[0], ids)
	}
	if !contains(ids, "build") {
		t.Errorf("expected the docker build memory to surface via the OR rewrite, got %v", ids)
	}
	if contains(ids, "other") {
		t.Errorf("unrelated memory should not match, got %v", ids)
	}

	// Implicit-AND behaviour of the old code: the three-term query matched
	// nothing. With OR rewriting it must return results.
	ids, err = s.FTS5Search("docker volume zzzznotaword", 10, "test")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Error("OR rewrite should still match on the shared terms")
	}
}

func TestFTS5SearchLegacyToggle(t *testing.T) {
	s, err := NewStore(":memory:", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.InsertMemory(&types.Memory{ID: "m1", Content: "docker volume prune", Type: types.TypeNote, Namespace: "test"}); err != nil {
		t.Fatal(err)
	}

	t.Setenv("NYAWA_RECALL_V2", "0")
	// Legacy path: raw query is an implicit AND; these three terms never all
	// appear, so the old behaviour returns nothing.
	ids, err := s.FTS5Search("docker volume zzzznotaword", 10, "test")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 0 {
		t.Errorf("legacy AND query should match nothing, got %v", ids)
	}

	t.Setenv("NYAWA_RECALL_V2", "1")
	ids, err = s.FTS5Search("docker volume zzzznotaword", 10, "test")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) == 0 {
		t.Error("v2 OR query should match on shared terms")
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

package server

import "testing"

func TestResolveCompactNamespace(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"empty defaults to context", "", "context"},
		{"explicit namespace is used", "baseline", "baseline"},
		{"explicit context stays context", "context", "context"},
		{"arbitrary namespace is preserved", "session-42", "session-42"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveCompactNamespace(tc.in); got != tc.want {
				t.Errorf("resolveCompactNamespace(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

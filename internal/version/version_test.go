package version

import "testing"

// TestNumberReportsInjectedValue proves that whatever the release workflow
// injects with -X ...version.Version is what the CLI/banner/endpoints report,
// and that a leading "v" from the git tag is normalised away.
func TestNumberReportsInjectedValue(t *testing.T) {
	orig := Version
	defer func() { Version = orig }()

	cases := []struct{ injected, want string }{
		{"1.2.1", "1.2.1"},
		{"v1.2.1", "1.2.1"},
		{" 1.2.1 ", "1.2.1"},
		{"v2.0.0-rc.1", "2.0.0-rc.1"},
	}
	for _, tc := range cases {
		Version = tc.injected
		if got := Number(); got != tc.want {
			t.Errorf("Number() with injected %q = %q, want %q", tc.injected, got, tc.want)
		}
		if got, want := String(), "nyawa v"+tc.want; got != want {
			t.Errorf("String() with injected %q = %q, want %q", tc.injected, got, want)
		}
	}
}

// TestNumberFallsBackToDefaultWhenNotInjected proves a build without ldflags
// (and a blank injection) reports the safe default rather than an empty or
// stale release number.
func TestNumberFallsBackToDefaultWhenNotInjected(t *testing.T) {
	orig := Version
	defer func() { Version = orig }()

	if DefaultVersion == "" {
		t.Fatal("DefaultVersion must not be empty")
	}
	for _, injected := range []string{"", "   ", "v"} {
		Version = injected
		if got := Number(); got != DefaultVersion {
			t.Errorf("Number() with empty injection %q = %q, want default %q", injected, got, DefaultVersion)
		}
	}
}

// TestDefaultVersionIsNotAReleaseTag guards the default: it must stay a
// non-release placeholder so an uninjected binary cannot lie to operators.
func TestDefaultVersionIsNotAReleaseTag(t *testing.T) {
	if DefaultVersion != "dev" {
		t.Errorf("DefaultVersion = %q, want %q — do not hardcode a release number", DefaultVersion, "dev")
	}
	if Version != DefaultVersion {
		t.Errorf("Version = %q under go test (no ldflags), want DefaultVersion %q", Version, DefaultVersion)
	}
}

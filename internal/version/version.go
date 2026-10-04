// Package version is the single source of truth for the version Nyawa reports
// about itself: the CLI (`nyawa version`, usage banner), the MCP handshake
// (serverInfo.version) and the HTTP endpoints (/status, /v1/stats, /health).
//
// The release version is injected at build time from the pushed git tag, so the
// reported version always matches the release artifact. It is no longer a
// hardcoded literal that had to be bumped by hand — and could silently be
// forgotten, which is how v1.2.1 shipped still reporting v1.2.0.
//
// It deliberately has no dependencies so any package can import it without
// risking an import cycle.
package version

import "strings"

// DefaultVersion is the version reported by a build that did not inject one via
// -ldflags (`go build`, `go test`, IDE builds). It is intentionally not a
// release number: an uninjected binary must not masquerade as a published
// release.
const DefaultVersion = "dev"

// Version is the injection slot and the ONLY version literal in the codebase.
// The release workflow overwrites it at link time with the pushed git tag:
//
//	go build -tags sqlite_fts5 \
//	  -ldflags "-X github.com/rezkyauliapratama/nyawa/internal/version.Version=1.2.1" \
//	  -o nyawa-linux-amd64 ./cmd/nyawa/
//
// Local builds leave it at DefaultVersion. Do not hardcode a release number
// here again.
var Version = DefaultVersion

// Number returns the bare semantic version (no leading "v") that is reported
// everywhere. A leading "v" is stripped so an injected tag "v1.2.1" and a
// literal "1.2.1" agree; an empty/blank injection falls back to DefaultVersion.
func Number() string {
	v := strings.TrimSpace(Version)
	v = strings.TrimPrefix(v, "v")
	if v == "" {
		return DefaultVersion
	}
	return v
}

// String returns the canonical display form used by the CLI.
func String() string { return "nyawa v" + Number() }

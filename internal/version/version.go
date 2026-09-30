// Package version is the single source of truth for the version Nyawa reports
// about itself: the CLI (`nyawa version`, usage banner), the MCP handshake
// (serverInfo.version) and the HTTP endpoints (/status, /v1/stats, /health).
//
// It deliberately has no dependencies so any package can import it without
// risking an import cycle.
package version

// Version is the semantic version of this build. This is the only place the
// number lives; bump it here when cutting a release.
const Version = "1.2.0"

// String returns the canonical display form used by the CLI.
func String() string { return "nyawa v" + Version }

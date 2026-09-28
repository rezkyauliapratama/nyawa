//go:build !unix

package index

import "time"

// fileLock is a no-op stand-in for platforms without flock(2). Save and Load
// remain atomic on these platforms (temp file + rename); only the
// cross-process mutual exclusion is unavailable.
type fileLock struct{}

// acquireFileLock always succeeds on platforms that lack flock(2).
func acquireFileLock(path string, timeout time.Duration) (*fileLock, error) {
	return &fileLock{}, nil
}

func (l *fileLock) release() {}

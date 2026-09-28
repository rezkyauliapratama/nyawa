//go:build unix

package index

import (
	"fmt"
	"os"
	"syscall"
	"time"
)

// fileLock is an advisory, cross-process exclusive lock backed by flock(2) on
// a dedicated lock file. The lock is held for as long as the underlying file
// descriptor is open and is released explicitly via release (or implicitly
// when the process exits, which makes it crash-safe).
type fileLock struct {
	f *os.File
}

// acquireFileLock opens path (creating it if needed) and takes an exclusive
// flock on it, retrying with a short backoff until timeout elapses. It never
// blocks indefinitely: if the lock cannot be obtained in time a descriptive
// error is returned.
func acquireFileLock(path string, timeout time.Duration) (*fileLock, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, fmt.Errorf("hnsw lock: open %s: %w", path, err)
	}
	deadline := time.Now().Add(timeout)
	for {
		err = syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		if err == nil {
			return &fileLock{f: f}, nil
		}
		if err != syscall.EWOULDBLOCK {
			f.Close()
			return nil, fmt.Errorf("hnsw lock: flock %s: %w", path, err)
		}
		if !time.Now().Before(deadline) {
			f.Close()
			return nil, fmt.Errorf("hnsw lock: timed out after %s waiting for %s (held by another process)", timeout, path)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

// release drops the flock and closes the lock file. It is safe to call on a
// nil receiver, which lets callers unconditionally defer (*fileLock).release().
func (l *fileLock) release() {
	if l == nil || l.f == nil {
		return
	}
	syscall.Flock(int(l.f.Fd()), syscall.LOCK_UN)
	l.f.Close()
}

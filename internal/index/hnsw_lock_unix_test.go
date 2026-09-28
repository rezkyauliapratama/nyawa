//go:build unix

package index

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// TestCrossProcessLockExcludes verifies that the flock based index lock
// actually excludes a second process: a helper subprocess holds the lock while
// the parent tries (and fails within the timeout) to take it.
func TestCrossProcessLockExcludes(t *testing.T) {
	if os.Getenv("HNSW_LOCK_HELPER") == "1" {
		t.Skip("running as lock helper subprocess")
	}

	dir := t.TempDir()
	lockPath := dir + "/idx.hnsw.lock"

	cmd := exec.Command(os.Args[0], "-test.run=TestLockHolderHelper", "-test.v")
	cmd.Env = append(os.Environ(), "HNSW_LOCK_HELPER=1", "HNSW_LOCK_PATH="+lockPath)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper: %v", err)
	}

	locked := make(chan struct{})
	go func() {
		sc := bufio.NewScanner(stdout)
		for sc.Scan() {
			if strings.Contains(sc.Text(), "helper-locked") {
				close(locked)
				return
			}
		}
	}()

	select {
	case <-locked:
	case <-time.After(10 * time.Second):
		cmd.Process.Kill()
		t.Fatal("helper subprocess did not report acquiring the lock")
	}

	start := time.Now()
	lk, err := acquireFileLock(lockPath, 300*time.Millisecond)
	if err == nil {
		lk.release()
		cmd.Process.Kill()
		t.Fatal("expected to fail acquiring a lock held by another process, but succeeded")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("expected a timeout error, got: %v", err)
	}
	if elapsed := time.Since(start); elapsed < 250*time.Millisecond {
		t.Errorf("returned too early (%v); should have waited until the timeout", elapsed)
	}

	if err := cmd.Wait(); err != nil {
		t.Fatalf("helper exited with error: %v", err)
	}

	// Once the helper exits its flock is released and we can acquire it.
	lk2, err := acquireFileLock(lockPath, 5*time.Second)
	if err != nil {
		t.Fatalf("expected to acquire lock after helper exit: %v", err)
	}
	lk2.release()
}

// TestLockHolderHelper is the subprocess body used by
// TestCrossProcessLockExcludes. It is inert in a normal test run.
func TestLockHolderHelper(t *testing.T) {
	if os.Getenv("HNSW_LOCK_HELPER") != "1" {
		t.Skip("not a lock helper subprocess")
	}
	lockPath := os.Getenv("HNSW_LOCK_PATH")
	lk, err := acquireFileLock(lockPath, 5*time.Second)
	if err != nil {
		t.Fatalf("helper acquire: %v", err)
	}
	defer lk.release()
	fmt.Println("helper-locked")
	time.Sleep(1500 * time.Millisecond)
}

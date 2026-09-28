package index

import "time"

// lockSuffix is appended to an HNSW index path to derive the cross-process
// lock file path (e.g. "memory.db.hnsw" -> "memory.db.hnsw.lock").
const lockSuffix = ".lock"

// fileLockTimeout bounds how long Save/Load wait to acquire the cross-process
// lock before giving up. It is a package variable so tests can shorten it.
var fileLockTimeout = 30 * time.Second

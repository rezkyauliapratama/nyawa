package rag

import (
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/rezkyauliapratama/nyawa/internal/index"
)

const ragIndexManifestVersion = 2

var (
	ErrRAGIndexManifestMissing = errors.New("RAG index manifest missing")
	ErrRAGIndexProfileMismatch = errors.New("RAG index profile mismatch")
)

// EmbeddingProfile identifies a vector space without storing credentials.
type EmbeddingProfile struct {
	Provider    string `json:"provider"`
	Model       string `json:"model"`
	BaseURL     string `json:"base_url,omitempty"`
	Dimensions  int    `json:"dimensions"`
	QueryTask   string `json:"query_task,omitempty"`
	PassageTask string `json:"passage_task,omitempty"`
}

type ragIndexManifest struct {
	Version   int
	IndexPath string
	Profile   EmbeddingProfile
}

// OpenVectorIndex opens a RAG-only HNSW index and binds it to a SQLite
// manifest. It never adopts a non-empty unmanifested index and never reuses the
// memory HNSW implicitly. SQLite serializes concurrent manifest initialization.
func OpenVectorIndex(db *sql.DB, path string, profile EmbeddingProfile) (*index.HNSW, error) {
	if db == nil {
		return nil, fmt.Errorf("RAG metadata database is required")
	}
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("RAG index path is required")
	}
	profile, err := normalizeEmbeddingProfile(profile)
	if err != nil {
		return nil, err
	}

	h := index.NewHNSW(index.DefaultHNSWConfig(profile.Dimensions))
	h.EnableRAGBoundedDegree()
	loadErr := h.Load(path)
	indexMissing := errors.Is(loadErr, os.ErrNotExist)
	if loadErr != nil && !indexMissing {
		return nil, fmt.Errorf("load RAG HNSW %s: %w", path, loadErr)
	}
	if !indexMissing {
		if err := h.EnsureDimension(profile.Dimensions); err != nil {
			return nil, fmt.Errorf("RAG HNSW dimension: %w", err)
		}
	}

	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS rag_index_manifest (
		singleton INTEGER PRIMARY KEY CHECK(singleton=1),
		version INTEGER NOT NULL,
		index_path TEXT NOT NULL,
		provider TEXT NOT NULL,
		model TEXT NOT NULL,
		base_url TEXT NOT NULL DEFAULT '',
		dimensions INTEGER NOT NULL,
		query_task TEXT NOT NULL DEFAULT '',
		passage_task TEXT NOT NULL DEFAULT '')`); err != nil {
		return nil, fmt.Errorf("create RAG index manifest: %w", err)
	}
	if err := ensureManifestTaskColumns(db); err != nil {
		return nil, fmt.Errorf("migrate RAG task profile: %w", err)
	}

	tx, err := db.Begin()
	if err != nil {
		return nil, fmt.Errorf("begin RAG manifest check: %w", err)
	}
	defer tx.Rollback()

	result, err := tx.Exec(`INSERT OR IGNORE INTO rag_index_manifest(singleton,version,index_path,provider,model,base_url,dimensions,query_task,passage_task) VALUES(1,?,?,?,?,?,?,?,?)`,
		ragIndexManifestVersion, path, profile.Provider, profile.Model, profile.BaseURL, profile.Dimensions, profile.QueryTask, profile.PassageTask)
	if err != nil {
		return nil, fmt.Errorf("insert RAG index manifest: %w", err)
	}
	inserted, err := result.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("inspect RAG index manifest insert: %w", err)
	}
	manifest, err := readRAGIndexManifest(tx)
	if err != nil {
		return nil, fmt.Errorf("read RAG index manifest: %w", err)
	}
	if manifest.Version != ragIndexManifestVersion || manifest.IndexPath != path || manifest.Profile != profile {
		return nil, fmt.Errorf("%w: stored=%+v configured_path=%q configured=%+v", ErrRAGIndexProfileMismatch, manifest, path, profile)
	}
	if inserted > 0 && !indexMissing && h.Size() > 0 {
		return nil, fmt.Errorf("%w: refusing to adopt %d vectors without provider/model metadata", ErrRAGIndexManifestMissing, h.Size())
	}

	if indexMissing {
		// Another process may have initialized the file while this process was
		// waiting for the serialized SQLite manifest write.
		if inserted == 0 {
			if err := h.Load(path); err != nil {
				if errors.Is(err, os.ErrNotExist) {
					return nil, fmt.Errorf("%w: manifest exists but HNSW file %s is missing", ErrRAGIndexProfileMismatch, path)
				}
				return nil, fmt.Errorf("load concurrently initialized RAG HNSW: %w", err)
			}
			if err := h.EnsureDimension(profile.Dimensions); err != nil {
				return nil, fmt.Errorf("RAG HNSW dimension: %w", err)
			}
		} else if err := h.Save(path); err != nil {
			return nil, fmt.Errorf("initialize empty RAG HNSW %s: %w", path, err)
		}
	}
	// The path, provider, model, dimensions, and task profile have now all
	// matched the manifest. Only this point may mutate a legacy HNSW file.
	if err := h.MigrateLegacyRAGGraph(path); err != nil {
		return nil, fmt.Errorf("migrate legacy RAG HNSW: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit RAG index manifest: %w", err)
	}
	return h, nil
}

func ensureManifestTaskColumns(db *sql.DB) error {
	rows, err := db.Query(`PRAGMA table_info(rag_index_manifest)`)
	if err != nil {
		return err
	}
	columns := map[string]bool{}
	for rows.Next() {
		var cid, notNull, primaryKey int
		var name, columnType string
		var defaultValue any
		if err := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); err != nil {
			rows.Close()
			return err
		}
		columns[name] = true
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return err
	}
	if err := rows.Close(); err != nil {
		return err
	}
	for _, column := range []string{"query_task", "passage_task"} {
		if columns[column] {
			continue
		}
		if _, alterErr := db.Exec(`ALTER TABLE rag_index_manifest ADD COLUMN ` + column + ` TEXT NOT NULL DEFAULT ''`); alterErr != nil {
			// Concurrent initializers can both observe the old schema. Accept an
			// ALTER failure only if a fresh schema read proves the other process
			// added this exact column.
			rows, checkErr := db.Query(`PRAGMA table_info(rag_index_manifest)`)
			if checkErr != nil {
				return fmt.Errorf("alter manifest column %s and recheck schema: %w (alter: %v)", column, checkErr, alterErr)
			}
			found := false
			for rows.Next() {
				var cid, notNull, primaryKey int
				var name, columnType string
				var defaultValue any
				if scanErr := rows.Scan(&cid, &name, &columnType, &notNull, &defaultValue, &primaryKey); scanErr != nil {
					rows.Close()
					return scanErr
				}
				if name == column {
					found = true
				}
			}
			rowsErr := rows.Err()
			closeErr := rows.Close()
			if rowsErr != nil {
				return rowsErr
			}
			if closeErr != nil {
				return closeErr
			}
			if !found {
				return alterErr
			}
		}
	}
	return nil
}

func normalizeEmbeddingProfile(profile EmbeddingProfile) (EmbeddingProfile, error) {
	profile.Provider = strings.TrimSpace(profile.Provider)
	profile.Model = strings.TrimSpace(profile.Model)
	profile.QueryTask = strings.TrimSpace(profile.QueryTask)
	profile.PassageTask = strings.TrimSpace(profile.PassageTask)
	if profile.Provider == "" || profile.Model == "" {
		return EmbeddingProfile{}, fmt.Errorf("RAG embedder provider and model are required")
	}
	if profile.Dimensions <= 0 {
		return EmbeddingProfile{}, fmt.Errorf("RAG embedder dimensions must be positive")
	}
	if raw := strings.TrimSpace(profile.BaseURL); raw != "" {
		u, err := url.Parse(raw)
		if err != nil || u.Host == "" || (u.Scheme != "http" && u.Scheme != "https") {
			return EmbeddingProfile{}, fmt.Errorf("RAG embedder base URL must be absolute HTTP(S)")
		}
		// Credentials and query parameters must never be persisted in metadata.
		u.User = nil
		u.RawQuery = ""
		u.Fragment = ""
		profile.BaseURL = strings.TrimRight(u.String(), "/")
	}
	return profile, nil
}

func readRAGIndexManifest(tx *sql.Tx) (ragIndexManifest, error) {
	var manifest ragIndexManifest
	err := tx.QueryRow(`SELECT version,index_path,provider,model,base_url,dimensions,query_task,passage_task FROM rag_index_manifest WHERE singleton=1`).Scan(
		&manifest.Version, &manifest.IndexPath, &manifest.Profile.Provider, &manifest.Profile.Model, &manifest.Profile.BaseURL, &manifest.Profile.Dimensions, &manifest.Profile.QueryTask, &manifest.Profile.PassageTask)
	if err != nil {
		return ragIndexManifest{}, err
	}
	manifest.Profile, err = normalizeEmbeddingProfile(manifest.Profile)
	return manifest, err
}

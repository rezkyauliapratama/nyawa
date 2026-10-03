// ftsprobe — mereproduksi leg BM25 FTS5 persis seperti mesin Nyawa.
//
// Mesin (internal/store/sqlite.go, FTS5Search) mengoper teks kueri MENTAH ke
// `memories_fts MATCH ?` ORDER BY rank, tanpa sanitasi. Probe ini menjalankan
// SQL yang sama memakai driver SQLite yang sama (mattn/go-sqlite3 + tag
// sqlite_fts5) supaya peringkat BM25 yang dilaporkan mencerminkan perilaku
// runtime, bukan perilaku build SQLite lain.
//
// Pakai (dari root repo):
//   go run -tags sqlite_fts5 ./scripts/recall-bench/ftsprobe <queries.tsv> [--ns ns]
//
// Keluaran: tabel per kueri + JSON ke stdout.
package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

const dbPath = "/opt/data/.nyawa/memory.db"

type qrow struct {
	Query   string   `json:"query"`
	Targets []string `json:"targets"`
	NS      string   `json:"ns"`
}

type result struct {
	Query       string            `json:"query"`
	Expr        string            `json:"expr"`
	OK          bool              `json:"ok"`
	Error       string            `json:"error,omitempty"`
	NResults    int               `json:"n_results"`
	PerTarget   map[string]*int   `json:"per_target_rank"`
	TopPreview  []string          `json:"top_ids"`
}

func loadQueries(path string) ([]qrow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []qrow
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) < 2 {
			continue
		}
		var targets []string
		for _, t := range strings.Split(parts[1], ",") {
			if t = strings.TrimSpace(t); t != "" {
				targets = append(targets, t)
			}
		}
		ns := ""
		if len(parts) > 2 {
			ns = strings.TrimSpace(parts[2])
		}
		rows = append(rows, qrow{Query: parts[0], Targets: targets, NS: ns})
	}
	return rows, nil
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: ftsprobe <queries.tsv> [--ns ns]")
		os.Exit(2)
	}
	ns := ""
	if len(os.Args) >= 4 && os.Args[2] == "--ns" {
		ns = os.Args[3]
	}
	rows, err := loadQueries(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "load queries:", err)
		os.Exit(1)
	}
	db, err := sql.Open("sqlite3", "file:"+dbPath+"?mode=ro")
	if err != nil {
		fmt.Fprintln(os.Stderr, "open db:", err)
		os.Exit(1)
	}
	defer db.Close()

	limit := 50 // = searchTopK mesin untuk limit 10 (forceMin(limit*3, 50))
	var out []result
	for _, row := range rows {
		useNS := row.NS
		if useNS == "" {
			useNS = ns
		}
		q := "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid WHERE memories_fts MATCH ? AND m.superseded_at IS NULL"
		args := []any{row.Query}
		if useNS != "" {
			q += " AND m.namespace=?"
			args = append(args, useNS)
		}
		q += " ORDER BY rank LIMIT ?"
		args = append(args, limit)

		res := result{Query: row.Query, Expr: row.Query, PerTarget: map[string]*int{}, OK: true}
		rws, err := db.Query(q, args...)
		if err != nil {
			res.OK = false
			res.Error = err.Error()
		} else {
			ids := []string{}
			for rws.Next() {
				var id string
				rws.Scan(&id)
				ids = append(ids, id)
			}
			rws.Close()
			res.NResults = len(ids)
			for i, id := range ids {
				if len(res.TopPreview) < 5 {
					res.TopPreview = append(res.TopPreview, id)
				}
				for _, t := range row.Targets {
					if id == t {
						rank := i + 1
						if _, seen := res.PerTarget[t]; !seen {
							res.PerTarget[t] = &rank
						}
					}
				}
			}
			for _, t := range row.Targets {
				if _, ok := res.PerTarget[t]; !ok {
					res.PerTarget[t] = nil
				}
			}
		}
		out = append(out, res)
	}

	// tabel ringkas ke stderr
	fmt.Fprintln(os.Stderr, "== engine-raw FTS5 (MATCH mentah, ORDER BY rank) ==")
	for i, r := range out {
		status := "OK"
		if !r.OK {
			status = "ERR: " + r.Error
		}
		ranks := []string{}
		for t, rk := range r.PerTarget {
			if rk == nil {
				ranks = append(ranks, fmt.Sprintf("%s=NOT", short(t)))
			} else {
				ranks = append(ranks, fmt.Sprintf("%s=%d", short(t), *rk))
			}
		}
		fmt.Fprintf(os.Stderr, "#%-2d n=%-3d %-45s | %s | %s\n", i+1, r.NResults, trunc(r.Query, 45), status, strings.Join(ranks, " "))
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(out)
}

func short(id string) string {
	if len(id) > 8 {
		return "…" + id[len(id)-8:]
	}
	return id
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}

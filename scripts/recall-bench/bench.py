#!/usr/bin/env python3
"""recall-bench — harness benchmark recall Nyawa.

Menjalankan sekumpulan kueri (queries.tsv) terhadap sebuah binary `nyawa recall`,
mem-parsing keluaran recall, lalu menghitung recall@1/@3/@5 dan MRR.

Sebagai pembanding leksikal, harness ini juga menjalankan pencarian BM25 FTS5
langsung ke DB (read-only) dalam tiga mode:
  - raw : persis seperti yang dilakukan mesin (store.FTS5Search mengoper query
          mentah ke `memories_fts MATCH ?` ORDER BY rank). Bisa gagal kalau query
          memuat sintaks FTS5 (mis. tanda hubung).
  - or  : token query digabung dengan OR  (pencarian leksikal paling longgar)
  - and : token query digabung dengan AND (pencarian leksikal paling ketat)

Tidak pernah menulis ke DB: koneksi dibuka dengan mode=ro.

Contoh:
  python3 bench.py
  python3 bench.py --binary /path/ke/nyawa --label kandidat
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BINARY = "/opt/data/.nyawa/nyawa"
DEFAULT_DB = "/opt/data/.nyawa/memory.db"
RESULT_MARKER = re.compile(r"^#(\d+) \[([-0-9.]+)\] (.*)$")
TOKEN_RE = re.compile(r"[0-9A-Za-z_]+")


# --------------------------------------------------------------------------- #
# DB helpers (read-only)
# --------------------------------------------------------------------------- #
def open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    # sebagian memori memuat byte non-UTF8; decode konsisten seperti parser output
    con.text_factory = lambda b: b.decode("utf-8", "replace")
    return con


def load_content_index(con: sqlite3.Connection) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """content.strip() -> [id...]  dan  id -> metadata."""
    by_content: dict[str, list[str]] = {}
    meta: dict[str, dict] = {}
    for mid, content, mtype, ns, length, sup in con.execute(
        "SELECT id, content, mem_type, namespace, length(content), superseded_at FROM memories"
    ):
        key = (content or "").strip()
        by_content.setdefault(key, []).append(mid)
        meta[mid] = {
            "id": mid,
            "mem_type": mtype,
            "namespace": ns,
            "length": length,
            "superseded": sup is not None,
        }
    return by_content, meta


# --------------------------------------------------------------------------- #
# recall output parsing
# --------------------------------------------------------------------------- #
def parse_recall(stdout: str) -> list[dict]:
    """Parse keluaran `nyawa recall`: blok '#N [score] content' (content multi-baris)."""
    results: list[dict] = []
    cur: dict | None = None
    for line in stdout.split("\n"):
        m = RESULT_MARKER.match(line)
        if m:
            if cur is not None:
                results.append(cur)
            cur = {"rank": int(m.group(1)), "score": float(m.group(2)), "content": m.group(3)}
        elif cur is not None:
            cur["content"] += "\n" + line
    if cur is not None:
        results.append(cur)
    for r in results:
        r["content"] = r["content"].rstrip("\n")
    return results


def run_recall(binary: str, db: str, query: str, ns: str | None, timeout: int) -> dict:
    cmd = [binary, "recall", db, query]
    if ns:
        cmd += ["--ns", ns]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout setelah {timeout}s", "cmd": cmd, "results": []}
    stdout = proc.stdout.decode("utf-8", "replace")
    stderr = proc.stderr.decode("utf-8", "replace")
    results = parse_recall(stdout)
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stderr": stderr.strip(),
        "results": results,
        "cmd": cmd,
    }


# --------------------------------------------------------------------------- #
# BM25 FTS5 (langsung ke DB, read-only)
# --------------------------------------------------------------------------- #
FTS_SQL = (
    "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid "
    "WHERE memories_fts MATCH ? AND m.superseded_at IS NULL "
    "ORDER BY rank LIMIT ?"
)


def fts_search(con: sqlite3.Connection, q: str, mode: str, limit: int) -> dict:
    if mode == "raw":
        expr = q
    else:
        tokens = TOKEN_RE.findall(q.lower())
        if not tokens:
            return {"ok": True, "mode": mode, "expr": "", "ids": []}
        op = " OR " if mode == "or" else " AND "
        expr = op.join(f'"{t}"' for t in tokens)
    try:
        ids = [row[0] for row in con.execute(FTS_SQL, (expr, limit))]
        return {"ok": True, "mode": mode, "expr": expr, "ids": ids}
    except Exception as exc:  # FTS5 membocorkan OperationalError untuk sintaks tak valid
        return {"ok": False, "mode": mode, "expr": expr, "error": f"{type(exc).__name__}: {exc}", "ids": []}


# --------------------------------------------------------------------------- #
# queries.tsv
# --------------------------------------------------------------------------- #
def load_queries(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"{path}:{lineno}: butuh minimal 2 kolom (query<TAB>target_ids)")
            targets = [t.strip() for t in parts[1].split(",") if t.strip()]
            ns = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
            rows.append({"lineno": lineno, "query": parts[0], "targets": targets, "ns": ns})
    return rows


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def first_rank_for_targets(order: list[str], targets: list[str]) -> int | None:
    """Peringkat target teratas (1-based) yang muncul di `order`, atau None."""
    for i, mid in enumerate(order, 1):
        if mid in targets:
            return i
    return None


def evaluate(rows: list[dict], con: sqlite3.Connection, binary: str, timeout: int,
             fts_limit: int, ns: str | None, engine_map: dict | None = None) -> dict:
    by_content, meta = load_content_index(con)
    engine_map = engine_map or {}

    per_query: list[dict] = []
    for row in rows:
        q = row["query"]
        targets = row["targets"]
        eff_ns = row["ns"] or ns

        # ----- jalankan recall -----
        rr = run_recall(binary, con_dbpath, q, eff_ns, timeout)
        order: list[str] = []
        ranked_results: list[dict] = []
        matched_by_content: dict[str, list[str]] = {}
        for r in rr["results"]:
            ids = by_content.get(r["content"].strip(), [])
            ranked_results.append({"rank": r["rank"], "score": r["score"], "ids": ids,
                                   "preview": r["content"][:120].replace("\n", " ")})
            for mid in ids:
                if mid not in matched_by_content:
                    matched_by_content[mid] = []
                matched_by_content[mid].append(r["rank"])
            # id pertama untuk urutan (cukup untuk pencocokan target)
            if ids:
                order.append(ids[0])
            else:
                order.append(f"<tak-terpetakan:rank{r['rank']}>")

        recall_ranks = {t: (matched_by_content[t][0] if t in matched_by_content else None) for t in targets}
        best_recall_rank = first_rank_for_targets(order, targets)

        # ----- BM25 FTS -----
        fts = {}
        for mode in ("raw", "or", "and"):
            res = fts_search(con, q, mode, fts_limit)
            rank = first_rank_for_targets(res["ids"], targets)
            per_target = {t: (res["ids"].index(t) + 1 if t in res["ids"] else None) for t in targets}
            fts[mode] = {**res, "best_rank": rank, "per_target": per_target}

        per_query.append({
            "query": q,
            "ns": eff_ns,
            "targets": targets,
            "target_meta": {t: meta.get(t) for t in targets},
            "recall": {
                "ok": rr["ok"],
                "returncode": rr.get("returncode"),
                "error": rr.get("error"),
                "stderr": rr.get("stderr", ""),
                "n_results": len(rr["results"]),
                "per_target_rank": recall_ranks,
                "best_rank": best_recall_rank,
                "top_order": ranked_results,
            },
            "fts": fts,
            "fts_engine_raw": _engine_raw(engine_map.get(q), targets),
        })
    return {"per_query": per_query}


def _engine_raw(er: dict | None, targets: list[str]) -> dict | None:
    """Normalisasi satu baris keluaran ftsprobe (leg FTS mentah mesin)."""
    if not er:
        return None
    if not er.get("ok", True):
        return {"ok": False, "error": er.get("error", ""), "n_results": 0,
                "per_target": {t: None for t in targets}, "best_rank": None}
    per = er.get("per_target_rank", {}) or {}
    ranks = [per.get(t) for t in targets]
    best = min([r for r in ranks if r], default=None)
    return {"ok": True, "error": "", "n_results": er.get("n_results", 0),
            "per_target": {t: per.get(t) for t in targets}, "best_rank": best}


def aggregate(per_query: list[dict], rank_key) -> dict:
    n = len(per_query)
    hits = {1: 0, 3: 0, 5: 0}
    rr_sum = 0.0
    found = 0
    for pq in per_query:
        r = rank_key(pq)
        if r is None:
            continue
        found += 1
        rr_sum += 1.0 / r
        for k in hits:
            if r <= k:
                hits[k] += 1
    return {
        "n_queries": n,
        "found": found,
        "recall@1": hits[1] / n if n else 0.0,
        "recall@3": hits[3] / n if n else 0.0,
        "recall@5": hits[5] / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def fmt_rank(r: int | None) -> str:
    return str(r) if r else "TIDAK KETEMU"


def write_markdown(path: str, doc: dict) -> None:
    meta = doc["meta"]
    agg = doc["aggregate"]
    lines: list[str] = []
    a = lines.append
    a(f"# BASELINE recall Nyawa — {meta['date']}")
    a("")
    a(f"- Binary: `{meta['binary']}` (sha256 `{meta['binary_sha256'][:16]}…`, `{meta['binary_version']}`)")
    a(f"- DB: `{meta['db']}` (mtime `{meta['db_mtime']}`, `{meta['db_total_memories']}` memori, dibaca `mode=ro`)")
    a(f"- Dijalankan: {meta['generated_at']}")
    a(f"- Label: `{meta['label']}`")
    a("")
    a("## Ringkasan")
    a("")
    has_engine = "fts_engine_raw" in agg
    header = "| Metrik | Vector+RRF (recall CLI) | BM25 FTS token-OR | BM25 FTS token-AND |"
    sep = "|---|---|---|---|"
    if has_engine:
        header += " BM25 FTS mesin (raw) |"
        sep += "---|"
    a(header)
    a(sep)
    for k in ("recall@1", "recall@3", "recall@5", "mrr"):
        row = f"| {k} | {agg['recall'][k]:.3f} | {agg['fts_or'][k]:.3f} | {agg['fts_and'][k]:.3f} |"
        if has_engine:
            row += f" {agg['fts_engine_raw'][k]:.3f} |"
        a(row)
    a("")
    a(f"Target yang bisa dijangkau (ada di hasil recall): {agg['found']}/{agg['recall']['n_queries']} kueri menemukan minimal satu target.")
    if has_engine:
        nz = agg.get("fts_engine_raw_nonzero", "?")
        a(f"Leg FTS mentah mesin mengembalikan hasil untuk {nz}/{agg['recall']['n_queries']} kueri; sisanya nol hasil (sinyal leksikal hilang).")
    a("")
    a("## Per kueri")
    a("")
    head = "| # | Kueri | Target | Peringkat recall | Peringkat FTS(raw, python) | FTS(OR) | FTS(AND) |"
    sepc = "|---|---|---|---|---|---|---|"
    if has_engine:
        head += " FTS mesin (raw) |"
        sepc += "---|"
    a(head)
    a(sepc)
    for i, pq in enumerate(doc["per_query"], 1):
        raw = pq["fts"]["raw"]
        raw_cell = "ERROR" if not raw["ok"] else fmt_rank(raw["best_rank"])
        row = (
            f"| {i} | {pq['query']} | {', '.join(pq['targets'])} | {fmt_rank(pq['recall']['best_rank'])} "
            f"| {raw_cell} | {fmt_rank(pq['fts']['or']['best_rank'])} | {fmt_rank(pq['fts']['and']['best_rank'])} |"
        )
        if has_engine:
            er = pq.get("fts_engine_raw")
            if er is None:
                cell = "-"
            elif not er["ok"]:
                cell = "ERROR"
            elif er["n_results"] == 0:
                cell = "0 hasil"
            else:
                cell = fmt_rank(er["best_rank"])
            row += f" {cell} |"
        a(row)
    a("")
    a("## Catatan / temuan")
    a("")
    a(doc.get("notes_markdown", "_(tidak ada)_"))
    a("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
con_dbpath = DEFAULT_DB


def main() -> int:
    global con_dbpath
    ap = argparse.ArgumentParser(description="Harness benchmark recall Nyawa")
    ap.add_argument("--binary", default=DEFAULT_BINARY, help="path binary nyawa (default: runtime)")
    ap.add_argument("--db", default=DEFAULT_DB, help="path DB SQLite (dibaca read-only)")
    ap.add_argument("--queries", default=os.path.join(SCRIPT_DIR, "queries.tsv"))
    ap.add_argument("--out-dir", default=SCRIPT_DIR)
    ap.add_argument("--label", default="runtime", help="label output (mis. runtime / kandidat)")
    ap.add_argument("--ns", default=None, help="namespace default kalau kolom ns di queries.tsv kosong")
    ap.add_argument("--timeout", type=int, default=120, help="timeout per panggilan recall (detik)")
    ap.add_argument("--fts-limit", type=int, default=100, help="top-K untuk pencarian BM25 FTS")
    ap.add_argument("--md-name", default=None, help="nama file markdown (default BASELINE-<tanggal>.md)")
    ap.add_argument("--json-name", default=None, help="nama file JSON")
    ap.add_argument("--notes-file", default=None, help="berkas markdown berisi catatan/temuan tambahan untuk disisipkan")
    ap.add_argument("--engine-fts-json", default=None,
                    help="keluaran ftsprobe (leg FTS mentah mesin, driver SQLite runtime) untuk digabung")
    args = ap.parse_args()
    con_dbpath = args.db

    if not os.path.exists(args.binary):
        print(f"binary tidak ada: {args.binary}", file=sys.stderr)
        return 2

    rows = load_queries(args.queries)
    if not rows:
        print("tidak ada kueri", file=sys.stderr)
        return 2

    con = open_db(args.db)
    # statistik DB
    total = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    db_mtime = datetime.fromtimestamp(os.path.getmtime(args.db), tz=timezone.utc).isoformat()
    try:
        version = subprocess.run([args.binary, "version"], capture_output=True, timeout=30).stdout.decode().strip()
    except Exception:
        version = "?"
    with open(args.binary, "rb") as fh:
        bin_sha = hashlib.sha256(fh.read()).hexdigest()

    print(f"== recall-bench | label={args.label} | binary={args.binary} | {len(rows)} kueri ==")
    engine_map = {}
    if args.engine_fts_json and os.path.exists(args.engine_fts_json):
        with open(args.engine_fts_json, encoding="utf-8") as fh:
            for er in json.load(fh):
                engine_map[er["query"]] = er
        print(f"   + engine-raw FTS dari {args.engine_fts_json} ({len(engine_map)} kueri)")
    ev = evaluate(rows, con, args.binary, args.timeout, args.fts_limit, args.ns, engine_map)
    con.close()

    per_query = ev["per_query"]
    agg = {
        "recall": aggregate(per_query, lambda pq: pq["recall"]["best_rank"]),
        "fts_or": aggregate(per_query, lambda pq: pq["fts"]["or"]["best_rank"]),
        "fts_and": aggregate(per_query, lambda pq: pq["fts"]["and"]["best_rank"]),
        "fts_raw_ok": sum(1 for pq in per_query if pq["fts"]["raw"]["ok"]),
        "found": 0,  # diisi di bawah dari leg recall
    }
    agg["found"] = agg["recall"]["found"]
    if engine_map:
        agg["fts_engine_raw"] = aggregate(
            per_query, lambda pq: (pq["fts_engine_raw"] or {}).get("best_rank"))
        agg["fts_engine_raw_nonzero"] = sum(
            1 for pq in per_query if (pq["fts_engine_raw"] or {}).get("n_results", 0) > 0)

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    label_suffix = "" if args.label == "runtime" else f"-{args.label}"
    md_name = args.md_name or f"BASELINE-{date}{label_suffix}.md"
    json_name = args.json_name or f"baseline-{date}{label_suffix}.json"

    notes = []
    unreachable = [t for pq in per_query for t in pq["targets"]
                   if (pq["target_meta"].get(t) or {}).get("superseded")]
    if unreachable:
        notes.append("- Target superseded (tidak mungkin terjangkau oleh recall aktif): "
                     + ", ".join(sorted(set(unreachable))))
    raw_err = [i + 1 for i, pq in enumerate(per_query) if not pq["fts"]["raw"]["ok"]]
    if raw_err:
        notes.append(f"- Leg FTS mentah (dinilai via SQLite Python) error pada kueri: {raw_err}")
    if "fts_engine_raw" in agg:
        zero = [i + 1 for i, pq in enumerate(per_query)
                if (pq.get("fts_engine_raw") or {}).get("n_results", 0) == 0]
        if zero:
            notes.append(f"- Leg FTS mentah **mesin** (driver SQLite runtime) mengembalikan NOL baris untuk kueri: {zero}")
    if agents := [i + 1 for i, pq in enumerate(per_query) if pq["recall"]["n_results"] == 0]:
        notes.append(f"- recall mengembalikan NOL hasil untuk kueri: {agents}")
    notes_md = "\n".join(notes) if notes else "_(tidak ada)_"
    if args.notes_file and os.path.exists(args.notes_file):
        with open(args.notes_file, encoding="utf-8") as fh:
            extra = fh.read().strip()
        if extra:
            notes_md = extra + ("\n\n" + notes_md if notes else "")

    doc = {
        "meta": {
            "date": date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "binary": os.path.abspath(args.binary),
            "binary_sha256": bin_sha,
            "binary_version": version,
            "db": os.path.abspath(args.db),
            "db_mtime": db_mtime,
            "db_total_memories": total,
            "queries_file": os.path.abspath(args.queries),
            "engine": "vector(HNSW/minilm-multilingual-384) + FTS5(porter unicode61) fused via RRF(60)",
        },
        "aggregate": agg,
        "per_query": per_query,
        "notes_markdown": notes_md,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, json_name)
    md_path = os.path.join(args.out_dir, md_name)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    write_markdown(md_path, doc)

    print("\n-- ringkasan --")
    print(f"recall  : {agg['recall']}")
    print(f"fts(or) : {agg['fts_or']}")
    print(f"fts(and): {agg['fts_and']}")
    print(f"\ntulis: {md_path}\ntulis: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

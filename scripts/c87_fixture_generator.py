#!/usr/bin/env python3
"""C8.7 corrected calibration fixture generator — 80 records, 120 cases, 60/60 split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "c8.7.corrected-calibration.v1"
SPLITS = ("calibration", "holdout")


def build_dataset() -> dict[str, Any]:
    records: list[dict] = []
    cases: list[dict] = []
    ns = "c87-synthetic-main"
    next_id = 0

    def a() -> str:
        nonlocal next_id; next_id += 1; return f"c87a-{next_id:04d}"

    def rec(split: str, ns_: str, kind: str, txt: str, *,
            status: str = "active", supersedes: list[str] | None = None) -> str:
        fid = a()
        records.append({"fixture_id": fid, "namespace": ns_, "type": kind, "split": split,
                        "status": status, "supersedes": supersedes or [], "content": txt})
        return fid

    def add(split: str, cat: str, q: str, ans: bool, *,
            exp: list[str] | None = None, forb: list[str] | None = None, ns_: str | None = None) -> None:
        n = sum(1 for c in cases if c["category"] == cat and c["split"] == split) + 1
        sc = "cal" if split == "calibration" else "hold"
        cases.append({"case_id": f"c87c-{sc}-{cat}-{n:02d}", "category": cat,
                       "split": split, "answerable": ans, "namespace": ns_ or ns,
                       "query": q, "expected_fixture_ids": exp or [],
                       "forbidden_fixture_ids": forb or []})

    for split in SPLITS:
        ic = split == "calibration"
        lib_a = ["silver", "amber", "quiet", "cobalt", "orchid", "maple", "violet", "copper", "golden", "lunar"]
        lib_b = ["lease", "backup", "approval", "retention", "scale", "queue", "retry", "cache", "handoff", "window"]

        # exact_identifier (artifact) — 5 records, 10 cases (2 per record)
        for i in range(5):
            ident = f"SYNTH-C87-ID-{i+1+(0 if ic else 100):04d}"
            aid = rec(split, ns, "artifact",
                      f"Exact identifier {ident} found at path /synthetic/c87/item-{i+1}.md.")
            add(split, "exact_identifier", f"Find synthetic record for {ident}.", True, exp=[aid])
            add(split, "exact_identifier", f"Lookup the path for identifier {ident}.", True, exp=[aid])

        # active_decision (decision) — 5 records, 10 cases
        for i in range(5):
            ent = lib_a[i if ic else i+5]
            aid = rec(split, ns, "decision",
                      f"Decision: {ent} cache uses policy ID-{hash(ent+split)%100}.")
            add(split, "active_decision", f"What policy applies to {ent} cache?", True, exp=[aid])
            add(split, "active_decision", f"Which rule governs the {ent} cache?", True, exp=[aid])

        # hard_constraint (constraint -> procedure) — 5 records, 10 cases
        for i in range(5):
            topic = lib_b[i if ic else i+5]
            aid = rec(split, ns, "constraint",
                      f"Constraint: {topic} refresh window is {hash(topic+split)%48+6}h.")
            add(split, "hard_constraint", f"What is the {topic} refresh window?", True, exp=[aid])
            add(split, "hard_constraint", f"How often does {topic} refresh?", True, exp=[aid])

        # supersession — 5 old + 5 current = 10 records, 10 cases
        for i in range(5):
            topic = lib_b[i if ic else i+5]
            oa = rec(split, ns, "decision",
                     f"Old {topic} policy: {hash(topic+'old'+split)%50+10}h.", status="superseded")
            ca = rec(split, ns, "decision",
                     f"Updated {topic} policy: {hash(topic+'cur'+split)%24+4}h.",
                     status="active", supersedes=[oa])
            add(split, "supersession", f"What is the current {topic} window? (updated)", True, exp=[ca], forb=[oa])
            add(split, "supersession", f"Latest rule for {topic} window?", True, exp=[ca], forb=[oa])

        # namespace_isolation (fact) — 5 records, 10 cases
        for i in range(5):
            ns2 = f"c87-scope-{hash(split)*(i+2)%100:02d}"
            aid = rec(split, ns2, "fact",
                      f"Scope {i+1} label: entity-{hash(split+str(i))%100}.")
            add(split, "namespace_isolation", f"What is scope {i+1} label?", True, exp=[aid], ns_=ns2)
            add(split, "namespace_isolation", f"Describe scope {i+1} label.", True, exp=[aid], ns_=ns2)

        # negative — 10 per split
        for q in [
            "Find identifier SYNTH-ID-9999.", "What policy for undefined topic?",
            "How many nonexistent queue workers?", "Name the invented glacier tuning.",
            "What label for scope 99 chat C?", "Fictional satellite routing key?",
            "Imaginary rainforest moon phase?", "Nonexistent routing value?",
            "Fictional lunar checksum?", "Find record for imaginary topic Z.",
        ] if ic else [
            "Find identifier SYNTH-ID-8888.", "What policy for missing topic?",
            "How many phantom maintenance workers?", "Name desert engine transmission key.",
            "What label for scope 98 chat D?", "Canyon beacon code?",
            "Which ocean choir paper key?", "Nonexistent handoff rule?",
            "Glacier ferry magnetic code?", "Find record for missing entity Q.",
        ]:
            add(split, "negative_confusable", q, False)

        # decoys — 10 per split
        for i in range(10):
            rec(split, ns, "fact", f"Decoy {i+1}: fictional unrelated note for {split}.")

    # Validate
    rs, cs = records, cases
    assert len(rs) == 80, f"records={len(rs)}"
    assert len(cs) == 120, f"cases={len(cs)}"
    cal_c = [c for c in cs if c["split"] == "calibration"]
    hold_c = [c for c in cs if c["split"] == "holdout"]
    assert len(cal_c) == 60 and len(hold_c) == 60
    assert sum(1 for c in cal_c if c["answerable"]) == 50
    assert sum(1 for c in hold_c if c["answerable"]) == 50
    cal_fids = {r["fixture_id"] for r in rs if r["split"] == "calibration"}
    hold_fids = {r["fixture_id"] for r in rs if r["split"] == "holdout"}
    assert cal_fids.isdisjoint(hold_fids), "split leak"
    assert all(r["type"] in ("artifact", "constraint", "decision", "fact") for r in rs), "bad type"
    assert all(r["status"] in ("active", "superseded") for r in rs), "bad status"
    all_fids = cal_fids | hold_fids
    for c in cs:
        for fid in c["expected_fixture_ids"] + c["forbidden_fixture_ids"]:
            assert fid in all_fids, f"missing {fid} in {c['case_id']}"

    return {"schema_version": SCHEMA,
            "description": "C8.7 corrected calibration corpus with types, status, supersedes.",
            "records": rs, "cases": cs, "k": 5,
            "gates": {"positive_recall_at_k_min": 0.9, "exact_identifier_recall_min": 1.0,
                       "negative_false_positive_rate_max": 0.0, "namespace_leaks_max": 0}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ds = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ds, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(args.output), "records": len(ds["records"]),
                      "cases": len(ds["cases"]), "schema": ds["schema_version"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

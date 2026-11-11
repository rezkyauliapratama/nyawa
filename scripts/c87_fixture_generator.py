#!/usr/bin/env python3
"""Generate the deterministic C8.7 calibration-only synthetic fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "c8.7.corrected-calibration.v1"


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    namespace = "c87-synthetic-main"
    next_id = 0

    def record(ns: str, kind: str, content: str, *, status: str = "active",
               supersedes: list[str] | None = None) -> str:
        nonlocal next_id
        next_id += 1
        fixture_id = f"c87a-{next_id:04d}"
        records.append({
            "fixture_id": fixture_id, "namespace": ns, "type": kind,
            "split": "calibration", "status": status,
            "supersedes": list(supersedes or []), "content": content,
        })
        return fixture_id

    def case(category: str, query: str, *, answerable: bool = True,
             expected: list[str] | None = None, forbidden: list[str] | None = None,
             ns: str = namespace) -> None:
        ordinal = 1 + sum(c["category"] == category for c in cases)
        cases.append({
            "case_id": f"c87c-cal-{category}-{ordinal:02d}",
            "category": category, "split": "calibration", "answerable": answerable,
            "namespace": ns, "query": query,
            "expected_fixture_ids": list(expected or []),
            "forbidden_fixture_ids": list(forbidden or []),
        })

    exact_ids = [f"SYNTH-C87-ID-{i:04d}" for i in range(1, 6)]
    for ident in exact_ids:
        fid = record(namespace, "artifact", f"Exact identifier {ident} at /synthetic/c87/{ident}.md.")
        case("exact_identifier", f"Find synthetic record for {ident}.", expected=[fid])
        case("exact_identifier", f"Lookup the path for identifier {ident}.", expected=[fid])

    decision_topics = ("silver", "amber", "quiet", "cobalt", "orchid")
    for topic in decision_topics:
        fid = record(namespace, "decision", f"Decision: {topic} cache uses policy POLICY-{topic.upper()}-17.")
        case("active_decision", f"What policy applies to {topic} cache?", expected=[fid])
        case("active_decision", f"Which rule governs the {topic} cache?", expected=[fid])

    constraint_topics = ("lease", "backup", "approval", "retention", "scale")
    for index, topic in enumerate(constraint_topics):
        hours = (index + 2) * 6
        fid = record(namespace, "constraint", f"Constraint: {topic} refresh window is {hours}h.")
        case("hard_constraint", f"What is the {topic} refresh window?", expected=[fid])
        case("hard_constraint", f"How often does {topic} refresh?", expected=[fid])

    for index, topic in enumerate(constraint_topics):
        old_hours, current_hours = 30 + index, 6 + index
        old = record(namespace, "decision", f"Old {topic} policy: {old_hours}h.", status="superseded")
        current = record(namespace, "decision", f"Updated {topic} policy: {current_hours}h.", supersedes=[old])
        case("supersession", f"What is the current {topic} window? (updated)", expected=[current], forbidden=[old])
        case("supersession", f"Latest rule for {topic} window?", expected=[current], forbidden=[old])

    for index in range(5):
        foreign_ns = f"c87-scope-{(index + 1) * 11:02d}"
        fid = record(foreign_ns, "fact", f"Scope {index + 1} label: entity-{(index + 1) * 13:02d}.")
        case("namespace_isolation", f"What is scope {index + 1} label?", expected=[fid], ns=foreign_ns)
        case("namespace_isolation", f"Describe scope {index + 1} label.", expected=[fid], ns=foreign_ns)
        case("namespace_isolation", f"What is scope {index + 1} label?", answerable=False,
             forbidden=[fid], ns=namespace)

    for query in (
        "Find identifier SYNTH-ID-9999.", "What policy for undefined topic?",
        "How many nonexistent queue workers?", "Name the invented glacier tuning.",
        "What label for scope 99 chat C?",
    ):
        case("negative_confusable", query, answerable=False)

    for index in range(10):
        record(namespace, "fact", f"Decoy {index + 1}: fictional unrelated calibration note.")

    assert len(records) == 40, f"calibration records={len(records)}"
    assert len(cases) == 60, f"calibration cases={len(cases)}"
    assert sum(c["answerable"] for c in cases) == 50
    fixture_ids = {r["fixture_id"] for r in records}
    assert all(r["split"] == "calibration" for r in records)
    assert all(c["split"] == "calibration" for c in cases)
    assert all(r["type"] in {"artifact", "constraint", "decision", "fact"} for r in records)
    assert all(r["status"] in {"active", "superseded"} for r in records)
    for item in cases:
        assert set(item["expected_fixture_ids"] + item["forbidden_fixture_ids"]).issubset(fixture_ids)
    return {
        "schema_version": SCHEMA,
        "description": "Deterministic synthetic C8.7 calibration-only corpus.",
        "records": records,
        "cases": cases,
        "k": 5,
        "gates": {
            "positive_recall_at_5_min": 0.90,
            "exact_identifier_recall_at_5_min": 1.0,
            "active_decision_recall_at_5_min": 0.90,
            "hard_constraint_recall_at_5_min": 1.0,
            "supersession_recall_at_5_min": 0.90,
            "namespace_isolation_recall_at_5_min": 0.90,
            "negative_false_positive_rate_at_k_max": 0.0,
            "raw_namespace_leak_count_max": 0,
            "selected_namespace_leak_count_max": 0,
            "selected_forbidden_candidate_exposure_count_max": 0,
            "superseded_selected_candidate_count_max": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixture": str(args.output), "records": len(dataset["records"]),
        "cases": len(dataset["cases"]), "schema": dataset["schema_version"],
        "splits": sorted({r["split"] for r in dataset["records"]}),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

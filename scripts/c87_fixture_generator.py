#!/usr/bin/env python3
"""Generate the legacy deterministic C8.7 calibration-only fixture for unit tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
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



SCHEMA_V2 = "c8.7.corrected-synthetic-development.v2"
CLASSIFICATION_V2 = "synthetic_development_not_independent_not_holdout_not_production_evidence"
V2_NAMESPACES = ("aurora_ops", "cinder_archive", "moss_ledger", "quartz_lab")


def _build_development_dataset_v2_unvalidated() -> dict[str, Any]:
    """Construct canonical v2 fixture data without invoking its validator."""
    records: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    def record(fid: str, ns: str, content: str, kind: str = "fact", status: str | None = "active",
               supersedes: list[str] | None = None) -> str:
        records.append({"fixture_id": fid, "namespace": ns, "type": kind,
                        "split": "synthetic_development", "status": status,
                        "supersedes": list(supersedes or []), "content": content})
        return fid

    def case(group: str, category: str, query: str, lang: str, *, answer: str = "",
             expected: list[str] | None = None, forbidden: list[str] | None = None,
             ns: str | None = None, scoped: bool = True, safety: str = "answer") -> None:
        cases.append({"case_id": f"C87V2-{len(cases) + 1:03d}", "group": group,
                      "category": category, "language": lang, "split": "synthetic_development",
                      "expected_safety": safety, "expected_answer": answer,
                      "expected_fixture_ids": list(expected or []),
                      "forbidden_fixture_ids": list(forbidden or []), "namespace": ns,
                      "scope_mode": "explicit" if scoped else "unspecified", "query": query})

    for i in range(1, 11):
        lang = "id" if i % 2 == 0 else "en"
        ns = V2_NAMESPACES[(i - 1) % len(V2_NAMESPACES)]
        ident = f"ORC-{i * 137 % 997:03d}-VQ"
        fid = record(f"D-ID-{i:02d}", ns, f"Orchid relay item {i} exact identifier {ident}; token {ident}.")
        q = (f"Cari identifier persis {ident} untuk Orchid relay item {i} dalam namespace {ns}." if lang == "id"
             else f"Find exact identifier {ident} for Orchid relay item {i} within namespace {ns}.")
        case("exact_identifier", "exact_identifier", q, lang, answer=ident, expected=[fid], ns=ns)

        path = f"/vault/{ns}/specimens/lichen-{i:02d}/route.map"
        fid = record(f"D-PATH-{i:02d}", ns, f"Canonical route for lichen sample {i}: {path}", "artifact")
        q = (f"Berikan path route map lichen sample {i} di namespace {ns}." if lang == "id"
             else f"Give the canonical route.map path for lichen sample {i} in namespace {ns}.")
        case("exact_path", "hard_constraint", q, lang, answer=path, expected=[fid], ns=ns)

        digest = hashlib.sha256(f"synthetic-corrected-kestrel-package-{i}".encode()).hexdigest()
        fid = record(f"D-HASH-{i:02d}", ns, f"Kestrel package {i} SHA-256 checksum: {digest}", "artifact")
        q = (f"Apa checksum SHA-256 Kestrel package {i} di namespace {ns}?" if lang == "id"
             else f"What is the SHA-256 checksum for Kestrel package {i} in namespace {ns}?")
        case("exact_hash", "hard_constraint", q, lang, answer=digest, expected=[fid], ns=ns)

        platform = chr(65 + i)
        fid = record(f"D-SEM-{i:02d}", ns,
                     f"Zephyr-{i:02d} wind sensor is installed on platform {platform}; anemometer location {platform}.")
        q = (f"Di platform mana anemometer Zephyr-{i:02d} berada dalam namespace {ns}?" if lang == "id"
             else f"Which platform hosts the Zephyr-{i:02d} anemometer in namespace {ns}?")
        case("semantic_paraphrase", "active_decision", q, lang, answer=f"platform {platform}", expected=[fid], ns=ns)

        old = record(f"D-OLD-{i:02d}", ns, f"Tern rig {i} retired inspection interval: {20+i} days.",
                     "decision", "superseded")
        new = record(f"D-NEW-{i:02d}", ns, f"Tern rig {i} current approved inspection interval: {40+i} days.",
                     "decision", supersedes=[old])
        q = (f"Interval inspeksi Tern rig {i} yang berlaku saat ini di namespace {ns}?" if lang == "id"
             else f"What is the currently approved inspection interval for Tern rig {i} in namespace {ns}?")
        case("supersession_current", "supersession", q, lang, answer=f"{40+i} days", expected=[new],
             forbidden=[old], ns=ns)

        fid = record(f"D-ABS-{i:02d}", ns,
                     f"Plover greenhouse {i} grows blue fennel with drip irrigation; pump brand Solace.")
        q = (f"Apa kode kalibrasi pompa Plover greenhouse {i} di namespace {ns}?" if lang == "id"
             else f"What is the calibration code for Plover greenhouse {i} pump in namespace {ns}?")
        case("same_topic_unanswerable", "negative_confusable", q, lang, ns=ns, safety="abstain")

        actual, requested = f"MICA-{i:02d}-R", f"MICA-{i:02d}-P"
        fid = record(f"D-NEAR-{i:02d}", ns,
                     f"Mica valve inventory identifier is {actual}; valve series Mica-{i:02d}.")
        q = (f"Temukan identifier persis {requested} untuk Mica valve dalam namespace {ns}." if lang == "id"
             else f"Find exact identifier {requested} for Mica valve within namespace {ns}.")
        case("near_miss_identifier", "negative_confusable", q, lang, ns=ns, safety="abstain", forbidden=[fid])

        foreign = V2_NAMESPACES[i % len(V2_NAMESPACES)]
        if foreign == ns:
            foreign = V2_NAMESPACES[(i + 1) % len(V2_NAMESPACES)]
        target = record(f"D-XA-{i:02d}", ns, f"Tamarind batch {i} storage bay: {chr(75+i)}-LOCK.")
        decoy = record(f"D-XB-{i:02d}", foreign, f"Tamarind batch {i} storage bay: {chr(75+i)}-DOCK.")
        q = (f"Dalam namespace {ns}, di mana Tamarind batch {i} disimpan?" if lang == "id"
             else f"Within namespace {ns}, where is Tamarind batch {i} stored?")
        case("cross_namespace_conflict", "namespace_isolation", q, lang, answer=f"{chr(75+i)}-LOCK",
             expected=[target], forbidden=[decoy], ns=ns)

        alias = f"EMBER-{i:02d}"
        a_ns, b_ns = V2_NAMESPACES[(i + 1) % 4], V2_NAMESPACES[(i + 2) % 4]
        a = record(f"D-AA-{i:02d}", a_ns, f"Alias {alias} maps to basin NORTH-{i}.")
        b = record(f"D-AB-{i:02d}", b_ns, f"Alias {alias} maps to basin SOUTH-{i}.")
        q = (f"Alias {alias} mengarah ke basin mana?" if lang == "id"
             else f"Which basin does alias {alias} point to?")
        case("namespace_ambiguity", "negative_confusable", q, lang, safety="abstain",
             forbidden=[a, b], ns=None, scoped=False)

        fid = record(f"D-MISSING-{i:02d}", ns, f"Marten greenhouse {i} control dial setting: DIAL-{i:02d}.", status=None)
        q = (f"Berapa posisi dial kontrol Marten greenhouse {i} di namespace {ns}?" if lang == "id"
             else f"What is the Marten greenhouse {i} control dial setting in namespace {ns}?")
        case("missing_state", "negative_confusable", q, lang, safety="abstain", forbidden=[fid], ns=ns)

        a = record(f"D-CA-{i:02d}", ns, f"Juniper rule {i} authoritative limit: {60+i} units.", "decision")
        b = record(f"D-CB-{i:02d}", ns, f"Juniper rule {i} authoritative limit: {80+i} units.", "decision")
        q = (f"Berapa batas resmi Juniper rule {i} di namespace {ns}?" if lang == "id"
             else f"What is the authoritative limit for Juniper rule {i} in namespace {ns}?")
        case("conflicting_state", "negative_confusable", q, lang, safety="abstain", forbidden=[a, b], ns=ns)

        fid = record(f"D-BI-{i:02d}", ns,
                     f"Osprey marker {i} is at the western gate; penanda Osprey berada di gerbang barat.")
        q = (f"Di mana penanda Osprey {i} berada dalam namespace {ns}?" if lang == "id"
             else f"Where is Osprey marker {i} located in namespace {ns}?")
        case("bilingual_semantic_transfer", "active_decision", q, lang,
             answer="gerbang barat" if lang == "id" else "western gate", expected=[fid], ns=ns)

    result = {"schema_version": SCHEMA_V2, "classification": CLASSIFICATION_V2,
              "sealed_holdout": False, "production_evidence": False,
              "description": "Corrected synthetic development v2; prior inspected lineage retained, not independent or holdout.",
              "corrections": {"underspecified_identifier_items": 10, "underspecified_hash_items": 10,
                              "item_number_explicit": True, "explicit_scope": True,
                              "supersession_lineage_preserved": True},
              "records": records, "cases": cases}
    result["summary"] = {"case_count": len(cases), "record_count": len(records),
                          "groups": dict(sorted(Counter(x["group"] for x in cases).items())),
                          "languages": dict(sorted(Counter(x["language"] for x in cases).items())),
                          "expected_safety": dict(sorted(Counter(x["expected_safety"] for x in cases).items()))}
    return result


def build_development_dataset_v2() -> dict[str, Any]:
    """Build and validate the deterministic synthetic-development v2 corpus."""
    dataset = _build_development_dataset_v2_unvalidated()
    validate_development_dataset_v2(dataset)
    return dataset


def validate_development_dataset_v2(dataset: dict[str, Any]) -> None:
    """Validate structure plus group-specific evidence/query contracts before runtime.

    The canonical comparison prevents silent label/query/evidence drift in this fixed
    benchmark. Explicit group rules independently enforce answer shape and uniqueness;
    neither establishes semantic truth, which still requires human fixture review.
    """
    if dataset.get("schema_version") != SCHEMA_V2 or dataset.get("classification") != CLASSIFICATION_V2:
        raise ValueError("fixture must be explicitly classified synthetic development v2")
    if dataset.get("sealed_holdout") is not False or dataset.get("production_evidence") is not False:
        raise ValueError("holdout and production evidence are forbidden")
    records, cases = dataset.get("records"), dataset.get("cases")
    if not isinstance(records, list) or not isinstance(cases, list) or (len(records), len(cases)) != (160, 120):
        raise ValueError("unexpected development v2 fixture counts")
    if any(not isinstance(row, dict) for row in records + cases):
        raise ValueError("fixture rows must be objects")
    by_id = {row.get("fixture_id"): row for row in records}
    if None in by_id or len(by_id) != len(records) or len({x.get("case_id") for x in cases}) != len(cases):
        raise ValueError("fixture identifiers must be unique")
    if any(x.get("split") != "synthetic_development" for x in records + cases):
        raise ValueError("non-development split is prohibited")
    for row in records:
        if row.get("namespace") not in V2_NAMESPACES or row.get("status") not in {"active", "superseded", None}:
            raise ValueError("invalid record namespace/status")
        if not isinstance(row.get("content"), str) or not row["content"].strip():
            raise ValueError("record content must be nonempty text")
        for prior in row.get("supersedes", []):
            old = by_id.get(prior)
            if old is None or old.get("status") != "superseded" or old.get("namespace") != row["namespace"]:
                raise ValueError("lineage must point to a same-namespace superseded record")

    group_contract = {
        "exact_identifier": ("answer", "exact_identifier"), "exact_path": ("answer", "hard_constraint"),
        "exact_hash": ("answer", "hard_constraint"), "semantic_paraphrase": ("answer", "active_decision"),
        "supersession_current": ("answer", "supersession"), "same_topic_unanswerable": ("abstain", "negative_confusable"),
        "near_miss_identifier": ("abstain", "negative_confusable"),
        "cross_namespace_conflict": ("answer", "namespace_isolation"),
        "namespace_ambiguity": ("abstain", "negative_confusable"), "missing_state": ("abstain", "negative_confusable"),
        "conflicting_state": ("abstain", "negative_confusable"),
        "bilingual_semantic_transfer": ("answer", "active_decision"),
    }
    seen_groups: Counter[str] = Counter()
    for row in cases:
        if row.get("language") not in {"en", "id"} or row.get("expected_safety") not in {"answer", "abstain"}:
            raise ValueError("invalid case language or answerability")
        group = row.get("group")
        if group not in group_contract:
            raise ValueError(f"unknown development-v2 case group: {group!r}")
        expected_safety, category = group_contract[group]
        if row["expected_safety"] != expected_safety or row.get("category") != category:
            raise ValueError(f"impossible safety/category label for group {group}: {row.get('case_id')}")
        seen_groups[group] += 1
        expected, forbidden = row.get("expected_fixture_ids", []), row.get("forbidden_fixture_ids", [])
        if not isinstance(expected, list) or not isinstance(forbidden, list):
            raise ValueError("evidence labels must be lists")
        if len(expected) != len(set(expected)) or len(forbidden) != len(set(forbidden)):
            raise ValueError("duplicate evidence identifiers in labels")
        if not set(expected + forbidden).issubset(by_id):
            raise ValueError("case label references unknown evidence")
        if not isinstance(row.get("query"), str) or not row["query"].strip():
            raise ValueError("case query must be nonempty text")
        if row["expected_safety"] == "answer":
            answer = row.get("expected_answer")
            if not isinstance(answer, str) or len(answer.strip()) < 5 or len(expected) != 1:
                raise ValueError(f"weak answer or non-unique gold source: {row['case_id']}")
            evidence = by_id[expected[0]]
            if answer.casefold() not in evidence["content"].casefold():
                raise ValueError(f"answer not supported by evidence: {row['case_id']}")
            if row.get("namespace") != evidence["namespace"]:
                raise ValueError("answer evidence must match explicit request scope")
            if evidence.get("status") != "active":
                raise ValueError("answer evidence must be active")
            if group == "exact_identifier" and not re.fullmatch(r"ORC-\d{3}-VQ", answer):
                raise ValueError(f"weak exact identifier answer: {row['case_id']}")
            if group == "exact_hash" and not re.fullmatch(r"[0-9a-f]{64}", answer):
                raise ValueError(f"weak checksum answer: {row['case_id']}")
            if group == "exact_path" and not re.fullmatch(r"/vault/[a-z_]+/specimens/lichen-\d{2}/route\.map", answer):
                raise ValueError(f"weak exact path answer: {row['case_id']}")
            if group == "semantic_paraphrase" and not re.fullmatch(r"platform [B-K]", answer):
                raise ValueError(f"weak semantic answer: {row['case_id']}")
            if group == "supersession_current" and not re.fullmatch(r"\d+ days", answer):
                raise ValueError(f"weak supersession answer: {row['case_id']}")
            if group == "cross_namespace_conflict" and not re.fullmatch(r"[L-U]-LOCK", answer):
                raise ValueError(f"weak scoped answer: {row['case_id']}")
            anchor_patterns = {
                "exact_identifier": r"Orchid relay item \d+", "exact_path": r"lichen sample \d+",
                "exact_hash": r"Kestrel package \d+", "semantic_paraphrase": r"Zephyr-\d{2}",
                "supersession_current": r"Tern rig \d+", "cross_namespace_conflict": r"Tamarind batch \d+",
                "bilingual_semantic_transfer": r"Osprey",
            }
            anchor_match = re.search(anchor_patterns[group], row["query"], re.I) if group in anchor_patterns else None
            anchor = anchor_match.group(0).casefold() if anchor_match else ""
            if not anchor:
                raise ValueError(f"query lacks group-specific evidence anchor: {row['case_id']}")
            if group == "bilingual_semantic_transfer":
                subject_number = re.search(r"Osprey(?: marker)?\s+(\d+)", row["query"], re.I)
                anchor = (rf"Osprey marker {subject_number.group(1)}\b" if subject_number else "")
            relevant = [candidate for candidate in records
                        if candidate.get("namespace") == row.get("namespace") and candidate.get("status") == "active"
                        and answer.casefold() in candidate["content"].casefold()
                        and anchor and (re.search(anchor, candidate["content"], re.I)
                                        if group == "bilingual_semantic_transfer"
                                        else anchor in candidate["content"].casefold())]
            if len(relevant) != 1 or relevant[0]["fixture_id"] != expected[0]:
                raise ValueError(f"ambiguous duplicate active evidence for {row['case_id']}")
        elif row.get("expected_answer") or expected:
            raise ValueError("abstention labels cannot contain answer/gold source")
        if row.get("scope_mode") == "explicit":
            if not isinstance(row.get("namespace"), str) or row["namespace"] not in row["query"]:
                raise ValueError("explicit request scope must be stated in query")
        elif row.get("scope_mode") != "unspecified" or row.get("namespace") is not None:
            raise ValueError("unspecified-scope test must not receive a namespace oracle")
        if group == "bilingual_semantic_transfer":
            query = row["query"].casefold()
            language_matches = ((row["language"] == "id" and "di mana" in query and "where is" not in query)
                                or (row["language"] == "en" and "where is" in query and "di mana" not in query))
            expected_answer = "gerbang barat" if row["language"] == "id" else "western gate"
            if not language_matches or row.get("expected_answer") != expected_answer:
                raise ValueError(f"bilingual query/label language mismatch: {row['case_id']}")
        if group == "conflicting_state":
            sources = [by_id[fid] for fid in forbidden]
            if (len(sources) != 2 or any(source.get("status") != "active" or source.get("namespace") != row.get("namespace")
                                         for source in sources)
                    or sources[0]["content"] == sources[1]["content"]):
                raise ValueError(f"conflicting-state negative lacks two distinct active sources: {row['case_id']}")

    expected_groups = {name: 10 for name in group_contract}
    if seen_groups != Counter(expected_groups):
        raise ValueError("development-v2 group distribution mismatch")
    summary = {"case_count": len(cases), "record_count": len(records),
               "groups": dict(sorted(Counter(x["group"] for x in cases).items())),
               "languages": dict(sorted(Counter(x["language"] for x in cases).items())),
               "expected_safety": dict(sorted(Counter(x["expected_safety"] for x in cases).items()))}
    if dataset.get("summary") != summary:
        raise ValueError("fixture summary mismatch")

    # Fixed synthetic benchmark contract: detect query swaps, weakened labels, evidence
    # substitutions, and duplicate active records even when counts remain unchanged.
    canonical = _build_development_dataset_v2_unvalidated()
    if records != canonical["records"]:
        raise ValueError("record set differs from deterministic group evidence contract")
    if cases != canonical["cases"]:
        raise ValueError("query/language/gold labels differ from deterministic group contract")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=("legacy", "development-v2"), default="legacy")
    args = parser.parse_args()
    dataset = build_development_dataset_v2() if args.version == "development-v2" else build_dataset()
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

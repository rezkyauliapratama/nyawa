#!/usr/bin/env python3
"""Generate a deterministic, synthetic-only C8.1 calibration/holdout corpus."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "c8.1.abstention-calibration.v1"
SPLITS = ("calibration", "holdout")
SECRET_SHAPED = re.compile(r"[A-Za-z0-9/+=]{40}")
POSITIVE_CATEGORIES = (
    "exact_identifier",
    "semantic_paraphrase",
    "bilingual_paraphrase",
    "namespace_isolation",
    "active_vs_retired",
)
NEGATIVE_SUBTYPES = (
    "hard_semantic",
    "hard_lexical",
    "near_identifier",
    "scope_confusion",
    "out_of_domain",
)


def _record(records: list[dict[str, str]], fixture_id: str, namespace: str, kind: str, marker: str, text: str) -> str:
    records.append({
        "fixture_id": fixture_id,
        "namespace": namespace,
        "type": kind,
        "content": f"Synthetic C8.1 fixture. Marker {marker}. {text}",
    })
    return marker


def _case(
    cases: list[dict[str, Any]],
    case_id: str,
    category: str,
    split: str,
    namespace: str,
    query: str,
    relevant: list[str],
    forbidden: list[str] | None = None,
    exclude_types: list[str] | None = None,
    subtype: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "case_id": case_id,
        "category": category,
        "split": split,
        "answerable": bool(relevant),
        "namespace": namespace,
        "query": query,
        "relevant_markers": relevant,
        "forbidden_markers": forbidden or [],
        "exclude_types": exclude_types or [],
        "min_score": 0.0,
    }
    if subtype:
        row["subtype"] = subtype
    cases.append(row)


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []
    main_ns = "c8-synthetic-main"

    # Exact identifiers: 20 distinct targets, split evenly to keep a useful
    # exact-ID holdout while avoiding any long secret-shaped token.
    for n in range(1, 21):
        split = "calibration" if n <= 10 else "holdout"
        token = f"SYNTH-C81-ID-{n:04d}"
        marker = _record(
            records, f"C81_EXACT_{n:02d}", main_ns, "fact", f"C81_EXACT_MARKER_{n:02d}",
            f"Fictional exact identifier {token} maps to /synthetic/c81/item-{n:02d}.md.",
        )
        _case(
            cases, f"Q81_EXACT_{n:02d}", "exact_identifier", split, main_ns,
            f"Find the synthetic record for exact identifier {token}.", [marker],
        )

    # Semantic paraphrases: separate calibration and holdout topics.
    for n in range(1, 11):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(
            records, f"C81_SEM_{n:02d}", main_ns, "decision", f"C81_SEM_MARKER_{n:02d}",
            f"Fictional decision: the {['silver', 'amber', 'quiet', 'cobalt', 'orchid', 'maple', 'violet', 'copper', 'golden', 'lunar'][n-1]} demo cache uses policy {n:02d}.",
        )
        query_terms = [
            "Which policy applies to the silver demo cache?",
            "What rule did the team select for the amber cache?",
            "How is the quiet demo cache configured?",
            "What is the fictional policy for the cobalt cache?",
            "Which rule governs the orchid cache in the demo?",
            "What choice was made for the maple cache?",
            "Which setting does the violet cache follow?",
            "What rule is used by the copper demo cache?",
            "How is the golden cache policy described?",
            "What policy was chosen for the lunar demo cache?",
        ]
        _case(cases, f"Q81_SEM_{n:02d}", "semantic_paraphrase", split, main_ns, query_terms[n-1], [marker])

    # Bilingual cases use distinct fictional topics and language directions.
    id_queries = [
        "Berapa kapasitas antrean anggrek untuk demo?",
        "Kebijakan apa yang dipilih untuk cache komet?",
        "Berapa lama jendela pembaruan peta?",
        "Siapa yang menyetujui rute burung layang-layang?",
        "Apa mode uji untuk sensor awan?",
        "How many workers does the fictional coral queue use?",
        "Which policy did the team choose for the copper orchard?",
        "What is the renewal window for the paper kite lease?",
        "Which reviewer owns the fictional amber bridge?",
        "What mode is used for the velvet telescope test?",
    ]
    id_facts = [
        "Indonesian: antrean anggrek memakai empat worker. English: the orchid queue uses four workers.",
        "Indonesian: cache komet mengikuti aturan lentera. English: the comet cache follows the lantern rule.",
        "Indonesian: jendela pembaruan peta berlangsung enam jam. English: the map refresh window lasts six hours.",
        "Indonesian: rute burung layang-layang disetujui satu reviewer. English: one reviewer approves the swallow route.",
        "Indonesian: sensor awan berjalan dalam mode dry-run. English: the cloud sensor runs in dry-run mode.",
        "English: the fictional coral queue uses three workers. Indonesian: antrean koral memakai tiga worker.",
        "English: the copper orchard uses the ribbon policy. Indonesian: kebun tembaga memakai kebijakan pita.",
        "English: the paper kite lease renews every eight hours. Indonesian: sewa layang-layang kertas diperbarui tiap delapan jam.",
        "English: the amber bridge is owned by reviewer two. Indonesian: jembatan amber ditangani reviewer dua.",
        "English: the velvet telescope test uses observe-only mode. Indonesian: uji teleskop beludru memakai mode observasi saja.",
    ]
    for n in range(1, 11):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(records, f"C81_BILINGUAL_{n:02d}", main_ns, "decision", f"C81_BILINGUAL_MARKER_{n:02d}", id_facts[n-1])
        _case(cases, f"Q81_BILINGUAL_{n:02d}", "bilingual_paraphrase", split, main_ns, id_queries[n-1], [marker])

    # Namespace pairs each have distinct values; a query is scoped to one side
    # and the opposite-side marker is explicitly treated as forbidden.
    labels_a = ["silver kite", "violet compass", "green comet", "red acorn", "quiet planet", "white lantern", "soft river", "brass leaf", "small harbor", "frosted bell"]
    labels_b = ["copper heron", "golden sail", "amber wave", "blue feather", "bright river", "marble cloud", "paper crane", "coral arch", "velvet star", "maple wheel"]
    scope_markers: dict[int, list[str]] = {}
    for n, (label_a, label_b) in enumerate(zip(labels_a, labels_b), start=1):
        split = "calibration" if n <= 5 else "holdout"
        ns_a = f"c8-synthetic-scope-{n:02d}-a"
        ns_b = f"c8-synthetic-scope-{n:02d}-b"
        marker_a = _record(records, f"C81_SCOPE_{n:02d}_A", ns_a, "fact", f"C81_SCOPE_{n:02d}_A_MARKER", f"Fictional scope {n:02d} chat A label is {label_a}.")
        marker_b = _record(records, f"C81_SCOPE_{n:02d}_B", ns_b, "fact", f"C81_SCOPE_{n:02d}_B_MARKER", f"Fictional scope {n:02d} chat B label is {label_b}.")
        scope_markers[n] = [marker_a, marker_b]
        _case(cases, f"Q81_SCOPE_{n:02d}", "namespace_isolation", split, ns_a, f"What label belongs to scope {n:02d} chat A?", [marker_a], [marker_b])

    # Active-vs-retired pairs test exclude_types on the same topic.
    active_facts = [
        ("lease", "short renewal window", "long renewal window"),
        ("backup", "two copies", "seven copies"),
        ("approval", "one reviewer", "four reviewers"),
        ("retention", "fourteen days", "ninety days"),
        ("scale", "three workers", "thirty workers"),
        ("queue", "four workers", "twelve workers"),
        ("retry", "two retries", "eight retries"),
        ("cache", "blue cedar tier", "red oak tier"),
        ("handoff", "manual review", "automatic approval"),
        ("window", "six-hour window", "twenty-four-hour window"),
    ]
    for n, (topic, active, retired) in enumerate(active_facts, start=1):
        split = "calibration" if n <= 5 else "holdout"
        marker_active = _record(records, f"C81_ACTIVE_{n:02d}", main_ns, "constraint", f"C81_ACTIVE_MARKER_{n:02d}", f"Current fictional {topic} policy is {active}.")
        marker_retired = _record(records, f"C81_RETIRED_{n:02d}", main_ns, "superseded", f"C81_RETIRED_MARKER_{n:02d}", f"Retired fictional {topic} policy was {retired}.")
        _case(cases, f"Q81_ACTIVE_{n:02d}", "active_vs_retired", split, main_ns, f"What is the current fictional {topic} policy?", [marker_active], [marker_retired], ["superseded"])

    # Semantically/lexically similar distractors deliberately tempt false hits.
    decoys = [
        ("C81_DECOY_01", "The silver demo cache's interface color is painted green; this is not its caching policy."),
        ("C81_DECOY_02", "A fictional lunar transit has an orbit label, not a checksum or memory identifier."),
        ("C81_DECOY_03", "The cobalt sensor is a diagram label; it does not specify the test mode."),
        ("C81_DECOY_04", "Copper orchard is a fictional UI theme, not an access or approval policy."),
        ("C81_DECOY_05", "The blue cedar phrase names a picture palette in this note, not a cache tier."),
        ("C81_DECOY_06", "A paper kite illustration has no lease duration or renewal schedule."),
        ("C81_DECOY_07", "The amber bridge is a decorative icon, not a named reviewer or workflow owner."),
        ("C81_DECOY_08", "A velvet telescope can be a story prop; this note assigns no operating mode."),
        ("C81_DECOY_09", "Fictional queue names may mention coral or orchid without stating worker capacity."),
        ("C81_DECOY_10", "A marble cloud is only a synthetic color swatch, not a retention setting."),
    ]
    for fixture_id, text in decoys:
        marker = fixture_id + "_MARKER"
        _record(records, fixture_id, main_ns, "note", marker, text)

    # 60 no-answer queries: 12 each across five distinct negative subtypes,
    # six per split. Several are hard negatives with vocabulary near seeded facts.
    negative_templates = {
        "hard_semantic": {
            "calibration": [
                "What fictional policy governs the silver cache's emergency salt rotation?",
                "How many workers run the nonexistent coral backup queue?",
                "Which rule sets the amber orchard's lunar failover?",
                "How long is the violet telescope's orbital renewal period?",
                "Which reviewer owns the green harbor's checksum approval?",
                "What is the red acorn cache's underwater retry schedule?",
            ],
            "holdout": [
                "Which rule controls the copper cache's emergency fog purge?",
                "How many workers are assigned to a marble maintenance queue?",
                "What is the quartz garden's eclipse failover policy?",
                "How long is the paper crane's sky renewal interval?",
                "Who approves the quiet bridge's glacier checksum?",
                "Which mode runs the bronze forest's submarine retry?",
            ],
        },
        "hard_lexical": {
            "calibration": [
                "What is the fictional silver cache policy for snow mode?",
                "Give the worker count for the cobalt queue, not the orchid queue.",
                "What duration is stored for the paper kite retention window?",
                "Which policy token belongs to the amber bridge approval group?",
                "What retry count is recorded for the quiet planet lease?",
                "Find the fictional blue feather routing value in the backup notes.",
            ],
            "holdout": [
                "What is the copper orchard cache policy for fog mode?",
                "Give the worker count for the maple queue, not the coral queue.",
                "What duration is stored for the velvet harbor retention window?",
                "Which policy token belongs to the brass leaf approval group?",
                "What retry count is recorded for the frosted bell lease?",
                "Find the fictional paper crane routing value in the handoff notes.",
            ],
        },
        "near_identifier": {
            "calibration": [
                "Find exact identifier SYNTH-C81-ID-9001.",
                "Find exact identifier SYNTH-C81-ID-9002.",
                "Find exact identifier SYNTH-C81-ID-9003.",
                "Find exact identifier SYNTH-C81-ID-9004.",
                "Find exact identifier SYNTH-C81-ID-9005.",
                "Find exact identifier SYNTH-C81-ID-9006.",
            ],
            "holdout": [
                "Find exact identifier SYNTH-C81-ID-9101.",
                "Find exact identifier SYNTH-C81-ID-9102.",
                "Find exact identifier SYNTH-C81-ID-9103.",
                "Find exact identifier SYNTH-C81-ID-9104.",
                "Find exact identifier SYNTH-C81-ID-9105.",
                "Find exact identifier SYNTH-C81-ID-9106.",
            ],
        },
        "scope_confusion": {
            "calibration": [
                "What label belongs to scope 01 chat B in the A namespace?",
                "What label belongs to scope 02 chat A in the B namespace?",
                "What label belongs to scope 06 chat A in the shared index?",
                "What label belongs to scope 07 chat B?",
                "Give the scope 08 chat A label from chat B's memory.",
                "What is the label for a fictional scope 99 chat C?",
            ],
            "holdout": [
                "What label belongs to scope 06 chat B in the A namespace?",
                "What label belongs to scope 07 chat A in the B namespace?",
                "What label belongs to scope 08 chat A in the default namespace?",
                "What label belongs to scope 09 chat B?",
                "Give the scope 10 chat A label from chat B's memory.",
                "What is the label for a fictional scope 98 chat C?",
            ],
        },
        "out_of_domain": {
            "calibration": [
                "What is the fictional lunar transit checksum for a submarine?",
                "Which invented crystal telescope has the violet transmission key?",
                "What is the cobalt waterfall's satellite routing password?",
                "Name the marble planet's nonexistent docking code.",
                "Which imaginary rainforest has a quartz moon phase?",
                "What is the invented glacier orchestra's tuning number?",
            ],
            "holdout": [
                "What is the fictional glacier ferry's magnetic berth code?",
                "Which invented coral observatory has an amber launch cipher?",
                "What is the bronze canyon's satellite beacon password?",
                "Name the velvet asteroid's nonexistent landing sequence.",
                "Which imaginary desert engine has a paper transmission key?",
                "What is the invented ocean choir's harmonic coordinate?",
            ],
        },
    }
    for subtype, by_split in negative_templates.items():
        for split in SPLITS:
            for n, query in enumerate(by_split[split], start=1):
                split_tag = "CAL" if split == "calibration" else "HOLD"
                case_id = f"Q81_NEG_{subtype.upper()}_{n:02d}_{split_tag}"
                forbidden = []
                if subtype == "scope_confusion":
                    scope_ids = {int(value) for value in re.findall(r"scope (\d+)", query, flags=re.IGNORECASE)}
                    forbidden = [marker for scope_id in scope_ids for marker in scope_markers.get(scope_id, [])]
                _case(cases, case_id, "negative_query", split, main_ns, query, [], forbidden=forbidden, subtype=subtype)

    dataset = {
        "schema_version": SCHEMA,
        "description": "Deterministic synthetic-only C8.1 abstention calibration dataset; calibration and holdout topics are disjoint.",
        "records": records,
        "cases": cases,
        "k": 5,
        "gates": {
            "positive_recall_at_k_min": 0.90,
            "exact_identifier_recall_min": 1.0,
            "negative_false_positive_rate_max": 0.05,
            "namespace_leaks_max": 0,
            "forbidden_marker_leaks_max": 0,
            "require_numeric_score_and_rrf": True,
            "require_modality_evidence": True,
        },
    }
    validate_dataset(dataset)
    return dataset


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != SCHEMA:
        raise ValueError("unsupported C8.1 fixture schema")
    records, cases = dataset.get("records"), dataset.get("cases")
    if not isinstance(records, list) or not isinstance(cases, list):
        raise ValueError("records and cases must be lists")
    if len(records) != 90 or len(cases) != 120:
        raise ValueError(f"expected 90 records and 120 cases; got {len(records)} and {len(cases)}")
    record_ids = [record.get("fixture_id", "") for record in records]
    case_ids = [case.get("case_id", "") for case in cases]
    if not all(record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("fixture IDs must be non-empty and unique")
    if not all(case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be non-empty and unique")
    queries = [case.get("query", "") for case in cases]
    if not all(isinstance(query, str) and query.strip() for query in queries) or len(set(queries)) != len(queries):
        raise ValueError("query text must be non-empty and unique across calibration and holdout")
    by_namespace: dict[str, list[str]] = {}
    all_content: list[str] = []
    for record in records:
        namespace, content = record.get("namespace", ""), record.get("content", "")
        if not namespace.startswith("c8-synthetic-"):
            raise ValueError(f"non-synthetic namespace: {namespace!r}")
        if not isinstance(content, str) or "Synthetic C8.1 fixture." not in content:
            raise ValueError(f"record {record['fixture_id']} is not synthetic")
        if SECRET_SHAPED.search(content):
            raise ValueError(f"record {record['fixture_id']} resembles a blocked secret-shaped string")
        by_namespace.setdefault(namespace, []).append(content)
        all_content.append(content)
    counts: dict[tuple[str, str], int] = {}
    subtype_counts: dict[tuple[str, str], int] = {}
    for case in cases:
        split, category = case.get("split"), case.get("category")
        if split not in SPLITS:
            raise ValueError(f"case {case['case_id']} has invalid split")
        counts[(split, category)] = counts.get((split, category), 0) + 1
        if category == "negative_query":
            subtype = case.get("subtype", "")
            subtype_counts[(split, subtype)] = subtype_counts.get((split, subtype), 0) + 1
        if case.get("min_score") != 0.0:
            raise ValueError("C8.1 must collect unthresholded top-K candidates")
        if not isinstance(case.get("answerable"), bool):
            raise ValueError("answerable label must be boolean")
        relevant, forbidden = case.get("relevant_markers"), case.get("forbidden_markers")
        if not isinstance(relevant, list) or not isinstance(forbidden, list):
            raise ValueError("relevant and forbidden markers must be lists")
        if bool(relevant) != case["answerable"]:
            raise ValueError(f"answerable label disagrees with markers for {case['case_id']}")
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"empty query: {case['case_id']}")
        namespace_content = "\n".join(by_namespace.get(case.get("namespace", ""), []))
        for marker in relevant:
            if marker not in namespace_content:
                raise ValueError(f"relevant marker {marker!r} not in scoped corpus for {case['case_id']}")
        for marker in forbidden:
            if not any(marker in content for content in all_content):
                raise ValueError(f"forbidden marker {marker!r} missing from corpus")
    if sum(bool(case["answerable"]) for case in cases) != 60:
        raise ValueError("expected exactly 60 answerable cases")
    if sum(not case["answerable"] for case in cases) != 60:
        raise ValueError("expected exactly 60 unanswerable cases")
    for split in SPLITS:
        split_cases = [case for case in cases if case["split"] == split]
        if sum(case["answerable"] for case in split_cases) != 30 or sum(not case["answerable"] for case in split_cases) != 30:
            raise ValueError(f"split {split} must contain 30 answerable and 30 unanswerable cases")
        expected_counts = {
            "exact_identifier": 10,
            "semantic_paraphrase": 5,
            "bilingual_paraphrase": 5,
            "namespace_isolation": 5,
            "active_vs_retired": 5,
            "negative_query": 30,
        }
        for category, expected in expected_counts.items():
            if counts.get((split, category), 0) != expected:
                raise ValueError(f"split {split} requires {expected} {category} cases")
        for subtype in NEGATIVE_SUBTYPES:
            if subtype_counts.get((split, subtype), 0) != 6:
                raise ValueError(f"split {split} requires six {subtype} negatives")
    if dataset.get("k") != 5:
        raise ValueError("C8.1 K must remain 5")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(args.output), "records": len(dataset["records"]), "cases": len(dataset["cases"]), "answerable": 60, "unanswerable": 60, "splits": {split: 60 for split in SPLITS}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

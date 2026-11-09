#!/usr/bin/env python3
"""Generate a fresh, deterministic synthetic C8.3 calibration/holdout corpus."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from c81_fixture_generator import SCHEMA_C83, SPLITS, validate_dataset


def _record(records: list[dict[str, str]], fixture_id: str, namespace: str, kind: str, marker: str, text: str) -> str:
    records.append({
        "fixture_id": fixture_id,
        "namespace": namespace,
        "type": kind,
        "content": f"Synthetic C8.3 fixture. Marker {marker}. {text}",
    })
    return marker


def _case(
    cases: list[dict[str, Any]], case_id: str, category: str, split: str,
    namespace: str, query: str, relevant: list[str], forbidden: list[str] | None = None,
    exclude_types: list[str] | None = None, subtype: str | None = None,
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
    main_ns = "c83-synthetic-main"

    # New identifier range and path family; none of the C8.1 identifiers repeat.
    for n in range(1, 21):
        split = "calibration" if n <= 10 else "holdout"
        token = f"SYNTH-C83-ID-{n:04d}"
        marker = _record(
            records, f"C83_EXACT_{n:02d}", main_ns, "fact", f"C83_EXACT_MARKER_{n:02d}",
            f"Fictional asset identifier {token} maps to /synthetic/c83/ledger/entry-{n:02d}.yaml.",
        )
        _case(cases, f"Q83_EXACT_{n:02d}", "exact_identifier", split, main_ns,
              f"Retrieve the ledger entry keyed by exact identifier {token}.", [marker])

    semantic = [
        ("birch relay", "uses a two-stage handoff", "How many stages are in the birch relay handoff?"),
        ("opal kiln", "runs in low-heat preview mode", "Which preview mode is assigned to the opal kiln?"),
        ("thistle archive", "keeps a rolling nine-day window", "What retention window does the thistle archive use?"),
        ("driftwood sensor", "reports once every thirteen minutes", "How often does the driftwood sensor report?"),
        ("amber compass", "follows the north-gate route", "Which route was chosen for the amber compass?"),
        ("willow shuttle", "has a four-seat capacity", "How many seats are configured for the willow shuttle?"),
        ("quartz garden", "uses a dawn-only irrigation cycle", "When does the quartz garden irrigation cycle run?"),
        ("indigo beacon", "retries twice before pausing", "How many retries does the indigo beacon allow?"),
        ("cedar ledger", "requires a two-person review", "How many reviewers must approve the cedar ledger?"),
        ("silver anvil", "uses a local-only staging lane", "Which staging lane is used by the silver anvil?"),
    ]
    for n, (topic, fact, query) in enumerate(semantic, start=1):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(records, f"C83_SEM_{n:02d}", main_ns, "decision", f"C83_SEM_MARKER_{n:02d}",
                         f"Fictional decision: the {topic} {fact}.")
        _case(cases, f"Q83_SEM_{n:02d}", "semantic_paraphrase", split, main_ns, query, [marker])

    bilingual_queries = [
        "Berapa jalur yang dipakai antrean lumut untuk demo?",
        "Kapan jadwal sinkronisasi kebun batu?",
        "Siapa yang memeriksa rancangan layang-layang kabut?",
        "Apa mode yang dipakai sensor lonceng?",
        "Berapa lama masa simpan catatan kompas?",
        "How many tiles are used by the fictional saffron board?",
        "Which review lane was chosen for the lantern ferry?",
        "What interval does the cloudberry monitor use?",
        "Who owns the fictional reed-delta approval?",
        "Which setting is used by the copperleaf trial?",
    ]
    bilingual_facts = [
        "Indonesian: antrean lumut memakai tiga jalur. English: the moss queue uses three lanes.",
        "Indonesian: kebun batu disinkronkan tiap tujuh jam. English: the stone garden syncs every seven hours.",
        "Indonesian: rancangan layang-layang kabut diperiksa reviewer satu. English: reviewer one checks the fog-kite design.",
        "Indonesian: sensor lonceng memakai mode observasi. English: the bell sensor uses observe mode.",
        "Indonesian: catatan kompas disimpan sebelas hari. English: compass notes are retained for eleven days.",
        "English: the fictional saffron board uses five tiles. Indonesian: papan safron memakai lima ubin.",
        "English: the lantern ferry uses the quiet-review lane. Indonesian: feri lentera memakai jalur tinjau senyap.",
        "English: the cloudberry monitor checks every seventeen minutes. Indonesian: monitor cloudberry memeriksa tiap tujuh belas menit.",
        "English: reviewer three owns the reed-delta approval. Indonesian: reviewer tiga menangani persetujuan delta alang-alang.",
        "English: the copperleaf trial uses staged mode. Indonesian: uji daun tembaga memakai mode bertahap.",
    ]
    for n, (query, fact) in enumerate(zip(bilingual_queries, bilingual_facts), start=1):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(records, f"C83_BILINGUAL_{n:02d}", main_ns, "decision", f"C83_BILINGUAL_MARKER_{n:02d}", fact)
        _case(cases, f"Q83_BILINGUAL_{n:02d}", "bilingual_paraphrase", split, main_ns, query, [marker])

    labels_a = ["moss violin", "opal bridge", "rust kite", "pearl orchard", "quiet lantern", "cedar tower", "ivory tunnel", "saffron wheel", "reed harbor", "misty compass"]
    labels_b = ["copper fern", "blue thimble", "amber ladder", "silver pebble", "coral window", "maple engine", "velvet pocket", "granite feather", "cloudberry arch", "brass acorn"]
    scope_markers: dict[int, list[str]] = {}
    for n, (label_a, label_b) in enumerate(zip(labels_a, labels_b), start=1):
        split = "calibration" if n <= 5 else "holdout"
        ns_a = f"c83-synthetic-scope-{n:02d}-a"
        ns_b = f"c83-synthetic-scope-{n:02d}-b"
        marker_a = _record(records, f"C83_SCOPE_{n:02d}_A", ns_a, "fact", f"C83_SCOPE_{n:02d}_A_MARKER", f"Tenant A scope {n:02d} has fictional label {label_a}.")
        marker_b = _record(records, f"C83_SCOPE_{n:02d}_B", ns_b, "fact", f"C83_SCOPE_{n:02d}_B_MARKER", f"Tenant B scope {n:02d} has fictional label {label_b}.")
        scope_markers[n] = [marker_a, marker_b]
        _case(cases, f"Q83_SCOPE_{n:02d}", "namespace_isolation", split, ns_a,
              f"In tenant A scope {n:02d}, what is its assigned label?", [marker_a], [marker_b])

    active_retired = [
        ("parcel", "three sorting bins", "eight sorting bins"),
        ("orchard", "a morning-only schedule", "an all-day schedule"),
        ("beacon", "one confirmation step", "five confirmation steps"),
        ("cabinet", "a six-day retention window", "a sixty-day retention window"),
        ("courier", "two delivery attempts", "seven delivery attempts"),
        ("fountain", "a local preview", "a public launch"),
        ("instrument", "a single calibration pass", "four calibration passes"),
        ("ledger", "a weekly checkpoint", "a monthly checkpoint"),
        ("bridge", "manual inspection", "automatic acceptance"),
        ("signal", "a ten-minute interval", "a one-hour interval"),
    ]
    for n, (topic, active, retired) in enumerate(active_retired, start=1):
        split = "calibration" if n <= 5 else "holdout"
        active_marker = _record(records, f"C83_ACTIVE_{n:02d}", main_ns, "constraint", f"C83_ACTIVE_MARKER_{n:02d}", f"Current fictional {topic} rule is {active}.")
        retired_marker = _record(records, f"C83_RETIRED_{n:02d}", main_ns, "superseded", f"C83_RETIRED_MARKER_{n:02d}", f"Retired fictional {topic} rule was {retired}.")
        _case(cases, f"Q83_ACTIVE_{n:02d}", "active_vs_retired", split, main_ns,
              f"What is the current fictional {topic} rule?", [active_marker], [retired_marker], ["superseded"])

    decoys = [
        "The birch relay casing is painted orange; this note says nothing about handoff stages.",
        "An opal kiln illustration has a blue border, not a heat or preview setting.",
        "A thistle archive cover is decorative and gives no retention duration.",
        "The driftwood sensor sketch contains a wave icon, not a reporting interval.",
        "A copperleaf logo is a color sample and has no trial configuration.",
        "The moss queue diagram shows arrows but no number of lanes.",
        "A reed harbor postcard is not a tenant label or scope assignment.",
        "The lantern ferry picture has no review-lane information.",
        "A granite feather drawing is a placeholder and defines no approval owner.",
        "The pearl orchard mural is fictional artwork, not an irrigation schedule.",
    ]
    for n, text in enumerate(decoys, start=1):
        _record(records, f"C83_DECOY_{n:02d}", main_ns, "note", f"C83_DECOY_MARKER_{n:02d}", text)

    negative_templates = {
        "hard_semantic": {
            "calibration": [
                "What fictional handoff stages does the birch relay use during an eclipse?",
                "How many inspection crews are assigned to the opal kiln's moon shift?",
                "Which frost-recovery rule controls the thistle archive?",
                "How long is the driftwood sensor's underwater maintenance cycle?",
                "Which reviewer owns the amber compass's satellite transfer?",
                "What is the fog-delivery schedule for the willow shuttle?",
            ],
            "holdout": [
                "Which dawn failover policy controls the quartz garden's winter route?",
                "How many night crews operate the indigo beacon's glacier mode?",
                "What is the cedar ledger's magnetic-archive window?",
                "Who approves the silver anvil's submarine staging lane?",
                "What is the cloudberry monitor's lunar calibration cycle?",
                "Which rule sets the copperleaf trial's desert handoff?",
            ],
        },
        "hard_lexical": {
            "calibration": [
                "What fictional birch relay policy uses violet staging?",
                "Give the worker count for the moss queue's backup route.",
                "What duration is stored for the copper fern retention rule?",
                "Which review token belongs to the lantern ferry night group?",
                "What attempt count is recorded for the parcel courier's lease?",
                "Find the fictional sapphire routing value in the garden notes.",
            ],
            "holdout": [
                "What is the quartz garden policy for storm mode?",
                "Give the retry count for the maple engine's archive route.",
                "What duration is stored for the velvet pocket review window?",
                "Which approval token belongs to the brass acorn group?",
                "What lane count is recorded for the cloudberry bridge?",
                "Find the fictional ivory handoff value in the beacon notes.",
            ],
        },
        "near_identifier": {
            "calibration": [f"Retrieve exact identifier SYNTH-C83-ID-{n:04d}." for n in range(9001, 9007)],
            "holdout": [f"Retrieve exact identifier SYNTH-C83-ID-{n:04d}." for n in range(9101, 9107)],
        },
        "scope_confusion": {
            "calibration": [
                "What label belongs to tenant B scope 01 while searching tenant A?",
                "Give the tenant A label for scope 02 from the B namespace.",
                "What label belongs to scope 03 in the shared namespace?",
                "Return the tenant B label for scope 04 without a scope filter.",
                "Give the tenant A scope 05 label from tenant B's memory.",
                "What is the fictional tenant C label for scope 99?",
            ],
            "holdout": [
                "What label belongs to tenant B scope 06 while searching tenant A?",
                "Give the tenant A label for scope 07 from the B namespace.",
                "What label belongs to scope 08 in the shared namespace?",
                "Return the tenant B label for scope 09 without a scope filter.",
                "Give the tenant A scope 10 label from tenant B's memory.",
                "What is the fictional tenant C label for scope 98?",
            ],
        },
        "out_of_domain": {
            "calibration": [
                "What is the imaginary comet's submarine docking coordinate?",
                "Which invented glacier orchestra has the amber tuning cipher?",
                "Name the fictional asteroid's underwater launch sequence.",
                "What is the quartz volcano's nonexistent telescope password?",
                "Which imaginary desert engine has a sapphire moon phase?",
                "Give the invented ocean choir's harmonic address.",
            ],
            "holdout": [
                "What is the fictional rainforest ferry's magnetic berth code?",
                "Which invented coral observatory has a violet launch cipher?",
                "Name the bronze canyon's satellite beacon sequence.",
                "What is the velvet meteor's nonexistent landing password?",
                "Which imaginary tundra engine has a paper transmission key?",
                "Give the invented mountain choir's harmonic coordinate.",
            ],
        },
    }
    for subtype, by_split in negative_templates.items():
        for split in SPLITS:
            for n, query in enumerate(by_split[split], start=1):
                split_tag = "CAL" if split == "calibration" else "HOLD"
                case_id = f"Q83_NEG_{subtype.upper()}_{n:02d}_{split_tag}"
                forbidden = []
                if subtype == "scope_confusion":
                    scope_ids = {int(value) for value in re.findall(r"scope (\d+)", query, flags=re.IGNORECASE)}
                    forbidden = [marker for scope_id in scope_ids for marker in scope_markers.get(scope_id, [])]
                _case(cases, case_id, "negative_query", split, main_ns, query, [], forbidden, subtype=subtype)

    dataset = {
        "schema_version": SCHEMA_C83,
        "description": "Fresh C8.3 synthetic-only corpus with disjoint calibration and previously unseen holdout cases.",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixture": str(args.output),
        "schema_version": dataset["schema_version"],
        "records": len(dataset["records"]),
        "cases": len(dataset["cases"]),
        "calibration": sum(case["split"] == "calibration" for case in dataset["cases"]),
        "holdout": sum(case["split"] == "holdout" for case in dataset["cases"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

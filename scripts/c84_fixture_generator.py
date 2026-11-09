#!/usr/bin/env python3
"""Generate a fresh C8.4 synthetic corpus for raw-modality calibration."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from c81_fixture_generator import SCHEMA_C84, SPLITS, validate_dataset


def _record(records: list[dict[str, str]], fixture_id: str, namespace: str, kind: str, marker: str, text: str) -> str:
    records.append({
        "fixture_id": fixture_id,
        "namespace": namespace,
        "type": kind,
        "content": f"Synthetic C8.4 fixture. Marker {marker}. {text}",
    })
    return marker


def _case(
    cases: list[dict[str, Any]], case_id: str, category: str, split: str,
    namespace: str, query: str, relevant: list[str], forbidden: list[str] | None = None,
    exclude_types: list[str] | None = None, subtype: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "case_id": case_id, "category": category, "split": split,
        "answerable": bool(relevant), "namespace": namespace, "query": query,
        "relevant_markers": relevant, "forbidden_markers": forbidden or [],
        "exclude_types": exclude_types or [], "min_score": 0.0,
    }
    if subtype:
        row["subtype"] = subtype
    cases.append(row)


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []
    main_ns = "c84-synthetic-main"

    for n in range(1, 21):
        split = "calibration" if n <= 10 else "holdout"
        token = f"SYNTH-C84-ID-{n:04d}"
        marker = _record(records, f"C84_CATALOG_{n:02d}", main_ns, "fact", f"C84_CATALOG_MARKER_{n:02d}",
                         f"Fictional register code {token} resolves to /synthetic/c84/catalog/item-{n:02d}.json.")
        _case(cases, f"Q84_CATALOG_{n:02d}", "exact_identifier", split, main_ns,
              f"Resolve the invented C8.4 register record for exact code {token}.", [marker])

    semantic = [
        ("maple index", "keeps a five-entry rolling window", "What window length is configured for the maple index?"),
        ("saffron drain", "opens after a twelve-second delay", "How long does the saffron drain wait before opening?"),
        ("cobalt envelope", "uses the inland courier lane", "Which lane carries the cobalt envelope?"),
        ("fig lantern", "needs three confirmation checks", "How many confirmations does the fig lantern require?"),
        ("marble switch", "stays in reversible preview mode", "Which mode is active for the marble switch?"),
        ("juniper counter", "resets at the fourth daily interval", "At which interval does the juniper counter reset?"),
        ("copper atlas", "keeps a north-west route lock", "What route lock is assigned to the copper atlas?"),
        ("violet drum", "sends a pulse every nineteen minutes", "What is the violet drum pulse interval?"),
        ("granite loom", "requires a paired operator review", "What review arrangement does the granite loom use?"),
        ("plum cabinet", "retains a local audit copy for eight days", "How long is the plum cabinet audit copy retained?"),
    ]
    for n, (topic, fact, query) in enumerate(semantic, start=1):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(records, f"C84_SEMANTIC_{n:02d}", main_ns, "decision", f"C84_SEMANTIC_MARKER_{n:02d}",
                         f"Fictional operating rule: the {topic} {fact}.")
        _case(cases, f"Q84_SEMANTIC_{n:02d}", "semantic_paraphrase", split, main_ns, query, [marker])

    bilingual = [
        ("Berapa katup yang digunakan kanal cemara?", "Indonesian: kanal cemara memakai dua katup. English: the fir channel uses two valves."),
        ("Kapan baki batu bara diperiksa?", "Indonesian: baki arang diperiksa tiap enam jam. English: the charcoal tray is inspected every six hours."),
        ("Siapa yang meninjau protokol balon hujan?", "Indonesian: operator dua meninjau protokol balon hujan. English: operator two reviews the rain-balloon protocol."),
        ("Apa status uji kompas kaca?", "Indonesian: uji kompas kaca memakai status karantina. English: the glass compass trial uses quarantine status."),
        ("Berapa lama arsip kerikil disimpan?", "Indonesian: arsip kerikil disimpan selama empat belas hari. English: the pebble archive is retained for fourteen days."),
        ("Which setting controls the fictional willow marker?", "English: the willow marker uses quiet mode. Indonesian: penanda willow memakai mode senyap."),
        ("How many shelves are assigned to the clay registry?", "English: the clay registry has seven shelves. Indonesian: registri tanah liat memiliki tujuh rak."),
        ("When does the tin orchard perform its check?", "English: the tin orchard checks at dusk. Indonesian: kebun timah diperiksa saat senja."),
        ("Who validates the blue-mica request?", "English: reviewer four validates the blue-mica request. Indonesian: peninjau empat memvalidasi permintaan mika biru."),
        ("Which route was selected for the wax kite?", "English: the wax kite follows the east-loop route. Indonesian: layang-layang lilin mengikuti rute lingkar timur."),
    ]
    for n, (query, fact) in enumerate(bilingual, start=1):
        split = "calibration" if n <= 5 else "holdout"
        marker = _record(records, f"C84_BILINGUAL_{n:02d}", main_ns, "decision", f"C84_BILINGUAL_MARKER_{n:02d}", fact)
        _case(cases, f"Q84_BILINGUAL_{n:02d}", "bilingual_paraphrase", split, main_ns, query, [marker])

    labels_a = ["bronze lily", "moss dial", "teal needle", "ivory dock", "birch lens", "amber glove", "cinder bell", "linen arch", "clay feather", "opal gate"]
    labels_b = ["silver reed", "coral pin", "plum wheel", "jade spool", "wool compass", "paper bridge", "cobalt cup", "maple bellows", "tin ladder", "onyx sleeve"]
    scope_markers: dict[int, list[str]] = {}
    for n, (label_a, label_b) in enumerate(zip(labels_a, labels_b), start=1):
        split = "calibration" if n <= 5 else "holdout"
        ns_a, ns_b = f"c84-synthetic-scope-{n:02d}-a", f"c84-synthetic-scope-{n:02d}-b"
        marker_a = _record(records, f"C84_SCOPE_{n:02d}_A", ns_a, "fact", f"C84_SCOPE_{n:02d}_A_MARKER", f"Workspace A scope {n:02d} has fictional label {label_a}.")
        marker_b = _record(records, f"C84_SCOPE_{n:02d}_B", ns_b, "fact", f"C84_SCOPE_{n:02d}_B_MARKER", f"Workspace B scope {n:02d} has fictional label {label_b}.")
        scope_markers[n] = [marker_a, marker_b]
        _case(cases, f"Q84_SCOPE_{n:02d}", "namespace_isolation", split, ns_a,
              f"In workspace A scope {n:02d}, which label is assigned?", [marker_a], [marker_b])

    state_rules = [
        ("kiln", "two warm-up cycles", "six warm-up cycles"),
        ("notebook", "a four-day expiry", "a forty-day expiry"),
        ("shuttle", "manual dispatch", "automatic dispatch"),
        ("waterwheel", "a three-hour pause", "a thirty-hour pause"),
        ("circuit", "a single validation pass", "five validation passes"),
        ("parcel", "a south-depot handoff", "a west-depot handoff"),
        ("wind sock", "a red warning state", "a green warning state"),
        ("key cabinet", "two custodian approvals", "one custodian approval"),
        ("lamp", "a dusk restart window", "a noon restart window"),
        ("bridge log", "local checksum verification", "remote signature verification"),
    ]
    for n, (topic, current, old) in enumerate(state_rules, start=1):
        split = "calibration" if n <= 5 else "holdout"
        active_marker = _record(records, f"C84_CURRENT_{n:02d}", main_ns, "constraint", f"C84_CURRENT_MARKER_{n:02d}", f"Current fictional {topic} setting is {current}.")
        retired_marker = _record(records, f"C84_OLD_{n:02d}", main_ns, "superseded", f"C84_OLD_MARKER_{n:02d}", f"Retired fictional {topic} setting was {old}.")
        _case(cases, f"Q84_CURRENT_{n:02d}", "active_vs_retired", split, main_ns,
              f"State the non-retired setting for the fictional {topic}.", [active_marker], [retired_marker], ["superseded"])

    decoys = [
        "The maple index cover uses a square icon and has no retention setting.",
        "The saffron drain sketch is decorative and states no delay value.",
        "A cobalt envelope drawing has no courier-lane assignment.",
        "The fig lantern casing color does not define confirmation count.",
        "A marble switch photograph gives no operating mode.",
        "The juniper counter poster contains no reset interval.",
        "The copper atlas cover has no route-lock information.",
        "A violet drum illustration states no pulse interval.",
        "The granite loom ornament has no review policy.",
        "A plum cabinet label has no audit-retention duration.",
    ]
    for n, text in enumerate(decoys, start=1):
        _record(records, f"C84_DECOY_{n:02d}", main_ns, "note", f"C84_DECOY_MARKER_{n:02d}", text)

    negatives = {
        "hard_semantic": {
            "calibration": [
                "Which maple-index lunar emergency controls its sixth archive window?",
                "How many underwater warm-up shifts does the saffron drain schedule?",
                "Who owns the cobalt envelope's glacier delivery protocol?",
                "What is the fig lantern's midnight recovery sequence?",
                "Which marble switch rule authorizes a desert relay?",
                "How long is the juniper counter's polar-night service interval?",
            ],
            "holdout": [
                "Which copper atlas rule schedules a volcanic route transfer?",
                "What is the violet drum's ocean-floor calibration cycle?",
                "Who approves the granite loom's solar handoff?",
                "Which plum cabinet policy manages a lunar checksum?",
                "How many canyon recovery teams operate the willow marker?",
                "What is the wax kite's glacier retention schedule?",
            ],
        },
        "hard_lexical": {
            "calibration": [
                "Give the fictional maple-index operator's satellite password.",
                "What spool count belongs to the cobalt envelope review group?",
                "Find the saffron drain's nonexistent archive key.",
                "Which fig lantern tenant owns the blue-mica approval token?",
                "What circuit lane is assigned to the pearl orchard?",
                "Return the wax kite's missing checksum phrase.",
            ],
            "holdout": [
                "What review token belongs to the copper atlas night shift?",
                "Give the retention password for the violet drum archive.",
                "Find the granite loom's nonexistent approval code.",
                "Which linen arch tenant owns the cobalt handoff secret?",
                "What parcel route is assigned to the onyx sleeve?",
                "Return the tin ladder's missing checksum phrase.",
            ],
        },
        "near_identifier": {
            "calibration": [f"Resolve exact catalog code SYNTH-C84-ID-{n:04d}." for n in range(9001, 9007)],
            "holdout": [f"Resolve exact catalog code SYNTH-C84-ID-{n:04d}." for n in range(9101, 9107)],
        },
        "scope_confusion": {
            "calibration": [
                "Return workspace B's label for scope 01 while scoped to A.",
                "Give workspace A's scope 02 label from workspace B.",
                "What is the label for scope 03 in a shared workspace?",
                "Return workspace B's label for scope 04 without a tenant filter.",
                "Give the A label for scope 05 from B's namespace.",
                "What is workspace C's fictional label for scope 99?",
            ],
            "holdout": [
                "Return workspace B's label for scope 06 while scoped to A.",
                "Give workspace A's scope 07 label from workspace B.",
                "What is the label for scope 08 in a shared workspace?",
                "Return workspace B's label for scope 09 without a tenant filter.",
                "Give the A label for scope 10 from B's namespace.",
                "What is workspace C's fictional label for scope 98?",
            ],
        },
        "out_of_domain": {
            "calibration": [
                "Which imaginary comet operates the underwater library lift?",
                "What is the invented tundra orchestra's crystal orbit number?",
                "Name the fictional volcano's velvet submarine harbor.",
                "Which paper asteroid has a blue moon engine?",
                "Give the imaginary ocean lighthouse's brass constellation.",
                "What is the invented canyon whale's silent weather key?",
            ],
            "holdout": [
                "Which imaginary forest meteor has a glass docking season?",
                "What is the invented glacier theatre's copper planet code?",
                "Name the fictional desert submarine's linen horizon.",
                "Which jade asteroid has a woolen moon engine?",
                "Give the imaginary mountain telescope's paper constellation.",
                "What is the invented coral whale's silent weather key?",
            ],
        },
    }
    for subtype, by_split in negatives.items():
        for split in SPLITS:
            for n, query in enumerate(by_split[split], start=1):
                split_tag = "CAL" if split == "calibration" else "HOLD"
                forbidden = []
                if subtype == "scope_confusion":
                    scope_ids = {int(v) for v in re.findall(r"scope (\d+)", query, flags=re.IGNORECASE)}
                    forbidden = [m for scope_id in scope_ids for m in scope_markers.get(scope_id, [])]
                _case(cases, f"Q84_NEG_{subtype.upper()}_{n:02d}_{split_tag}", "negative_query", split, main_ns, query, [], forbidden, subtype=subtype)

    dataset = {
        "schema_version": SCHEMA_C84,
        "description": "Fresh C8.4 synthetic-only corpus with new source-score features and an unopened holdout.",
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
            "require_raw_modality_scores": True,
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
        "fixture": str(args.output), "schema_version": dataset["schema_version"],
        "records": len(dataset["records"]), "cases": len(dataset["cases"]),
        "calibration": sum(row["split"] == "calibration" for row in dataset["cases"]),
        "holdout": sum(row["split"] == "holdout" for row in dataset["cases"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a fresh, split-isolated C8.6 abstention fixture with identifier guards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from c81_fixture_generator import SCHEMA_C86, validate_dataset


SEMANTIC_FACTS = {
    "calibration": [
        ("ember relay", "keeps fourteen jobs in the cold pool", "How many jobs does the ember relay keep in cold storage?"),
        ("cobalt ledger", "refreshes its mirror at the sixth interval", "At which interval does the cobalt ledger refresh its mirror?"),
        ("willow beacon", "uses a two-reviewer handoff", "How many reviewers are required for the willow beacon handoff?"),
        ("amber spool", "retains its local digest for nine days", "How long does the amber spool retain its local digest?"),
        ("granite queue", "sends a batch after eleven records", "How many records trigger a granite queue batch?"),
        ("plum sensor", "uses quiet mode during the evening window", "Which mode does the plum sensor use in the evening?"),
        ("copper atlas", "keeps the west-loop route locked", "Which route remains locked for the copper atlas?"),
        ("marble archive", "stores the signed copy in the north cabinet", "Where is the marble archive signed copy stored?"),
        ("juniper switch", "waits seven seconds before retrying", "How long does the juniper switch wait before retrying?"),
        ("silver channel", "requires a paired operator review", "What review arrangement does the silver channel require?"),
    ],
    "holdout": [
        ("birch relay", "keeps seventeen jobs in the warm pool", "How many jobs does the birch relay keep in warm storage?"),
        ("teal ledger", "refreshes its mirror at the ninth interval", "At which interval does the teal ledger refresh its mirror?"),
        ("ivory beacon", "uses a three-reviewer handoff", "How many reviewers are required for the ivory beacon handoff?"),
        ("saffron spool", "retains its local digest for twelve days", "How long does the saffron spool retain its local digest?"),
        ("quartz queue", "sends a batch after thirteen records", "How many records trigger a quartz queue batch?"),
        ("coral sensor", "uses observe-only mode during the morning window", "Which mode does the coral sensor use in the morning?"),
        ("maple atlas", "keeps the east-loop route locked", "Which route remains locked for the maple atlas?"),
        ("velvet archive", "stores the signed copy in the east cabinet", "Where is the velvet archive signed copy stored?"),
        ("cinder switch", "waits nine seconds before retrying", "How long does the cinder switch wait before retrying?"),
        ("opal channel", "requires a paired auditor review", "What review arrangement does the opal channel require?"),
    ],
}

SEMANTIC_CONTEXT = {
    "calibration": ["cold pool", "mirror index", "reviewer handoff", "local digest", "batch trigger", "evening mode", "west-loop route", "signed copy", "retry wait", "operator review"],
    "holdout": ["warm pool", "snapshot mirror", "reviewer handoff", "local digest", "batch trigger", "morning mode", "east-loop route", "signed copy", "retry wait", "auditor review"],
}

BILINGUAL_FACTS = {
    "calibration": [
        ("Berapa katup yang dipakai kanal pinus?", "Indonesian: kanal pinus memakai tiga katup. English: the pine channel uses three valves."),
        ("Kapan baki zirkon diperiksa?", "Indonesian: baki zirkon diperiksa setiap lima jam. English: the zircon tray is inspected every five hours."),
        ("Siapa yang meninjau protokol balon senja?", "Indonesian: operator tiga meninjau protokol balon senja. English: operator three reviews the dusk-balloon protocol."),
        ("Apa status uji kompas porselen?", "Indonesian: uji kompas porselen memakai status observasi. English: the porcelain compass trial uses observe status."),
        ("Berapa lama gudang gading disimpan?", "Indonesian: gudang gading disimpan selama sebelas hari. English: the ivory storehouse is retained for eleven days."),
        ("How many shelves are assigned to the moss registry?", "English: the moss registry has six shelves. Indonesian: registri lumut memiliki enam rak."),
        ("When does the brass orchard run its check?", "English: the brass orchard checks at sunrise. Indonesian: kebun kuningan diperiksa saat matahari terbit."),
        ("Which route was selected for the linen kite?", "English: the linen kite follows the north-loop route. Indonesian: layang-layang linen mengikuti rute lingkar utara."),
    ],
    "holdout": [
        ("Berapa katup yang dipakai kanal cemara?", "Indonesian: kanal cemara memakai empat katup. English: the fir channel uses four valves."),
        ("Kapan baki basalt diperiksa?", "Indonesian: baki basalt diperiksa setiap delapan jam. English: the basalt tray is inspected every eight hours."),
        ("Siapa yang meninjau protokol layang kabut?", "Indonesian: operator lima meninjau protokol layang kabut. English: operator five reviews the fog-kite protocol."),
        ("Apa status uji prisma kaca?", "Indonesian: uji prisma kaca memakai status karantina. English: the glass prism trial uses quarantine status."),
        ("Berapa lama arsip marmer disimpan?", "Indonesian: arsip marmer disimpan selama enam belas hari. English: the marble archive is retained for sixteen days."),
        ("How many shelves are assigned to the reed registry?", "English: the reed registry has nine shelves. Indonesian: registri alang-alang memiliki sembilan rak."),
        ("When does the slate orchard run its check?", "English: the slate orchard checks at dusk. Indonesian: kebun batu tulis diperiksa saat senja."),
        ("Which route was selected for the canvas kite?", "English: the canvas kite follows the south-loop route. Indonesian: layang-layang kanvas mengikuti rute lingkar selatan."),
    ],
}

ACTIVE_FACTS = {
    "calibration": [
        ("kiln", "two warm-up cycles", "six warm-up cycles"),
        ("notebook", "a five-day expiry", "a fifty-day expiry"),
        ("shuttle", "manual dispatch", "automatic dispatch"),
        ("waterwheel", "a four-hour pause", "a forty-hour pause"),
        ("circuit", "a single validation pass", "seven validation passes"),
        ("parcel", "a north-depot handoff", "a west-depot handoff"),
    ],
    "holdout": [
        ("furnace", "three warm-up cycles", "eight warm-up cycles"),
        ("journal", "a six-day expiry", "a sixty-day expiry"),
        ("tram", "supervised dispatch", "automatic dispatch"),
        ("turbine", "a five-hour pause", "a fifty-hour pause"),
        ("gateway", "two validation passes", "nine validation passes"),
        ("shipment", "an east-depot handoff", "a south-depot handoff"),
    ],
}

LABELS_A = {
    "calibration": ["amber kite", "violet compass", "green comet", "red acorn", "quiet planet", "white lantern"],
    "holdout": ["birch feather", "teal harbor", "silver drum", "ivory tunnel", "saffron wheel", "granite balloon"],
}
LABELS_B = {
    "calibration": ["copper heron", "golden sail", "blue feather", "bright river", "marble cloud", "maple wheel"],
    "holdout": ["opal bridge", "cobalt lantern", "linen orchard", "coral mirror", "basalt ribbon", "plum engine"],
}


def _record(records: list[dict[str, str]], split: str, fixture_id: str, namespace: str, kind: str, marker: str, text: str) -> str:
    records.append({
        "fixture_id": fixture_id,
        "split": split,
        "namespace": namespace,
        "type": kind,
        "content": f"Synthetic C8.6 fixture. Marker {marker}. {text}",
    })
    return marker


def _case(
    cases: list[dict[str, Any]], split: str, case_id: str, category: str,
    namespace: str, query: str, relevant: list[str] | None = None,
    forbidden: list[str] | None = None, exclude_types: list[str] | None = None,
    subtype: str | None = None,
) -> None:
    relevant = relevant or []
    row: dict[str, Any] = {
        "case_id": case_id, "category": category, "split": split,
        "answerable": bool(relevant), "namespace": namespace, "query": query,
        "relevant_markers": relevant, "forbidden_markers": forbidden or [],
        "exclude_types": exclude_types or [], "min_score": 0.0,
    }
    if subtype:
        row["subtype"] = subtype
    cases.append(row)


def _code(split: str, number: int) -> str:
    base = 1000 if split == "calibration" else 5000
    serial = base + number
    style = (number - 1) % 5
    if style == 0:
        return f"ERR-2026-{serial:04d}"
    if style == 1:
        return f"DOC_{chr(65 + number % 26)}{serial:04d}"
    if style == 2:
        return f"TKT/2026/{serial:04d}"
    if style == 3:
        return f"POL.26.{serial:04d}"
    return f"INV-{chr(65 + number % 26)}{chr(65 + (number + 7) % 26)}-{serial:04d}"


def _near_code(split: str, number: int) -> str:
    # Close to an existing code family, but absent from every fixture record.
    return _code(split, number + 20)


def build_dataset() -> dict[str, Any]:
    records: list[dict[str, str]] = []
    cases: list[dict[str, Any]] = []
    for split in ("calibration", "holdout"):
        prefix = "CAL" if split == "calibration" else "HOLD"
        main_ns = f"c86-synthetic-{split}-main"
        semantic = SEMANTIC_FACTS[split]
        bilingual = BILINGUAL_FACTS[split]
        active = ACTIVE_FACTS[split]

        for n in range(1, 21):
            code = _code(split, n)
            marker = _record(records, split, f"C86_{prefix}_EXACT_{n:03d}", main_ns, "fact",
                             f"C86_{prefix}_EXACT_MARKER_{n:03d}",
                             f"Exact reference {code} resolves to /synthetic/c86/{prefix.lower()}/record-{n:03d}.json.")
            _case(cases, split, f"Q86_{prefix}_EXACT_{n:03d}", "exact_identifier", main_ns,
                  f"Find the record for exact identifier {code}.", [marker])

        for n, (topic, fact, query) in enumerate(semantic, start=1):
            marker = _record(records, split, f"C86_{prefix}_SEM_{n:03d}", main_ns, "decision",
                             f"C86_{prefix}_SEM_MARKER_{n:03d}", f"Fictional operating rule: the {topic} {fact}.")
            _case(cases, split, f"Q86_{prefix}_SEM_{n:03d}", "semantic_paraphrase", main_ns, query, [marker])

        for n, (query, fact) in enumerate(bilingual, start=1):
            marker = _record(records, split, f"C86_{prefix}_BI_{n:03d}", main_ns, "decision",
                             f"C86_{prefix}_BI_MARKER_{n:03d}", fact)
            _case(cases, split, f"Q86_{prefix}_BI_{n:03d}", "bilingual_paraphrase", main_ns, query, [marker])

        scope_markers: dict[int, tuple[str, str, str, str]] = {}
        for n, (label_a, label_b) in enumerate(zip(LABELS_A[split], LABELS_B[split]), start=1):
            ns_a = f"c86-synthetic-{split}-scope-{n:02d}-a"
            ns_b = f"c86-synthetic-{split}-scope-{n:02d}-b"
            marker_a = _record(records, split, f"C86_{prefix}_SCOPE_{n:02d}_A", ns_a, "fact",
                               f"C86_{prefix}_SCOPE_{n:02d}_A_MARKER", f"Workspace A scope {n:02d} has fictional label {label_a}.")
            marker_b = _record(records, split, f"C86_{prefix}_SCOPE_{n:02d}_B", ns_b, "fact",
                               f"C86_{prefix}_SCOPE_{n:02d}_B_MARKER", f"Workspace B scope {n:02d} has fictional label {label_b}.")
            scope_markers[n] = (ns_a, ns_b, marker_a, marker_b)
            _case(cases, split, f"Q86_{prefix}_SCOPE_{n:02d}", "namespace_isolation", ns_a,
                  f"In the {prefix.lower()} trial, which label is assigned in workspace A scope {n:02d}?", [marker_a], [marker_b])

        for n, (topic, current, retired) in enumerate(active, start=1):
            active_marker = _record(records, split, f"C86_{prefix}_ACTIVE_{n:02d}", main_ns, "constraint",
                                    f"C86_{prefix}_ACTIVE_MARKER_{n:02d}", f"Current fictional {topic} setting is {current}.")
            retired_marker = _record(records, split, f"C86_{prefix}_RETIRED_{n:02d}", main_ns, "superseded",
                                     f"C86_{prefix}_RETIRED_MARKER_{n:02d}", f"Retired fictional {topic} setting was {retired}.")
            _case(cases, split, f"Q86_{prefix}_ACTIVE_{n:02d}", "active_vs_retired", main_ns,
                  f"For the {prefix.lower()} ledger, which setting remains active for the fictional {topic}?", [active_marker], [retired_marker], ["superseded"])

        for n in range(1, 21):
            topic, _, _ = semantic[(n - 1) % len(semantic)]
            marker = _record(records, split, f"C86_{prefix}_DECOY_{n:03d}", main_ns, "note",
                             f"C86_{prefix}_DECOY_MARKER_{n:03d}",
                             f"The {topic} sketch mentions an emergency override password and a recovery token, but defines neither value.")
            _case(cases, split, f"Q86_{prefix}_LEX_{n:03d}", "negative_query", main_ns,
                  f"Give the {topic} emergency override password and recovery token for case {n:03d}.",
                  subtype="hard_lexical")

        # Thirty absent codes per split exercise literal matching across five formats.
        for n in range(1, 31):
            code = _near_code(split, n)
            _case(cases, split, f"Q86_{prefix}_NEAR_{n:03d}", "negative_query", main_ns,
                  f"Resolve the exact record for identifier {code}.", subtype="near_identifier")

        for n in range(1, 21):
            topic, _, _ = semantic[(n - 1) % len(semantic)]
            context = SEMANTIC_CONTEXT[split][(n - 1) % len(semantic)]
            if n <= 10:
                query = f"Who maintains the {topic} {context}?" if n % 2 else f"What recovery code protects the {topic} {context}?"
            else:
                query = f"Which operator owns the {context} used by the {topic}?" if n % 2 else f"What fallback secret secures the {topic} {context}?"
            _case(cases, split, f"Q86_{prefix}_HSEM_{n:03d}", "negative_query", main_ns,
                  query, subtype="hard_semantic")

        for n in range(1, 11):
            scope = (n - 1) % len(scope_markers) + 1
            ns_a, _, _, marker_b = scope_markers[scope]
            wording = "return workspace B's label while scoped to A" if n % 2 else "give the other tenant's label without leaving workspace A"
            _case(cases, split, f"Q86_{prefix}_SCOPE_NEG_{n:03d}", "negative_query", ns_a,
                  f"For the {prefix.lower()} assessment {n:02d}, scope {scope:02d}: {wording}; use only the current namespace.", forbidden=[marker_b], subtype="scope_confusion")

        out_of_domain = [
            "What is the boiling point of pure water at sea level?",
            "Explain the difference between a comet and an asteroid.",
            "How do I make sourdough starter from rye flour?",
            "What year was the first public railway opened?",
            "Why do leaves change color in autumn?",
            "Describe the basic rules of table tennis doubles.",
            "What causes tides on the open ocean?",
            "How does a solar eclipse differ from a lunar eclipse?",
            "Which instruments are used in a string quartet?",
            "What is the chemical symbol for tungsten?",
        ]
        for n, query in enumerate(out_of_domain, start=1):
            _case(cases, split, f"Q86_{prefix}_OOD_{n:03d}", "negative_query", main_ns,
                  f"{query} Reference {prefix.lower()} {n:03d}.", subtype="out_of_domain")

        for n in range(1, 11):
            topic, _, _ = active[(n - 1) % len(active)]
            field = ("recovery password" if n % 2 else "solar activation phrase") if n <= 6 else ("emergency override phrase" if n % 2 else "backup contact")
            _case(cases, split, f"Q86_{prefix}_MISSING_{n:03d}", "negative_query", main_ns,
                  f"In the {prefix.lower()} trial, what is the current {field} for the fictional {topic}?", subtype="missing_attribute")

    dataset: dict[str, Any] = {
        "schema_version": SCHEMA_C86,
        "description": (
            "Fresh C8.6 identifier-aware calibration and holdout fixture. 50 positive and 100 negative "
            "queries per split; records, identifiers, and topic instances are disjoint across splits."
        ),
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
            "require_cross_encoder_scores": True,
        },
    }
    validate_dataset(dataset)
    return dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build_dataset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "fixture": str(args.output), "schema_version": dataset["schema_version"],
        "records": len(dataset["records"]), "cases": len(dataset["cases"]),
        "calibration": sum(row["split"] == "calibration" for row in dataset["cases"]),
        "calibration_negative": sum(row["split"] == "calibration" and not row["answerable"] for row in dataset["cases"]),
        "holdout": sum(row["split"] == "holdout" for row in dataset["cases"]),
        "holdout_negative": sum(row["split"] == "holdout" and not row["answerable"] for row in dataset["cases"]),
        "holdout_policy": "unqueried until a feasible calibration policy exists",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

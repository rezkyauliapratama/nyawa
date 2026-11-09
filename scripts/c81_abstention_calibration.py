#!/usr/bin/env python3
"""Synthetic-only C8.1 abstention calibration with a frozen holdout split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any, TypeGuard

from c81_fixture_generator import SCHEMA, SCHEMA_C83, SCHEMA_C84, SCHEMA_C85, SCHEMA_C86, build_dataset, validate_dataset
from c83_fixture_generator import build_dataset as build_c83_dataset
from c84_fixture_generator import build_dataset as build_c84_dataset
from c85_fixture_generator import build_dataset as build_c85_dataset
from c86_fixture_generator import build_dataset as build_c86_dataset

IMAGE_C81 = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:1f79ea37a0ff0168c227fc4d682ac5d0a2caa570873fe29e54fb2734eaaee365"
)
IMAGE_C83 = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:70b2935609973314d85f587f71f75cb44bd9f0a587a85f9d0457fbd988593495"
)
IMAGE_C84 = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:bcbc858b5df17d734858eb5db7c1faa521a327576d8070e97bc248653eff5246"
)
IMAGE_C85 = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:57fb872aca5dd9bf24547d33d920787fb6fef73a8f6e63087f8e785fc27c2fea"
)
IMAGE_C86 = IMAGE_C85
APPROVED_IMAGES = {SCHEMA: IMAGE_C81, SCHEMA_C83: IMAGE_C83, SCHEMA_C84: IMAGE_C84, SCHEMA_C85: IMAGE_C85, SCHEMA_C86: IMAGE_C86}
IMAGE = IMAGE_C81
LABEL_KEY = "org.nyawa.abstention-calibration"
MEMORY_LIMIT = 4 * 1024**3
MIN_AVAILABLE_MEMORY = 5 * 1024**3
FIXTURE_DEFAULT = Path(__file__).resolve().parents[1] / "tests/fixtures/c81-abstention-calibration-fixtures.json"
SIGNALS = (
    "score", "rrf_score", "final_margin", "rrf_margin", "lexical_coverage", "lexical_margin",
    "vector_rank", "fts5_rank", "both_modalities", "exact_identifier_match",
)
RAW_SIGNALS = ("vector_distance", "fts5_rank_score", "vector_distance_margin", "fts5_rank_score_margin")
CROSS_ENCODER_SIGNALS = ("cross_encoder_score", "cross_encoder_margin")
LEXICAL_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or the to was what when where which who why with did does do this that these those find give tell me fictional synthetic fixture marker c8 c81 c83 c84 c85 c86 synth id dan yang apa berapa untuk dari pada ini itu adalah sebagai dengan bagaimana siapa kapan dimana mengapa antrean kebijakan mode berapa sebuah sebuahnya".split()
)
STRUCTURED_IDENTIFIER = re.compile(
    r"(?<![A-Z0-9])(?=[A-Z0-9._/-]*[A-Z])(?=[A-Z0-9._/-]*[0-9])[A-Z0-9]+(?:[-_./][A-Z0-9]+)+(?![A-Z0-9])",
    re.IGNORECASE,
)

HTTP_HELPER = r"""
import json, sys, urllib.error, urllib.request
method, path, raw = sys.argv[1:4]
data = None if raw == "-" else raw.encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:3300" + path, data=data, method=method)
if data is not None:
    req.add_header("Content-Type", "application/json")
try:
    with urllib.request.urlopen(req, timeout=30) as response:
        status, payload = response.status, response.read()
except urllib.error.HTTPError as error:
    status, payload = error.code, error.read()
try:
    body = json.loads(payload.decode("utf-8"))
except Exception:
    body = payload.decode("utf-8", "replace")
print(json.dumps({"status": status, "body": body}, separators=(",", ":")))
"""


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def _content(item: dict[str, Any]) -> str:
    return str(item.get("Content", item.get("content", "")))


def _numeric(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _both_modalities_present(candidate: dict[str, Any]) -> bool:
    vector_rank, fts5_rank = candidate.get("vector_rank"), candidate.get("fts5_rank")
    return _numeric(vector_rank) and _numeric(fts5_rank) and vector_rank > 0 and fts5_rank > 0


def _rank_within(candidate: dict[str, Any], signal: str, threshold: float | None) -> bool:
    value = candidate.get(signal)
    return threshold is not None and _numeric(value) and value > 0 and value <= threshold


def _field(item: dict[str, Any], upper: str, lower: str) -> Any:
    return item.get(upper, item.get(lower))


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in LEXICAL_STOPWORDS}


def _lexical_coverage(query: str, content: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(content)) / len(query_tokens)


def _structured_identifiers(text: str) -> set[str]:
    return {
        value.upper() for value in STRUCTURED_IDENTIFIER.findall(text)
        if not re.fullmatch(r"C[0-9]+\.[0-9]+", value, re.IGNORECASE)
    }


def _exact_identifier_match(query: str, content: str) -> bool:
    query_ids = _structured_identifiers(query)
    return bool(query_ids) and query_ids.issubset(_structured_identifiers(content))


def _margin(candidates: list[dict[str, Any]], field: str) -> float | None:
    values = [candidate[field] for candidate in candidates if _numeric(candidate.get(field))]
    if not values:
        return None
    values.sort(reverse=True)
    return float(values[0] - values[1]) if len(values) > 1 else float(values[0])


def _lower_better_margin(candidates: list[dict[str, Any]], field: str) -> float | None:
    values = [candidate[field] for candidate in candidates if _numeric(candidate.get(field))]
    if len(values) < 2:
        return None
    values.sort()
    return float(values[1] - values[0])


def score_case(
    case: dict[str, Any],
    items: list[dict[str, Any]],
    latency_ms: float,
    k: int,
    memory_to_fixture: dict[str, str],
    candidate_scorer: Any | None = None,
) -> dict[str, Any]:
    relevant = case["relevant_markers"]
    forbidden = case["forbidden_markers"]
    candidates: list[dict[str, Any]] = []
    chars = 0
    numeric = True
    modality_numeric_all = True
    raw_scores_numeric_all = True
    for rank, item in enumerate(items[:k], start=1):
        content = _content(item)
        score = _field(item, "Score", "score")
        rrf_score = _field(item, "RRFScore", "rrf_score")
        vector_rank = _field(item, "VectorRank", "vector_rank")
        fts5_rank = _field(item, "FTS5Rank", "fts5_rank")
        vector_distance = _field(item, "VectorDistance", "vector_distance")
        fts5_rank_score = _field(item, "FTS5RankScore", "fts5_rank_score")
        cross_encoder_score = _field(item, "CrossEncoderScore", "cross_encoder_score")
        numeric = numeric and _numeric(score) and _numeric(rrf_score)
        modality_numeric = (
            isinstance(vector_rank, int) and not isinstance(vector_rank, bool) and vector_rank >= 0
            and isinstance(fts5_rank, int) and not isinstance(fts5_rank, bool) and fts5_rank >= 0
        )
        raw_score_values_numeric = modality_numeric and (
            (vector_rank == 0 or _numeric(vector_distance))
            and (fts5_rank == 0 or _numeric(fts5_rank_score))
        )
        raw_scores_numeric_all = raw_scores_numeric_all and raw_score_values_numeric
        raw_score_available = modality_numeric and (vector_rank > 0 or fts5_rank > 0) and (
            (vector_rank == 0 or _numeric(vector_distance))
            and (fts5_rank == 0 or _numeric(fts5_rank_score))
        )
        modality_numeric_all = modality_numeric_all and modality_numeric
        item_id = str(_field(item, "ID", "id") or "")
        marker_hits = [marker for marker in relevant if marker in content]
        forbidden_hits = [marker for marker in forbidden if marker in content]
        candidate = {
            "fixture_id": memory_to_fixture.get(item_id, "unknown-fixture"),
            "rank": rank,
            "score": float(score) if _numeric(score) else None,
            "rrf_score": float(rrf_score) if _numeric(rrf_score) else None,
            "temporal_boost": float(item.get("TemporalBoost", 0.0)) if _numeric(item.get("TemporalBoost", 0.0)) else None,
            "importance_boost": float(item.get("ImportanceBoost", 0.0)) if _numeric(item.get("ImportanceBoost", 0.0)) else None,
            "vector_rank": int(vector_rank) if modality_numeric else None,
            "fts5_rank": int(fts5_rank) if modality_numeric else None,
            "vector_distance": float(vector_distance) if _numeric(vector_distance) else None,
            "fts5_rank_score": float(fts5_rank_score) if _numeric(fts5_rank_score) else None,
            "cross_encoder_score": float(cross_encoder_score) if _numeric(cross_encoder_score) else None,
            "cross_encoder_score_numeric": _numeric(cross_encoder_score),
            "modality_evidence_present": modality_numeric,
            "raw_modality_score_present": bool(raw_score_available),
            "raw_modality_scores_numeric": bool(raw_score_values_numeric),
            "both_modalities": bool(modality_numeric and vector_rank > 0 and fts5_rank > 0),
            "lexical_coverage": _lexical_coverage(case["query"], content),
            "exact_identifier_match": _exact_identifier_match(case["query"], content),
            "type": str(item.get("Type", item.get("type", ""))),
            "relevant": bool(marker_hits),
            "forbidden_markers_found": forbidden_hits,
        }
        candidates.append(candidate)
        chars += len(content)
    if candidate_scorer is not None:
        qwen_scores = candidate_scorer(case["query"], [_content(item) for item in items[:k]])
        if len(qwen_scores) != len(candidates):
            raise ValueError("candidate scorer returned a score count that does not match candidates")
        for candidate, value in zip(candidates, qwen_scores):
            candidate["qwen_answerability_score"] = float(value) if _numeric(value) else None
            candidate["qwen_answerability_score_numeric"] = _numeric(value)
    else:
        for candidate in candidates:
            candidate["qwen_answerability_score"] = None
            candidate["qwen_answerability_score_numeric"] = False
    relevant_candidates = [candidate for candidate in candidates if candidate["relevant"]]
    first_relevant_rank = relevant_candidates[0]["rank"] if relevant_candidates else None
    forbidden_hits = [marker for candidate in candidates for marker in candidate["forbidden_markers_found"]]
    scored_candidates = [
        candidate for candidate in candidates
        if (_numeric(candidate["vector_rank"]) and candidate["vector_rank"] > 0)
        or (_numeric(candidate["fts5_rank"]) and candidate["fts5_rank"] > 0)
    ]
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "subtype": case.get("subtype"),
        "split": case["split"],
        "answerable": case["answerable"],
        "namespace": case["namespace"],
        "relevant_count": len(relevant),
        "relevant_found": len(relevant_candidates),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": (1.0 / first_relevant_rank) if first_relevant_rank is not None else (0.0 if relevant else None),
        "result_count": len(candidates),
        "scores_numeric": numeric,
        "modality_ranks_numeric": modality_numeric_all,
        "modality_evidence_present": bool(candidates) and all(candidate["modality_evidence_present"] for candidate in candidates),
        "raw_modality_scores_available": bool(scored_candidates) and all(candidate["raw_modality_score_present"] for candidate in scored_candidates),
        "raw_modality_scores_numeric": bool(scored_candidates) and all(candidate["raw_modality_scores_numeric"] for candidate in scored_candidates),
        "cross_encoder_scores_numeric": bool(candidates) and all(candidate["cross_encoder_score_numeric"] for candidate in candidates),
        "qwen_answerability_scores_numeric": bool(candidates) and all(candidate["qwen_answerability_score_numeric"] for candidate in candidates),
        "fixture_ids_mapped": all(candidate["fixture_id"] != "unknown-fixture" for candidate in candidates),
        "latency_ms": round(float(latency_ms), 2),
        "candidate_chars": chars,
        "estimated_candidate_tokens": math.ceil(chars / 4),
        "final_margin": _margin(candidates, "score"),
        "rrf_margin": _margin(candidates, "rrf_score"),
        "lexical_margin": _margin(candidates, "lexical_coverage"),
        "vector_distance_margin": _lower_better_margin(candidates, "vector_distance"),
        "fts5_rank_score_margin": _lower_better_margin(candidates, "fts5_rank_score"),
        "cross_encoder_margin": _margin(candidates, "cross_encoder_score"),
        "forbidden_markers_found": forbidden_hits,
        "candidates": candidates,
    }


def evaluate_policy(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    signal: str | None = None,
    threshold: float | dict[str, float] | None = None,
) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in rows}
    positives = [case for case in cases if case["answerable"]]
    exact = [case for case in positives if case["category"] == "exact_identifier"]
    negatives = [case for case in cases if not case["answerable"]]
    correct_queries = exact_correct = false_positive_queries = 0
    answerable_coverage = abstentions = accepted_candidate_count = correct_candidate_count = 0
    forbidden_leaks = namespace_leaks = 0
    reciprocal_ranks: list[float] = []
    accepted_case_rows: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []

    for case in cases:
        row = by_id[case["case_id"]]
        candidates = row["candidates"]
        if signal in ("identifier_guard_cross_encoder", "qwen_identifier_guard"):
            accepted = []
            score_field = "qwen_answerability_score" if signal == "qwen_identifier_guard" else "cross_encoder_score"
            query_ids = _structured_identifiers(case.get("query", ""))
            for candidate in candidates:
                score = candidate.get(score_field)
                if threshold is None or isinstance(threshold, dict) or not _numeric(score) or score < threshold:
                    continue
                if query_ids and not candidate.get("exact_identifier_match", False):
                    continue
                accepted.append(candidate)
        elif signal == "qwen_answerability_score":
            accepted = [
                candidate for candidate in candidates
                if threshold is not None and not isinstance(threshold, dict)
                and _numeric(candidate.get(signal)) and candidate[signal] >= threshold
            ]
        elif signal == "dual_raw_cross_encoder":
            accepted = []
            if isinstance(threshold, dict):
                for candidate in candidates:
                    vector_distance, fts5_rank_score = candidate.get("vector_distance"), candidate.get("fts5_rank_score")
                    cross_score = candidate.get("cross_encoder_score")
                    if not (_numeric(vector_distance) and _numeric(fts5_rank_score) and _numeric(cross_score)):
                        continue
                    if vector_distance > threshold["vector_distance_max"] or fts5_rank_score > threshold["fts5_rank_score_max"] or cross_score < threshold["cross_encoder_min"]:
                        continue
                    accepted.append(candidate)
        elif signal in ("cross_encoder_score_rrf", "cross_encoder_score_lexical"):
            accepted = []
            if isinstance(threshold, dict):
                secondary = "rrf_score" if signal == "cross_encoder_score_rrf" else "lexical_coverage"
                for candidate in candidates:
                    cross_score, other_score = candidate.get("cross_encoder_score"), candidate.get(secondary)
                    if _numeric(cross_score) and _numeric(other_score) and cross_score >= threshold["cross_encoder_min"] and other_score >= threshold[f"{secondary}_min"]:
                        accepted.append(candidate)
        elif signal in ("dual_raw", "dual_raw_score", "dual_raw_rrf", "dual_raw_lexical"):
            accepted = []
            if isinstance(threshold, dict):
                for candidate in candidates:
                    vector_distance, fts5_rank_score = candidate.get("vector_distance"), candidate.get("fts5_rank_score")
                    if not (_numeric(vector_distance) and _numeric(fts5_rank_score)):
                        continue
                    if vector_distance > threshold["vector_distance_max"] or fts5_rank_score > threshold["fts5_rank_score_max"]:
                        continue
                    field = {"dual_raw_score": "score", "dual_raw_rrf": "rrf_score", "dual_raw_lexical": "lexical_coverage"}.get(signal)
                    if field is not None:
                        value, minimum = candidate.get(field), threshold.get(f"{field}_min")
                        if minimum is None or not _numeric(value) or value < minimum:
                            continue
                    accepted.append(candidate)
        elif signal in ("dual_rank", "dual_rank_score", "dual_rank_rrf", "dual_rank_lexical"):
            accepted = []
            if isinstance(threshold, dict):
                for candidate in candidates:
                    vector_rank, fts5_rank = candidate.get("vector_rank"), candidate.get("fts5_rank")
                    if not (_numeric(vector_rank) and _numeric(fts5_rank)):
                        continue
                    if vector_rank <= 0 or fts5_rank <= 0 or vector_rank > threshold["vector_max"] or fts5_rank > threshold["fts5_max"]:
                        continue
                    field = {"dual_rank_score": "score", "dual_rank_rrf": "rrf_score", "dual_rank_lexical": "lexical_coverage"}.get(signal)
                    if field is not None:
                        value, minimum = candidate.get(field), threshold.get(f"{field}_min")
                        if minimum is None or not _numeric(value) or value < minimum:
                            continue
                    accepted.append(candidate)
        elif signal in ("final_margin", "rrf_margin", "lexical_margin", "vector_distance_margin", "fts5_rank_score_margin", "cross_encoder_margin"):
            margin = row.get(signal)
            accepted = candidates if threshold is not None and not isinstance(threshold, dict) and _numeric(margin) and margin >= threshold else []
        elif signal in ("vector_distance", "fts5_rank_score"):
            accepted = [
                candidate for candidate in candidates
                if threshold is not None and not isinstance(threshold, dict)
                and _numeric(candidate.get(signal)) and candidate[signal] <= threshold
            ]
        elif signal == "both_modalities":
            accepted = [candidate for candidate in candidates if _both_modalities_present(candidate)]
        elif signal == "exact_identifier_match":
            accepted = [candidate for candidate in candidates if candidate.get("exact_identifier_match", False)]
        elif signal in ("vector_rank", "fts5_rank"):
            accepted = [candidate for candidate in candidates if not isinstance(threshold, dict) and _rank_within(candidate, signal, threshold)]
        elif signal is not None:
            accepted = [
                candidate for candidate in candidates
                if threshold is not None and not isinstance(threshold, dict) and _numeric(candidate.get(signal)) and candidate[signal] >= threshold
            ]
        else:
            accepted = candidates
        accepted_case_rows.append((case, accepted))
        if accepted:
            accepted_candidate_count += len(accepted)
        else:
            abstentions += 1
        if case["answerable"]:
            if accepted:
                answerable_coverage += 1
            hit = any(candidate["relevant"] for candidate in accepted)
            correct_queries += int(hit)
            first_hit = next((rank for rank, candidate in enumerate(accepted, start=1) if candidate["relevant"]), None)
            reciprocal_ranks.append(1.0 / first_hit if first_hit is not None else 0.0)
            if case["category"] == "exact_identifier":
                exact_correct += int(hit)
            correct_candidate_count += sum(bool(candidate["relevant"]) for candidate in accepted)
        elif accepted:
            false_positive_queries += 1
        forbidden_for_case = sum(bool(candidate["forbidden_markers_found"]) for candidate in accepted)
        forbidden_leaks += forbidden_for_case
        if case["category"] == "namespace_isolation" or case.get("subtype") == "scope_confusion":
            namespace_leaks += forbidden_for_case

    precision = correct_candidate_count / accepted_candidate_count if accepted_candidate_count else 0.0
    rows_with_candidates = [
        by_id[case["case_id"]]
        for case in cases
        if by_id[case["case_id"]].get("result_count", len(by_id[case["case_id"]].get("candidates", []))) > 0
    ]
    modality_evidence_available = bool(rows_with_candidates) and all(row.get("modality_evidence_present", False) for row in rows_with_candidates)
    modality_ranks_numeric = bool(rows_with_candidates) and all(row.get("modality_ranks_numeric", False) for row in rows_with_candidates)
    raw_modality_scores_available = bool(rows_with_candidates) and all(row.get("raw_modality_scores_available", False) for row in rows_with_candidates)
    raw_modality_scores_numeric = bool(rows_with_candidates) and all(row.get("raw_modality_scores_numeric", False) for row in rows_with_candidates)
    cross_encoder_scores_numeric = bool(rows_with_candidates) and all(row.get("cross_encoder_scores_numeric", False) for row in rows_with_candidates)
    qwen_answerability_scores_numeric = bool(rows_with_candidates) and all(row.get("qwen_answerability_scores_numeric", False) for row in rows_with_candidates)
    numeric_score_and_rrf_fields = bool(cases) and all(by_id[case["case_id"]].get("scores_numeric", False) for case in cases)
    fixture_ids_mapped = all(by_id[case["case_id"]].get("fixture_ids_mapped", True) for case in cases)
    metrics = {
        "positive_queries": len(positives),
        "negative_queries": len(negatives),
        "positive_recall_at_k": correct_queries / len(positives) if positives else 0.0,
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "exact_identifier_recall": exact_correct / len(exact) if exact else 0.0,
        "negative_false_positive_queries": false_positive_queries,
        "negative_false_positive_rate": false_positive_queries / len(negatives) if negatives else 0.0,
        "answerable_coverage": answerable_coverage / len(positives) if positives else 0.0,
        "abstentions": abstentions,
        "abstention_rate": abstentions / len(cases) if cases else 0.0,
        "accepted_candidate_count": accepted_candidate_count,
        "accepted_precision": precision,
        "namespace_leaks": namespace_leaks,
        "forbidden_marker_leaks": forbidden_leaks,
        "modality_evidence_available": modality_evidence_available,
        "modality_ranks_numeric": modality_ranks_numeric,
        "raw_modality_scores_available": raw_modality_scores_available,
        "raw_modality_scores_numeric": raw_modality_scores_numeric,
        "cross_encoder_scores_numeric": cross_encoder_scores_numeric,
        "qwen_answerability_scores_numeric": qwen_answerability_scores_numeric,
        "numeric_score_and_rrf_fields": numeric_score_and_rrf_fields,
        "fixture_ids_mapped": fixture_ids_mapped,
    }
    return {"metrics": metrics, "accepted_case_rows": accepted_case_rows}


def passes_gates(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return (
        metrics["positive_recall_at_k"] >= float(gates["positive_recall_at_k_min"])
        and metrics["exact_identifier_recall"] >= float(gates["exact_identifier_recall_min"])
        and metrics["negative_false_positive_rate"] <= float(gates["negative_false_positive_rate_max"])
        and metrics["namespace_leaks"] <= int(gates["namespace_leaks_max"])
        and metrics["forbidden_marker_leaks"] <= int(gates["forbidden_marker_leaks_max"])
        and (
            not gates.get("require_modality_evidence", False)
            or (metrics["modality_evidence_available"] and metrics["modality_ranks_numeric"])
        )
        and (
            not gates.get("require_raw_modality_scores", False)
            or (metrics["raw_modality_scores_available"] and metrics["raw_modality_scores_numeric"])
        )
        and (
            not gates.get("require_cross_encoder_scores", False)
            or metrics["cross_encoder_scores_numeric"]
        )
    )


def threshold_values(rows: list[dict[str, Any]], signal: str) -> list[float]:
    if signal == "identifier_guard_cross_encoder":
        signal = "cross_encoder_score"
    if signal == "qwen_identifier_guard":
        signal = "qwen_answerability_score"
    if signal == "both_modalities":
        return [1.0] if any(row.get("modality_evidence_present") for row in rows) else []
    if signal == "exact_identifier_match":
        return [1.0]
    values: set[float] = set()
    row_signals = ("final_margin", "rrf_margin", "lexical_margin", "vector_distance_margin", "fts5_rank_score_margin", "cross_encoder_margin")
    if signal in row_signals:
        for row in rows:
            value = row.get(signal)
            if _numeric(value):
                values.add(float(value))
    else:
        for row in rows:
            for candidate in row["candidates"]:
                value = candidate.get(signal)
                if _numeric(value):
                    values.add(float(value))
    if not values:
        return []
    ordered = sorted(values)
    if signal in ("vector_distance", "fts5_rank_score"):
        return [math.nextafter(ordered[0], -math.inf), *ordered]
    ordered.append(math.nextafter(ordered[-1], math.inf))
    return ordered


def calibrate_signal(
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    signal: str,
    require_qwen_scores: bool = False,
) -> dict[str, Any]:
    gates = dataset["gates"]
    points = []
    for threshold in threshold_values(rows, signal):
        metrics = evaluate_policy(cases, rows, signal, threshold)["metrics"]
        feasible = passes_gates(metrics, gates) and (
            not require_qwen_scores or (
                metrics.get("qwen_answerability_scores_numeric", False)
                and metrics.get("numeric_score_and_rrf_fields", False)
                and metrics.get("fixture_ids_mapped", False)
            )
        )
        point = {"threshold": threshold, **metrics, "feasible": feasible}
        points.append(point)
    feasible = [point for point in points if point["feasible"]]
    best = max(
        feasible,
        key=lambda point: (
            point["positive_recall_at_k"],
            point["accepted_precision"],
            -point["negative_false_positive_rate"],
            -point["threshold"],
        ),
        default=None,
    )
    best_any = max(
        points,
        key=lambda point: (
            point["positive_recall_at_k"],
            point["exact_identifier_recall"],
            -point["negative_false_positive_rate"],
            point["accepted_precision"],
        ),
        default=None,
    )
    return {
        "signal": signal,
        "thresholds_evaluated": len(points),
        "feasible_threshold_count": len(feasible),
        "best_feasible": best,
        "best_recall_tradeoff": best_any,
    }


def _sample_thresholds(values: set[float], limit: int = 10) -> list[float]:
    ordered = sorted(values)
    if len(ordered) <= limit:
        return ordered
    indexes = {round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)}
    return [ordered[i] for i in sorted(indexes)]


def calibrate_joint_signals(
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sweep two-source rank gates and their score/lexical conjunctions on calibration only."""
    value_sets: dict[str, set[float]] = {
        "vector_rank": set(), "fts5_rank": set(), "score": set(),
        "rrf_score": set(), "lexical_coverage": set(),
    }
    for row in rows:
        for candidate in row["candidates"]:
            for field, values in value_sets.items():
                value = candidate.get(field)
                if not _numeric(value) or (field in ("vector_rank", "fts5_rank") and value <= 0):
                    continue
                values.add(float(value))

    vector_values = value_sets["vector_rank"]
    fts_values = value_sets["fts5_rank"]
    score_values = value_sets["score"]
    rrf_values = value_sets["rrf_score"]
    lexical_values = value_sets["lexical_coverage"]
    rank_pairs = [(v, f) for v in _sample_thresholds(vector_values) for f in _sample_thresholds(fts_values)]
    rule_specs: list[tuple[str, list[tuple[dict[str, float], str | None]]]] = []
    base_params = [{"vector_max": v, "fts5_max": f} for v, f in rank_pairs]
    rule_specs.append(("dual_rank", [(params, None) for params in base_params]))
    for signal, field, values in (
        ("dual_rank_score", "score", score_values),
        ("dual_rank_rrf", "rrf_score", rrf_values),
        ("dual_rank_lexical", "lexical_coverage", lexical_values),
    ):
        sampled = _sample_thresholds(values)
        rule_specs.append((signal, [({**params, f"{field}_min": threshold}, field) for params in base_params for threshold in sampled]))

    results: list[dict[str, Any]] = []
    gates = dataset["gates"]
    for signal, specs in rule_specs:
        feasible: list[dict[str, Any]] = []
        points: list[dict[str, Any]] = []
        for params, _ in specs:
            metrics = evaluate_policy(cases, rows, signal, params)["metrics"]
            point = {"threshold": params, **metrics, "feasible": passes_gates(metrics, gates)}
            points.append(point)
            if point["feasible"]:
                feasible.append(point)
        best = max(
            feasible,
            key=lambda point: (
                point["positive_recall_at_k"], point["accepted_precision"],
                -point["negative_false_positive_rate"],
            ),
            default=None,
        )
        best_any = max(
            points,
            key=lambda point: (
                point["positive_recall_at_k"], point["exact_identifier_recall"],
                -point["negative_false_positive_rate"], point["accepted_precision"],
            ),
            default=None,
        )
        results.append({
            "signal": signal,
            "thresholds_evaluated": len(points),
            "feasible_threshold_count": len(feasible),
            "best_feasible": best,
            "best_recall_tradeoff": best_any,
        })
    return results


def calibrate_raw_joint_signals(
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sweep joint raw vector-distance and FTS5-rank thresholds on calibration only."""
    value_sets: dict[str, set[float]] = {
        "vector_distance": set(), "fts5_rank_score": set(), "score": set(),
        "rrf_score": set(), "lexical_coverage": set(),
    }
    for row in rows:
        for candidate in row["candidates"]:
            for field, values in value_sets.items():
                value = candidate.get(field)
                if _numeric(value):
                    values.add(float(value))
    vector_values = value_sets["vector_distance"]
    fts_values = value_sets["fts5_rank_score"]
    score_values = value_sets["score"]
    rrf_values = value_sets["rrf_score"]
    lexical_values = value_sets["lexical_coverage"]
    raw_pairs = [(v, f) for v in _sample_thresholds(vector_values) for f in _sample_thresholds(fts_values)]
    base_params = [{"vector_distance_max": v, "fts5_rank_score_max": f} for v, f in raw_pairs]
    rule_specs: list[tuple[str, list[dict[str, float]]]] = [("dual_raw", base_params)]
    for signal, field, values in (
        ("dual_raw_score", "score", score_values),
        ("dual_raw_rrf", "rrf_score", rrf_values),
        ("dual_raw_lexical", "lexical_coverage", lexical_values),
    ):
        rule_specs.append((signal, [
            {**params, f"{field}_min": minimum}
            for params in base_params for minimum in _sample_thresholds(values)
        ]))

    results: list[dict[str, Any]] = []
    gates = dataset["gates"]
    for signal, thresholds in rule_specs:
        points = []
        for params in thresholds:
            metrics = evaluate_policy(cases, rows, signal, params)["metrics"]
            points.append({"threshold": params, **metrics, "feasible": passes_gates(metrics, gates)})
        feasible = [point for point in points if point["feasible"]]
        results.append({
            "signal": signal,
            "thresholds_evaluated": len(points),
            "feasible_threshold_count": len(feasible),
            "best_feasible": max(feasible, key=lambda p: (p["positive_recall_at_k"], p["accepted_precision"], -p["negative_false_positive_rate"]), default=None),
            "best_recall_tradeoff": max(points, key=lambda p: (p["positive_recall_at_k"], p["exact_identifier_recall"], -p["negative_false_positive_rate"], p["accepted_precision"]), default=None),
        })
    return results


def calibrate_cross_encoder_joint_signals(
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sweep cross-encoder cutoffs and selected conjunctions on calibration only."""
    values: dict[str, set[float]] = {
        "cross_encoder_score": set(), "score": set(), "rrf_score": set(),
        "lexical_coverage": set(), "vector_distance": set(), "fts5_rank_score": set(),
    }
    for row in rows:
        for candidate in row["candidates"]:
            for field, found in values.items():
                value = candidate.get(field)
                if _numeric(value):
                    found.add(float(value))
    cross_thresholds = _sample_thresholds(values["cross_encoder_score"])
    rule_specs: list[tuple[str, list[dict[str, float]]]] = []
    for signal, field in (
        ("cross_encoder_score_rrf", "rrf_score"),
        ("cross_encoder_score_lexical", "lexical_coverage"),
    ):
        rule_specs.append((signal, [
            {"cross_encoder_min": cross, f"{field}_min": other}
            for cross in cross_thresholds for other in _sample_thresholds(values[field])
        ]))
    raw_pairs = [
        (distance, rank_score)
        for distance in _sample_thresholds(values["vector_distance"])
        for rank_score in _sample_thresholds(values["fts5_rank_score"])
    ]
    rule_specs.append(("dual_raw_cross_encoder", [
        {"vector_distance_max": distance, "fts5_rank_score_max": rank_score, "cross_encoder_min": cross}
        for distance, rank_score in raw_pairs for cross in cross_thresholds
    ]))

    results: list[dict[str, Any]] = []
    gates = dataset["gates"]
    for signal, thresholds in rule_specs:
        points = []
        for params in thresholds:
            metrics = evaluate_policy(cases, rows, signal, params)["metrics"]
            points.append({"threshold": params, **metrics, "feasible": passes_gates(metrics, gates)})
        feasible = [point for point in points if point["feasible"]]
        results.append({
            "signal": signal,
            "thresholds_evaluated": len(points),
            "feasible_threshold_count": len(feasible),
            "best_feasible": max(feasible, key=lambda p: (p["positive_recall_at_k"], p["accepted_precision"], -p["negative_false_positive_rate"]), default=None),
            "best_recall_tradeoff": max(points, key=lambda p: (p["positive_recall_at_k"], p["exact_identifier_recall"], -p["negative_false_positive_rate"], p["accepted_precision"]), default=None),
        })
    return results


def summarize(
    dataset: dict[str, Any],
    calibration_case_results: list[dict[str, Any]],
    holdout_case_results: list[dict[str, Any]] | None = None,
    qwen_only: bool = False,
) -> dict[str, Any]:
    cases = dataset["cases"]
    gates = dataset["gates"]
    calibration_cases = [case for case in cases if case["split"] == "calibration"]
    calibration_by_id = {row["case_id"]: row for row in calibration_case_results}
    if any(case["case_id"] not in calibration_by_id for case in calibration_cases):
        raise ValueError("calibration result set is incomplete")
    calibration_rows = [calibration_by_id[case["case_id"]] for case in calibration_cases]
    calibration_raw = evaluate_policy(calibration_cases, calibration_rows)["metrics"]
    if dataset["schema_version"] == SCHEMA_C84:
        signals = [*SIGNALS, *RAW_SIGNALS]
    elif dataset["schema_version"] in (SCHEMA_C85, SCHEMA_C86):
        signals = [*SIGNALS, *RAW_SIGNALS, *CROSS_ENCODER_SIGNALS]
    else:
        signals = list(SIGNALS)
    if qwen_only:
        sweeps = [
            calibrate_signal(dataset, calibration_cases, calibration_rows, "qwen_identifier_guard", require_qwen_scores=True)
        ]
    else:
        sweeps = [calibrate_signal(dataset, calibration_cases, calibration_rows, signal) for signal in signals]
    if qwen_only:
        joint_sweeps = []
    elif dataset["schema_version"] == SCHEMA_C83:
        joint_sweeps = calibrate_joint_signals(dataset, calibration_cases, calibration_rows)
    elif dataset["schema_version"] == SCHEMA_C84:
        joint_sweeps = (
            calibrate_joint_signals(dataset, calibration_cases, calibration_rows)
            + calibrate_raw_joint_signals(dataset, calibration_cases, calibration_rows)
        )
    elif dataset["schema_version"] == SCHEMA_C85:
        joint_sweeps = (
            calibrate_joint_signals(dataset, calibration_cases, calibration_rows)
            + calibrate_raw_joint_signals(dataset, calibration_cases, calibration_rows)
            + calibrate_cross_encoder_joint_signals(dataset, calibration_cases, calibration_rows)
        )
    elif dataset["schema_version"] == SCHEMA_C86:
        joint_sweeps = (
            calibrate_joint_signals(dataset, calibration_cases, calibration_rows)
            + calibrate_raw_joint_signals(dataset, calibration_cases, calibration_rows)
            + calibrate_cross_encoder_joint_signals(dataset, calibration_cases, calibration_rows)
            + [calibrate_signal(dataset, calibration_cases, calibration_rows, "identifier_guard_cross_encoder")]
        )
    else:
        joint_sweeps = []
    feasible_candidates = [
        {"signal": sweep["signal"], **sweep["best_feasible"]}
        for sweep in sweeps if sweep["best_feasible"] is not None
    ]
    feasible_candidates.extend(
        {"signal": sweep["signal"], **sweep["best_feasible"]}
        for sweep in joint_sweeps if sweep["best_feasible"] is not None
    )
    selected = max(
        feasible_candidates,
        key=lambda point: (
            point["positive_recall_at_k"],
            point["accepted_precision"],
            -point["negative_false_positive_rate"],
            point["signal"],
        ),
        default=None,
    )

    holdout_policy_metrics = None
    holdout_pass = None
    holdout_evaluated = False
    observed_results = list(calibration_case_results)
    if holdout_case_results is not None:
        if selected is None:
            raise ValueError("refusing to inspect holdout without a feasible calibration policy")
        holdout_cases = [case for case in cases if case["split"] == "holdout"]
        holdout_by_id = {row["case_id"]: row for row in holdout_case_results}
        if any(case["case_id"] not in holdout_by_id for case in holdout_cases):
            raise ValueError("holdout result set is incomplete")
        holdout_rows = [holdout_by_id[case["case_id"]] for case in holdout_cases]
        holdout_policy_metrics = evaluate_policy(
            holdout_cases, holdout_rows, selected["signal"], selected["threshold"]
        )["metrics"]
        holdout_pass = passes_gates(holdout_policy_metrics, gates) and (
            not qwen_only or holdout_policy_metrics.get("qwen_answerability_scores_numeric", False)
        )
        holdout_evaluated = True
        observed_results.extend(holdout_case_results)

    all_numeric = all(row["scores_numeric"] for row in observed_results)
    fixture_ids_mapped = all(row.get("fixture_ids_mapped", True) for row in observed_results)
    returned_candidates = [candidate for row in observed_results for candidate in row["candidates"]]
    modality_evidence_available = bool(returned_candidates) and all(candidate.get("modality_evidence_present", False) for candidate in returned_candidates)
    candidate_rows = [row for row in observed_results if row.get("result_count", len(row.get("candidates", []))) > 0]
    modality_ranks_numeric = bool(candidate_rows) and all(row.get("modality_ranks_numeric", False) for row in candidate_rows)
    raw_modality_scores_available = bool(candidate_rows) and all(row.get("raw_modality_scores_available", False) for row in candidate_rows)
    raw_modality_scores_numeric = bool(candidate_rows) and all(row.get("raw_modality_scores_numeric", False) for row in candidate_rows)
    cross_encoder_scores_numeric = bool(candidate_rows) and all(row.get("cross_encoder_scores_numeric", False) for row in candidate_rows)
    qwen_answerability_scores_numeric = bool(candidate_rows) and all(row.get("qwen_answerability_scores_numeric", False) for row in candidate_rows)
    source_counts = {"vector_only": 0, "fts5_only": 0, "both": 0, "neither": 0}
    if modality_evidence_available:
        for candidate in returned_candidates:
            vector_hit = candidate["vector_rank"] > 0
            fts5_hit = candidate["fts5_rank"] > 0
            key = "both" if vector_hit and fts5_hit else "vector_only" if vector_hit else "fts5_only" if fts5_hit else "neither"
            source_counts[key] += 1
    status = "PASS" if (
        selected and holdout_evaluated and holdout_pass and all_numeric and fixture_ids_mapped
        and (not qwen_only or qwen_answerability_scores_numeric)
    ) else "FAIL"
    if dataset["schema_version"] == SCHEMA_C85:
        protocol = "local cross-encoder calibration on C8.4's unopened holdout; holdout queried only after a feasible calibration policy"
    elif dataset["schema_version"] == SCHEMA_C86:
        protocol = "fresh identifier-aware C8.6 calibration; holdout queried only after a feasible calibration policy"
    elif dataset["schema_version"] == SCHEMA_C84:
        protocol = "calibration-only selection; holdout queried only after a feasible policy exists"
    else:
        protocol = "calibration-only univariate and joint threshold selection; holdout queried only after selection"
    latencies = sorted(row["latency_ms"] for row in observed_results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0
    return {
        "status": status,
        "schema_version": dataset["schema_version"],
        "protocol": protocol,
        "gates": gates,
        "raw_retrieval": {"calibration": calibration_raw},
        "threshold_sweeps": sweeps,
        "joint_threshold_sweeps": joint_sweeps,
        "selected_policy": selected,
        "holdout_evaluated": holdout_evaluated,
        "holdout_policy_metrics": holdout_policy_metrics,
        "holdout_gates_pass": holdout_pass,
        "numeric_score_and_rrf_fields": all_numeric,
        "fixture_ids_mapped": fixture_ids_mapped,
        "modality_evidence_available": modality_evidence_available,
        "modality_ranks_numeric": modality_ranks_numeric,
        "raw_modality_scores_available": raw_modality_scores_available,
        "raw_modality_scores_numeric": raw_modality_scores_numeric,
        "cross_encoder_scores_numeric": cross_encoder_scores_numeric,
        "qwen_answerability_scores_numeric": qwen_answerability_scores_numeric,
        "source_counts": source_counts,
        "latency_median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
        "candidate_chars_total": sum(row["candidate_chars"] for row in observed_results),
        "estimated_candidate_tokens_total": sum(row["estimated_candidate_tokens"] for row in observed_results),
        "case_results": observed_results,
        "provider_called": False,
        "hermes_called": False,
        "context_injected": False,
    }


def run_calibration_phases(
    dataset: dict[str, Any],
    retrieve_case: Any,
    qwen_only: bool = False,
    before_holdout: Any | None = None,
) -> dict[str, Any]:
    """Query holdout only after the selected calibration policy passes its gates."""
    calibration_cases = [case for case in dataset["cases"] if case["split"] == "calibration"]
    calibration_results = [retrieve_case(case) for case in calibration_cases]
    calibration_report = summarize(dataset, calibration_results, qwen_only=qwen_only)
    selected = calibration_report["selected_policy"]
    if selected is None:
        return calibration_report
    if qwen_only and (
        selected.get("signal") != "qwen_identifier_guard"
        or not selected.get("feasible", False)
        or not selected.get("qwen_answerability_scores_numeric", False)
        or not passes_gates(selected, dataset["gates"])
    ):
        raise RuntimeError("refusing holdout access without a feasible numeric Qwen-only policy")

    if before_holdout is not None:
        before_holdout()
    holdout_cases = [case for case in dataset["cases"] if case["split"] == "holdout"]
    holdout_results = [retrieve_case(case) for case in holdout_cases]
    final_report = summarize(dataset, calibration_results, holdout_results, qwen_only=qwen_only)
    if final_report["selected_policy"] != selected:
        raise RuntimeError("calibration-selected policy changed after holdout evaluation")
    return final_report


def preflight() -> None:
    meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
    available_kib = next(int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
    available = available_kib * 1024
    if available < MIN_AVAILABLE_MEMORY:
        raise RuntimeError(f"runner preflight failed: MemAvailable={available}; need {MIN_AVAILABLE_MEMORY}; no container started")
    cpus = os.cpu_count()
    if cpus is None or cpus < 2:
        raise RuntimeError("runner preflight failed: at least 2 CPUs required; no container started")
    stat = os.statvfs(os.environ.get("RUNNER_TEMP", "/tmp"))
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 4 * 1024**3:
        raise RuntimeError("runner preflight failed: less than 4 GiB free disk; no container started")
    print(json.dumps({"preflight": "PASS", "mem_available_bytes": available, "cpu_count": cpus, "runner_temp_free_bytes": free_bytes}, sort_keys=True))


def docker(args: list[str], timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["docker", *args], text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed ({result.returncode}): {(result.stderr or result.stdout).strip()[-1000:]}")
    return result


def inside_http(container: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = "-" if payload is None else json.dumps(payload, separators=(",", ":"))
    result = docker(["exec", container, "python3", "-c", HTTP_HELPER, method, path, raw], timeout=45)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Nyawa helper returned invalid JSON: {result.stdout[:300]!r}") from error


def verify_container(container: str, image: str) -> None:
    inspected = json.loads(docker(["inspect", container]).stdout)[0]
    host = inspected["HostConfig"]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if inspected["Config"].get("Image") != image or labels.get(LABEL_KEY) != "true":
        raise RuntimeError("abstention calibration container image/ownership mismatch")
    if host.get("NetworkMode") != "none" or not host.get("ReadonlyRootfs"):
        raise RuntimeError("calibration container is not network=none/read-only")
    if host.get("PortBindings") or host.get("Binds") or host.get("Mounts"):
        raise RuntimeError("calibration container unexpectedly has a port or host/volume mount")
    if host.get("Memory") != MEMORY_LIMIT or host.get("MemorySwap") != MEMORY_LIMIT or host.get("PidsLimit") != 128:
        raise RuntimeError("calibration container resource limits do not match the declared bounds")
    if any(mount.get("Type") != "tmpfs" for mount in inspected.get("Mounts", [])):
        raise RuntimeError("calibration container has a non-tmpfs mount")


def cleanup(container: str, image: str) -> None:
    result = docker(["inspect", container], timeout=15, check=False)
    if result.returncode:
        return
    inspected = json.loads(result.stdout)[0]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if labels.get(LABEL_KEY) != "true" or inspected["Config"].get("Image") != image:
        raise RuntimeError("refusing to remove a container not created by this calibration run")
    docker(["rm", "--force", container], timeout=30)
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("calibration cleanup readback failed: container still exists")


def run_e2e(
    container: str,
    image: str,
    fixture_path: Path,
    candidate_scorer: Any | None = None,
    qwen_only: bool = False,
    preflight_passed: bool = False,
) -> dict[str, Any]:
    dataset = load_dataset(fixture_path)
    if qwen_only and (dataset["schema_version"] != SCHEMA_C86 or candidate_scorer is None):
        raise ValueError("Qwen-only calibration requires C8.6 fixtures and a local candidate scorer")
    expected_image = APPROVED_IMAGES.get(dataset["schema_version"])
    if expected_image is None or image != expected_image:
        raise RuntimeError("no matching immutable image digest is approved for this fixture schema")
    if not preflight_passed:
        preflight()
    docker(["pull", image], timeout=300)
    image_info = json.loads(docker(["image", "inspect", image]).stdout)[0]
    if image not in image_info.get("RepoDigests", []):
        raise RuntimeError("pulled image digest does not match the pinned image")
    if dataset["schema_version"] in (SCHEMA_C85, SCHEMA_C86):
        labels = image_info.get("Config", {}).get("Labels") or {}
        if labels.get("io.nyawa.cross-encoder.enabled") != "true":
            raise RuntimeError("pinned cross-encoder image does not declare the local scorer enabled")
        expected_model = "Horizon-Labs/multilingual-reranker-small@23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398"
        if labels.get("io.nyawa.cross-encoder.model") != expected_model:
            raise RuntimeError("pinned cross-encoder image model label does not match the approved small-model revision")
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("container name already exists; refusing to replace it")
    stage = {SCHEMA: "c81", SCHEMA_C83: "c83", SCHEMA_C84: "c84", SCHEMA_C85: "c85", SCHEMA_C86: "c86"}[dataset["schema_version"]]
    db_path = f"/scratch/{stage}-abstention.db"
    container_env = [
        "--env", "MEMORY_EMBEDDER=local",
        "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model",
        "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
    ]
    if dataset["schema_version"] in (SCHEMA_C85, SCHEMA_C86):
        container_env.extend([
            "--env", "NYAWA_ENABLE_CROSS_ENCODER=1",
            "--env", "NYAWA_RERANKER_MODEL_DIR=/opt/nyawa/model/reranker",
        ])
    try:
        docker([
            "run", "--detach", "--name", container,
            "--label", f"{LABEL_KEY}=true",
            "--network=none", "--read-only",
            "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
            "--restart=no", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            *container_env,
            image, "serve", db_path,
        ], timeout=60)
        verify_container(container, image)
        deadline = time.monotonic() + 180
        root = health = None
        last_error = "not ready"
        while time.monotonic() < deadline:
            state = json.loads(docker(["inspect", "--format", "{{json .State}}", container]).stdout)
            if not state.get("Running"):
                logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
                raise RuntimeError(f"Nyawa exited before readiness ({state.get('ExitCode')}): {logs}")
            try:
                root = inside_http(container, "GET", "/")
                health = inside_http(container, "GET", "/v1/health")
                if root["status"] == 200 and health["status"] == 200:
                    break
                last_error = f"root={root['status']} health={health['status']}"
            except Exception as error:
                last_error = str(error)
            time.sleep(2)
        else:
            logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
            raise RuntimeError(f"Nyawa readiness timeout: {last_error}; {logs}")
        root_body = root["body"]
        if root_body.get("version") != "1.2.4" or root_body.get("status") != "running":
            raise RuntimeError(f"unexpected Nyawa identity: {root_body}")

        inserted: dict[str, str] = {}
        reverse_ids: dict[str, str] = {}

        def insert_records(records: list[dict[str, Any]]) -> None:
            for record in records:
                response = inside_http(container, "POST", "/v1/memories", {
                    "content": record["content"], "namespace": record["namespace"], "type": record["type"]
                })
                if response["status"] != 201 or not response["body"].get("id"):
                    raise RuntimeError(f"synthetic insert failed for {record['fixture_id']}: HTTP {response['status']}")
                memory_id = str(response["body"]["id"])
                inserted[record["fixture_id"]] = memory_id
                reverse_ids[memory_id] = record["fixture_id"]

        def vector_index_count() -> tuple[int, dict[str, Any]]:
            stats_response = inside_http(container, "GET", "/v1/stats")
            if stats_response["status"] != 200:
                raise RuntimeError(f"stats returned HTTP {stats_response['status']}")
            store_stats = stats_response["body"].get("store", stats_response["body"])
            return int(store_stats.get("vector_indexed", 0)), store_stats

        initial_records = [
            record for record in dataset["records"]
            if not qwen_only or record["split"] == "calibration"
        ]
        insert_records(initial_records)
        vector_indexed, store_stats = vector_index_count()
        if vector_indexed < len(inserted):
            raise RuntimeError(f"not all inserted synthetic records were vector-indexed: {store_stats}")

        def insert_holdout_records() -> None:
            nonlocal vector_indexed
            holdout_records = [record for record in dataset["records"] if record["split"] == "holdout"]
            insert_records(holdout_records)
            vector_indexed, stats = vector_index_count()
            if vector_indexed < len(inserted):
                raise RuntimeError(f"not all synthetic records were vector-indexed: {stats}")

        def retrieve_case(case: dict[str, Any]) -> dict[str, Any]:
            start = time.monotonic()
            response = inside_http(container, "POST", "/v1/recall", {
                "query": case["query"],
                "namespace": case["namespace"],
                "limit": int(dataset["k"]),
                "min_score": 0.0,
                "exclude_types": case.get("exclude_types", []),
            })
            elapsed = (time.monotonic() - start) * 1000
            if response["status"] != 200:
                raise RuntimeError(f"recall failed for {case['case_id']}: HTTP {response['status']}")
            items = response["body"].get("results", [])
            if not isinstance(items, list):
                raise RuntimeError(f"recall results malformed for {case['case_id']}")
            return score_case(case, items, elapsed, int(dataset["k"]), reverse_ids, candidate_scorer)

        report = run_calibration_phases(
            dataset, retrieve_case, qwen_only=qwen_only,
            before_holdout=insert_holdout_records if qwen_only else None,
        )
        report.update({
            "image": image,
            "runtime_version": root_body["version"],
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "synthetic_records_inserted": len(inserted),
            "vector_indexed": vector_indexed,
            "queries_run": len(report["case_results"]),
            "result_limit_k": int(dataset["k"]),
        })
        return report
    finally:
        cleanup(container, image)


class C81CalibrationTests(unittest.TestCase):
    def test_generated_fixture_counts_and_split_disjointness(self) -> None:
        dataset = build_dataset()
        validate_dataset(dataset)
        self.assertEqual(len(dataset["records"]), 90)
        self.assertEqual(len(dataset["cases"]), 120)
        self.assertEqual(sum(case["answerable"] for case in dataset["cases"]), 60)
        self.assertEqual(sum(not case["answerable"] for case in dataset["cases"]), 60)
        calibration = {case["query"] for case in dataset["cases"] if case["split"] == "calibration"}
        holdout = {case["query"] for case in dataset["cases"] if case["split"] == "holdout"}
        self.assertTrue(calibration.isdisjoint(holdout))
        calibration_markers = {marker for case in dataset["cases"] if case["split"] == "calibration" for marker in case["relevant_markers"]}
        holdout_markers = {marker for case in dataset["cases"] if case["split"] == "holdout" for marker in case["relevant_markers"]}
        self.assertTrue(calibration_markers.isdisjoint(holdout_markers))

    def test_c83_corpus_is_fresh_and_split_disjoint(self) -> None:
        c81 = build_dataset()
        c83 = build_c83_dataset()
        validate_dataset(c83)
        self.assertEqual(c83["schema_version"], SCHEMA_C83)
        self.assertEqual(len(c83["records"]), 90)
        self.assertEqual(len(c83["cases"]), 120)
        c81_queries = {case["query"] for case in c81["cases"]}
        c83_queries = {case["query"] for case in c83["cases"]}
        self.assertTrue(c81_queries.isdisjoint(c83_queries))
        marker_pattern = re.compile(r"Marker ([A-Z0-9_]+)")
        c81_markers = {marker for record in c81["records"] for marker in marker_pattern.findall(record["content"])}
        c83_markers = {marker for record in c83["records"] for marker in marker_pattern.findall(record["content"])}
        self.assertTrue(c81_markers.isdisjoint(c83_markers))
        for split in ("calibration", "holdout"):
            self.assertEqual(sum(case["split"] == split for case in c83["cases"]), 60)

    def test_c84_runtime_image_pin_is_immutable(self) -> None:
        self.assertEqual(
            IMAGE_C84,
            "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@sha256:bcbc858b5df17d734858eb5db7c1faa521a327576d8070e97bc248653eff5246",
        )
        self.assertEqual(APPROVED_IMAGES[SCHEMA_C84], IMAGE_C84)

    def test_c84_corpus_is_fresh_and_split_disjoint(self) -> None:
        old_corpora = [build_dataset(), build_c83_dataset()]
        c84 = build_c84_dataset()
        validate_dataset(c84)
        self.assertEqual(c84["schema_version"], SCHEMA_C84)
        self.assertEqual(len(c84["records"]), 90)
        self.assertEqual(len(c84["cases"]), 120)
        c84_queries = {case["query"] for case in c84["cases"]}
        marker_pattern = re.compile(r"Marker ([A-Z0-9_]+)")
        c84_markers = {marker for record in c84["records"] for marker in marker_pattern.findall(record["content"])}
        for old in old_corpora:
            self.assertTrue(c84_queries.isdisjoint({case["query"] for case in old["cases"]}))
            old_markers = {marker for record in old["records"] for marker in marker_pattern.findall(record["content"])}
            self.assertTrue(c84_markers.isdisjoint(old_markers))
        for split in ("calibration", "holdout"):
            split_cases = [case for case in c84["cases"] if case["split"] == split]
            self.assertEqual(len(split_cases), 60)
            self.assertEqual(sum(case["answerable"] for case in split_cases), 30)

    def test_score_case_records_both_score_signals_without_content(self) -> None:
        case = {"case_id": "Q", "category": "exact_identifier", "split": "calibration", "answerable": True, "namespace": "c8-synthetic-main", "query": "Find exact identifier SYNTH-C81-ID-0001", "relevant_markers": ["GOLD"], "forbidden_markers": []}
        result = score_case(case, [{"ID": "m1", "Content": "Synthetic C8.1 record GOLD SYNTH-C81-ID-0001", "Score": 0.9, "RRFScore": 0.6, "Rank": 1, "VectorRank": 2, "FTS5Rank": 1}], 12, 5, {"m1": "F01"})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["fixture_id"], "F01")
        self.assertEqual(candidate["score"], 0.9)
        self.assertEqual(candidate["rrf_score"], 0.6)
        self.assertTrue(candidate["relevant"])
        self.assertTrue(candidate["modality_evidence_present"])
        self.assertEqual((candidate["vector_rank"], candidate["fts5_rank"]), (2, 1))
        self.assertTrue(candidate["both_modalities"])
        self.assertTrue(candidate["exact_identifier_match"])
        c83_case = {**case, "query": "Retrieve exact identifier SYNTH-C83-ID-0001"}
        c83_result = score_case(c83_case, [{"ID": "m3", "Content": "GOLD SYNTH-C83-ID-0001", "Score": 0.8, "RRFScore": 0.5, "VectorRank": 1, "FTS5Rank": 2}], 5, 5, {"m3": "F03"})
        self.assertTrue(c83_result["candidates"][0]["exact_identifier_match"])
        self.assertGreater(candidate["lexical_coverage"], 0.0)
        self.assertNotIn("content", candidate)
        self.assertTrue(result["scores_numeric"])
        self.assertTrue(result["fixture_ids_mapped"])
        unknown = score_case(case, [{"ID": "m2", "Content": "other", "Score": 0.1, "RRFScore": 0.2}], 1, 5, {})
        self.assertFalse(unknown["fixture_ids_mapped"])
        self.assertFalse(unknown["candidates"][0]["modality_evidence_present"])
        legacy = score_case(case, [{"ID": "m1", "Content": "GOLD", "Score": 0.8, "RRFScore": 0.5}], 1, 5, {"m1": "F01"})
        self.assertFalse(legacy["modality_evidence_present"])
        self.assertFalse(legacy["modality_ranks_numeric"])

    def test_modality_agreement_can_abstain_on_single_source_candidate(self) -> None:
        cases = [
            {"case_id": "P", "answerable": True, "category": "semantic_paraphrase"},
            {"case_id": "N", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P", "candidates": [{"score": 0.8, "rrf_score": 0.7, "vector_rank": 2, "fts5_rank": 1, "relevant": True, "forbidden_markers_found": []}]},
            {"case_id": "N", "candidates": [{"score": 1.0, "rrf_score": 0.8, "vector_rank": 1, "fts5_rank": 0, "relevant": False, "forbidden_markers_found": []}]},
        ]
        result = evaluate_policy(cases, rows, "both_modalities", 1.0)["metrics"]
        self.assertEqual(result["positive_recall_at_k"], 1.0)
        self.assertEqual(result["negative_false_positive_rate"], 0.0)

    def test_raw_modality_scores_are_numeric_lower_is_better_features(self) -> None:
        case = {
            "case_id": "Q84", "category": "exact_identifier", "split": "calibration",
            "answerable": True, "namespace": "c84-synthetic-main",
            "query": "Resolve exact identifier SYNTH-C84-ID-0001",
            "relevant_markers": ["GOLD"], "forbidden_markers": [],
        }
        row = score_case(case, [
            {"ID": "m1", "Content": "GOLD SYNTH-C84-ID-0001", "Score": 0.8, "RRFScore": 0.7,
             "VectorRank": 1, "FTS5Rank": 1, "VectorDistance": 0.0, "FTS5RankScore": 0.0},
            {"ID": "m2", "Content": "noise", "Score": 0.4, "RRFScore": 0.3,
             "VectorRank": 2, "FTS5Rank": 2, "VectorDistance": 0.8, "FTS5RankScore": 0.5},
        ], 2.0, 5, {"m1": "F1", "m2": "F2"})
        self.assertTrue(row["raw_modality_scores_available"])
        self.assertTrue(row["raw_modality_scores_numeric"])
        self.assertEqual(row["candidates"][0]["vector_distance"], 0.0)
        self.assertEqual(row["candidates"][0]["fts5_rank_score"], 0.0)
        self.assertEqual(row["vector_distance_margin"], 0.8)
        self.assertEqual(row["fts5_rank_score_margin"], 0.5)
        cases = [case]
        self.assertEqual(evaluate_policy(cases, [row], "vector_distance", 0.01)["metrics"]["positive_recall_at_k"], 1.0)
        self.assertEqual(evaluate_policy(cases, [row], "fts5_rank_score", 0.01)["metrics"]["positive_recall_at_k"], 1.0)

    def test_joint_rank_threshold_abstains_on_single_source_candidate(self) -> None:
        dataset = {"gates": {
            "positive_recall_at_k_min": 0.9,
            "exact_identifier_recall_min": 1.0,
            "negative_false_positive_rate_max": 0.05,
            "namespace_leaks_max": 0,
            "forbidden_marker_leaks_max": 0,
            "require_modality_evidence": True,
        }}
        cases = [
            {"case_id": "P", "answerable": True, "category": "exact_identifier"},
            {"case_id": "N", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P", "result_count": 1, "scores_numeric": True, "modality_evidence_present": True, "modality_ranks_numeric": True,
             "candidates": [{"score": 0.9, "rrf_score": 0.8, "vector_rank": 1, "fts5_rank": 2, "lexical_coverage": 1.0, "relevant": True, "forbidden_markers_found": []}]},
            {"case_id": "N", "result_count": 1, "scores_numeric": True, "modality_evidence_present": True, "modality_ranks_numeric": True,
             "candidates": [{"score": 0.9, "rrf_score": 0.8, "vector_rank": 1, "fts5_rank": 0, "lexical_coverage": 0.0, "relevant": False, "forbidden_markers_found": []}]},
        ]
        joint = calibrate_joint_signals(dataset, cases, rows)
        dual_rank = next(policy for policy in joint if policy["signal"] == "dual_rank")
        self.assertEqual(dual_rank["feasible_threshold_count"], 1)
        self.assertEqual(dual_rank["best_feasible"]["positive_recall_at_k"], 1.0)
        self.assertEqual(dual_rank["best_feasible"]["negative_false_positive_rate"], 0.0)

    def test_modality_evidence_gate_fails_closed_when_fields_are_missing(self) -> None:
        gates = {
            "positive_recall_at_k_min": 0.9,
            "exact_identifier_recall_min": 1.0,
            "negative_false_positive_rate_max": 0.05,
            "namespace_leaks_max": 0,
            "forbidden_marker_leaks_max": 0,
            "require_modality_evidence": True,
        }
        metrics = {
            "positive_recall_at_k": 1.0,
            "exact_identifier_recall": 1.0,
            "negative_false_positive_rate": 0.0,
            "namespace_leaks": 0,
            "forbidden_marker_leaks": 0,
            "modality_evidence_available": False,
            "modality_ranks_numeric": False,
        }
        self.assertFalse(passes_gates(metrics, gates))

    def test_raw_modality_score_gate_fails_closed_when_missing(self) -> None:
        gates = {
            "positive_recall_at_k_min": 0.9,
            "exact_identifier_recall_min": 1.0,
            "negative_false_positive_rate_max": 0.05,
            "namespace_leaks_max": 0,
            "forbidden_marker_leaks_max": 0,
            "require_raw_modality_scores": True,
        }
        metrics = {
            "positive_recall_at_k": 1.0, "exact_identifier_recall": 1.0,
            "negative_false_positive_rate": 0.0, "namespace_leaks": 0,
            "forbidden_marker_leaks": 0, "raw_modality_scores_available": False,
            "raw_modality_scores_numeric": False,
        }
        self.assertFalse(passes_gates(metrics, gates))

    def test_policy_metrics_and_zero_fpr_gate(self) -> None:
        cases = [
            {"case_id": "P", "answerable": True, "category": "exact_identifier"},
            {"case_id": "N", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P", "candidates": [{"score": 0.9, "rrf_score": 0.7, "relevant": True, "forbidden_markers_found": []}]},
            {"case_id": "N", "candidates": [{"score": 0.8, "rrf_score": 0.6, "relevant": False, "forbidden_markers_found": []}]},
        ]
        result = evaluate_policy(cases, rows, "score", 0.85)["metrics"]
        self.assertEqual(result["positive_recall_at_k"], 1.0)
        self.assertEqual(result["negative_false_positive_rate"], 0.0)
        self.assertEqual(result["exact_identifier_recall"], 1.0)

    def test_threshold_selection_uses_calibration_only(self) -> None:
        dataset = {"gates": {"positive_recall_at_k_min": 0.9, "exact_identifier_recall_min": 1.0, "negative_false_positive_rate_max": 0.05, "namespace_leaks_max": 0, "forbidden_marker_leaks_max": 0}}
        cases = [
            {"case_id": "P1", "answerable": True, "category": "exact_identifier"},
            {"case_id": "P2", "answerable": True, "category": "semantic_paraphrase"},
            {"case_id": "N1", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P1", "candidates": [{"score": 0.9, "rrf_score": 0.7, "relevant": True, "forbidden_markers_found": []}], "final_margin": 0.9, "rrf_margin": 0.7},
            {"case_id": "P2", "candidates": [{"score": 0.8, "rrf_score": 0.6, "relevant": True, "forbidden_markers_found": []}], "final_margin": 0.8, "rrf_margin": 0.6},
            {"case_id": "N1", "candidates": [{"score": 0.7, "rrf_score": 0.5, "relevant": False, "forbidden_markers_found": []}], "final_margin": 0.7, "rrf_margin": 0.5},
        ]
        selected = calibrate_signal(dataset, cases, rows, "score")
        self.assertIsNotNone(selected["best_feasible"])
        self.assertEqual(selected["best_feasible"]["threshold"], 0.8)

    def test_infeasible_calibration_does_not_query_holdout(self) -> None:
        dataset = build_dataset()
        calls: list[str] = []

        def retrieve(case: dict[str, Any]) -> dict[str, Any]:
            calls.append(case["case_id"])
            return {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_evidence_present": False, "modality_ranks_numeric": False,
                "result_count": 0, "candidates": [], "fixture_ids_mapped": True,
                "latency_ms": 1.0, "candidate_chars": 0, "estimated_candidate_tokens": 0,
            }

        report = run_calibration_phases(dataset, retrieve)
        calibration_ids = {case["case_id"] for case in dataset["cases"] if case["split"] == "calibration"}
        holdout_ids = {case["case_id"] for case in dataset["cases"] if case["split"] == "holdout"}
        self.assertEqual(set(calls), calibration_ids)
        self.assertTrue(holdout_ids.isdisjoint(calls))
        self.assertFalse(report["holdout_evaluated"])
        self.assertIsNone(report["holdout_gates_pass"])
        self.assertNotIn("holdout", report["raw_retrieval"])
        self.assertEqual(len(report["case_results"]), len(calibration_ids))

    def test_c83_summary_selects_calibration_joint_rule_and_evaluates_holdout(self) -> None:
        dataset = build_c83_dataset()
        rows = []
        for case in dataset["cases"]:
            candidates = []
            if case["answerable"]:
                candidates = [{
                    "fixture_id": f"TARGET_{case['case_id']}", "rank": 1, "score": 0.9,
                    "rrf_score": 0.8, "temporal_boost": 0.0, "importance_boost": 0.0,
                    "vector_rank": 1, "fts5_rank": 1, "modality_evidence_present": True,
                    "both_modalities": True, "exact_identifier_match": True, "lexical_coverage": 0.8,
                    "type": "fact", "relevant": True, "forbidden_markers_found": [],
                }]
            else:
                candidates = [{
                    "fixture_id": f"NOISE_{case['case_id']}", "rank": 1, "score": 0.9,
                    "rrf_score": 0.8, "temporal_boost": 0.0, "importance_boost": 0.0,
                    "vector_rank": 1, "fts5_rank": 0, "modality_evidence_present": True,
                    "both_modalities": False, "exact_identifier_match": False, "lexical_coverage": 0.2,
                    "type": "fact", "relevant": False, "forbidden_markers_found": [],
                }]
            rows.append({
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_ranks_numeric": True, "modality_evidence_present": True,
                "result_count": 1, "candidates": candidates,
                "final_margin": 0.9, "rrf_margin": 0.8, "lexical_margin": 0.6,
                "latency_ms": 1.0, "candidate_chars": 10, "estimated_candidate_tokens": 3,
                "fixture_ids_mapped": True,
            })
        case_splits = {case["case_id"]: case["split"] for case in dataset["cases"]}
        calibration_rows = [row for row in rows if case_splits[row["case_id"]] == "calibration"]
        holdout_rows = [row for row in rows if case_splits[row["case_id"]] == "holdout"]
        report = summarize(dataset, calibration_rows, holdout_rows)
        self.assertEqual(report["schema_version"], SCHEMA_C83)
        self.assertTrue(report["joint_threshold_sweeps"])
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["holdout_evaluated"])
        self.assertTrue(report["holdout_gates_pass"])
        self.assertIsNotNone(report["selected_policy"])
        mutated_holdout_rows = [
            {**row, "candidates": [{**candidate} for candidate in row["candidates"]]}
            for row in holdout_rows
        ]
        for row in mutated_holdout_rows:
            for candidate in row["candidates"]:
                candidate["vector_rank"] = 50
                candidate["fts5_rank"] = 50
                candidate["score"] = 0.01
        after_holdout_mutation = summarize(dataset, calibration_rows, mutated_holdout_rows)
        self.assertEqual(report["joint_threshold_sweeps"], after_holdout_mutation["joint_threshold_sweeps"])
        self.assertEqual(report["selected_policy"], after_holdout_mutation["selected_policy"])

    def test_c84_raw_joint_rule_selects_and_evaluates_fresh_holdout(self) -> None:
        dataset = build_c84_dataset()
        calibration_rows, holdout_rows = [], []
        for case in dataset["cases"]:
            if case["answerable"]:
                candidate = {
                    "fixture_id": f"TARGET_{case['case_id']}", "rank": 1,
                    "score": 0.9, "rrf_score": 0.8, "vector_rank": 1, "fts5_rank": 1,
                    "vector_distance": 0.1, "fts5_rank_score": -0.5,
                    "modality_evidence_present": True, "raw_modality_score_present": True,
                    "raw_modality_scores_numeric": True, "both_modalities": True,
                    "exact_identifier_match": True, "lexical_coverage": 0.8,
                    "type": "fact", "relevant": True, "forbidden_markers_found": [],
                }
            else:
                candidate = {
                    "fixture_id": f"NOISE_{case['case_id']}", "rank": 1,
                    "score": 0.9, "rrf_score": 0.8, "vector_rank": 1, "fts5_rank": 0,
                    "vector_distance": 0.1, "fts5_rank_score": None,
                    "modality_evidence_present": True, "raw_modality_score_present": True,
                    "raw_modality_scores_numeric": True, "both_modalities": False,
                    "exact_identifier_match": False, "lexical_coverage": 0.2,
                    "type": "fact", "relevant": False, "forbidden_markers_found": [],
                }
            row = {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_ranks_numeric": True, "modality_evidence_present": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "result_count": 1, "candidates": [candidate],
                "final_margin": 0.9, "rrf_margin": 0.8, "lexical_margin": 0.6,
                "vector_distance_margin": 0.5, "fts5_rank_score_margin": 0.4,
                "latency_ms": 1.0, "candidate_chars": 10, "estimated_candidate_tokens": 3,
                "fixture_ids_mapped": True,
            }
            (calibration_rows if case["split"] == "calibration" else holdout_rows).append(row)
        report = summarize(dataset, calibration_rows, holdout_rows)
        raw_rule = next(row for row in report["joint_threshold_sweeps"] if row["signal"] == "dual_raw")
        self.assertGreater(raw_rule["feasible_threshold_count"], 0)
        self.assertTrue(report["holdout_evaluated"])
        self.assertTrue(report["holdout_gates_pass"])
        self.assertTrue(report["raw_modality_scores_available"])
        self.assertTrue(report["raw_modality_scores_numeric"])

    def test_c85_runtime_image_pin_is_immutable(self) -> None:
        self.assertEqual(
            IMAGE_C85,
            "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@sha256:57fb872aca5dd9bf24547d33d920787fb6fef73a8f6e63087f8e785fc27c2fea",
        )
        self.assertEqual(APPROVED_IMAGES[SCHEMA_C85], IMAGE_C85)

    def test_c85_rekeys_only_the_unopened_c84_fixture_for_cross_encoder(self) -> None:
        c84, c85 = build_c84_dataset(), build_c85_dataset()
        validate_dataset(c85)
        self.assertEqual(c85["schema_version"], SCHEMA_C85)
        self.assertTrue(c85["gates"]["require_cross_encoder_scores"])
        self.assertEqual(len(c85["records"]), 90)
        self.assertEqual(len(c85["cases"]), 120)
        self.assertTrue(all(record["namespace"].startswith("c85-synthetic-") for record in c85["records"]))
        self.assertTrue(all("Synthetic C8.5 fixture." in record["content"] for record in c85["records"]))
        self.assertTrue(all("C84" not in case["case_id"] for case in c85["cases"]))
        self.assertEqual(
            sum(case["split"] == "holdout" for case in c84["cases"]),
            sum(case["split"] == "holdout" for case in c85["cases"]),
        )

    def test_c85_cross_encoder_policy_requires_numeric_scores_and_can_gate_holdout(self) -> None:
        dataset = build_c85_dataset()
        calibration_rows, holdout_rows = [], []
        for case in dataset["cases"]:
            score = 3.0 if case["answerable"] else -3.0
            candidate = {
                "fixture_id": f"C85_{case['case_id']}", "rank": 1,
                "score": 0.8, "rrf_score": 0.7, "vector_rank": 1, "fts5_rank": 1,
                "vector_distance": 0.2, "fts5_rank_score": -0.4,
                "cross_encoder_score": score, "cross_encoder_score_numeric": True,
                "modality_evidence_present": True, "raw_modality_score_present": True,
                "raw_modality_scores_numeric": True, "both_modalities": True,
                "exact_identifier_match": case["category"] == "exact_identifier",
                "lexical_coverage": 0.8 if case["answerable"] else 0.2,
                "type": "fact", "relevant": case["answerable"], "forbidden_markers_found": [],
            }
            row = {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_ranks_numeric": True, "modality_evidence_present": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "cross_encoder_scores_numeric": True, "result_count": 1, "candidates": [candidate],
                "final_margin": 0.1, "rrf_margin": 0.1, "lexical_margin": 0.1,
                "vector_distance_margin": 0.1, "fts5_rank_score_margin": 0.1,
                "cross_encoder_margin": 6.0, "latency_ms": 1.0,
                "candidate_chars": 10, "estimated_candidate_tokens": 3,
                "fixture_ids_mapped": True,
            }
            (calibration_rows if case["split"] == "calibration" else holdout_rows).append(row)
        report = summarize(dataset, calibration_rows, holdout_rows)
        self.assertIsNotNone(report["selected_policy"])
        self.assertTrue(report["holdout_evaluated"])
        self.assertTrue(report["holdout_gates_pass"])
        self.assertTrue(report["cross_encoder_scores_numeric"])

    def test_c86_runtime_image_pin_uses_approved_r5_digest(self) -> None:
        self.assertEqual(IMAGE_C86, IMAGE_C85)
        self.assertEqual(APPROVED_IMAGES[SCHEMA_C86], IMAGE_C86)

    def test_structured_identifier_parser_and_exact_match_are_format_agnostic(self) -> None:
        identifiers = ("ERR-2026-1001", "DOC_A1002", "TKT/2026/1003", "POL.26.1004", "INV-BE-1005")
        for identifier in identifiers:
            with self.subTest(identifier=identifier):
                self.assertEqual(_structured_identifiers(f"Look up {identifier}"), {identifier})
                self.assertTrue(_exact_identifier_match(f"Look up {identifier}", f"Record {identifier.lower()} is present"))
                self.assertFalse(_exact_identifier_match(f"Look up {identifier}", f"Record {identifier[:-1]}9 is present"))
        self.assertFalse(_structured_identifiers("How many items were logged in 2026?"))
        self.assertFalse(_structured_identifiers("Read the C8.6 calibration notes."))

    def test_c86_fixture_is_fresh_larger_and_split_isolated(self) -> None:
        dataset = build_c86_dataset()
        validate_dataset(dataset)
        self.assertEqual(dataset["schema_version"], SCHEMA_C86)
        self.assertEqual(len(dataset["records"]), 164)
        self.assertEqual(len(dataset["cases"]), 300)
        old = [build_dataset(), build_c83_dataset(), build_c84_dataset(), build_c85_dataset()]
        queries = {case["query"] for case in dataset["cases"]}
        markers = {match for record in dataset["records"] for match in re.findall(r"Marker ([A-Z0-9_]+)", record["content"])}
        for prior in old:
            self.assertTrue(queries.isdisjoint({case["query"] for case in prior["cases"]}))
            prior_markers = {match for record in prior["records"] for match in re.findall(r"Marker ([A-Z0-9_]+)", record["content"])}
            self.assertTrue(markers.isdisjoint(prior_markers))
        calibration_records = {record["fixture_id"] for record in dataset["records"] if record["split"] == "calibration"}
        holdout_records = {record["fixture_id"] for record in dataset["records"] if record["split"] == "holdout"}
        self.assertTrue(calibration_records.isdisjoint(holdout_records))
        calibration_ids = {token for case in dataset["cases"] if case["split"] == "calibration" for token in _structured_identifiers(case["query"])}
        holdout_ids = {token for case in dataset["cases"] if case["split"] == "holdout" for token in _structured_identifiers(case["query"])}
        self.assertTrue(calibration_ids.isdisjoint(holdout_ids))
        calibration_content = {record["content"] for record in dataset["records"] if record["split"] == "calibration"}
        holdout_content = {record["content"] for record in dataset["records"] if record["split"] == "holdout"}
        self.assertTrue(calibration_content.isdisjoint(holdout_content))
        for split in ("calibration", "holdout"):
            split_cases = [case for case in dataset["cases"] if case["split"] == split]
            self.assertEqual(sum(case["answerable"] for case in split_cases), 50)
            self.assertEqual(sum(not case["answerable"] for case in split_cases), 100)

    def test_identifier_guard_calibrates_on_calibration_only(self) -> None:
        dataset = build_c86_dataset()
        calibration_cases = [case for case in dataset["cases"] if case["split"] == "calibration"]
        content_by_marker = {
            marker: record["content"]
            for record in dataset["records"]
            for marker in re.findall(r"Marker ([A-Z0-9_]+)", record["content"])
        }
        rows = []
        for case in calibration_cases:
            query_ids = _structured_identifiers(case["query"])
            if case["answerable"]:
                content = content_by_marker[case["relevant_markers"][0]]
                exact = _exact_identifier_match(case["query"], content)
                score, relevant = 2.0, True
            else:
                exact = False
                score, relevant = (4.0, False) if query_ids else (0.5, False)
            candidate = {
                "score": 0.9, "rrf_score": 0.8, "cross_encoder_score": score,
                "vector_rank": 1, "fts5_rank": 1, "vector_distance": 0.1,
                "fts5_rank_score": -0.5, "modality_evidence_present": True,
                "raw_modality_score_present": True, "raw_modality_scores_numeric": True,
                "both_modalities": True, "exact_identifier_match": exact,
                "lexical_coverage": 0.7, "type": "fact", "relevant": relevant,
                "forbidden_markers_found": [],
            }
            rows.append({
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_ranks_numeric": True, "modality_evidence_present": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "cross_encoder_scores_numeric": True, "result_count": 1, "candidates": [candidate],
                "final_margin": 0.1, "rrf_margin": 0.1, "lexical_margin": 0.1,
                "vector_distance_margin": 0.1, "fts5_rank_score_margin": 0.1,
                "cross_encoder_margin": score, "latency_ms": 1.0,
                "candidate_chars": 24, "estimated_candidate_tokens": 5,
                "fixture_ids_mapped": True,
            })
        sweep = calibrate_signal(dataset, calibration_cases, rows, "identifier_guard_cross_encoder")
        self.assertIsNotNone(sweep["best_feasible"])
        self.assertTrue(passes_gates(sweep["best_feasible"], dataset["gates"]))
        report = summarize(dataset, rows)
        self.assertIsNotNone(report["selected_policy"])
        self.assertFalse(report["holdout_evaluated"])

    def test_identifier_guard_filters_near_ids_but_keeps_natural_language_policy(self) -> None:
        cases = [
            {"case_id": "P-ID", "query": "Find ERR-2026-1001", "answerable": True, "category": "exact_identifier"},
            {"case_id": "P-NL", "query": "What mode does the ember relay use?", "answerable": True, "category": "semantic_paraphrase"},
            {"case_id": "N-ID", "query": "Resolve ERR-2026-1021", "answerable": False, "category": "negative_query"},
            {"case_id": "N-NL", "query": "Who approved the orbital warranty?", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P-ID", "candidates": [{"cross_encoder_score": 2.0, "qwen_answerability_score": 2.0, "exact_identifier_match": True, "relevant": True, "forbidden_markers_found": []}]},
            {"case_id": "P-NL", "candidates": [{"cross_encoder_score": 2.0, "qwen_answerability_score": 2.0, "exact_identifier_match": False, "relevant": True, "forbidden_markers_found": []}]},
            {"case_id": "N-ID", "candidates": [{"cross_encoder_score": 3.0, "qwen_answerability_score": 3.0, "exact_identifier_match": False, "relevant": False, "forbidden_markers_found": []}]},
            {"case_id": "N-NL", "candidates": [{"cross_encoder_score": 3.0, "qwen_answerability_score": 3.0, "exact_identifier_match": False, "relevant": False, "forbidden_markers_found": []}]},
        ]
        for signal in ("identifier_guard_cross_encoder", "qwen_identifier_guard"):
            with self.subTest(signal=signal):
                accepted = dict((case["case_id"], items) for case, items in evaluate_policy(cases, rows, signal, 1.5)["accepted_case_rows"])
                self.assertEqual(len(accepted["P-ID"]), 1)
                self.assertEqual(len(accepted["P-NL"]), 1)
                self.assertEqual(len(accepted["N-ID"]), 0)
                self.assertEqual(len(accepted["N-NL"]), 1)

    def test_c85_cross_encoder_gate_fails_closed_when_missing(self) -> None:
        gates = {
            "positive_recall_at_k_min": 0.9, "exact_identifier_recall_min": 1.0,
            "negative_false_positive_rate_max": 0.05, "namespace_leaks_max": 0,
            "forbidden_marker_leaks_max": 0, "require_cross_encoder_scores": True,
        }
        metrics = {
            "positive_recall_at_k": 1.0, "exact_identifier_recall": 1.0,
            "negative_false_positive_rate": 0.0, "namespace_leaks": 0,
            "forbidden_marker_leaks": 0, "cross_encoder_scores_numeric": False,
        }
        self.assertFalse(passes_gates(metrics, gates))

    def test_qwen_score_callback_is_numeric_and_preserves_raw_evidence(self) -> None:
        case = {
            "case_id": "Q86", "category": "exact_identifier", "split": "calibration",
            "answerable": True, "namespace": "c86-synthetic-main",
            "query": "Find ERR-2026-1001", "relevant_markers": ["GOLD"],
            "forbidden_markers": [],
        }
        received: list[tuple[str, list[str]]] = []

        def fake_scorer(query: str, documents: list[str]) -> list[float]:
            received.append((query, documents))
            return [1.75]

        row = score_case(case, [{
            "ID": "m1", "Content": "GOLD exact reference ERR-2026-1001",
            "Score": 0.8, "RRFScore": 0.7, "VectorRank": 1, "FTS5Rank": 1,
            "VectorDistance": 0.1, "FTS5RankScore": -0.4,
            "CrossEncoderScore": 0.5,
        }], 1.0, 5, {"m1": "F1"}, fake_scorer)
        self.assertEqual(received, [(case["query"], ["GOLD exact reference ERR-2026-1001"])])
        self.assertEqual(row["candidates"][0]["qwen_answerability_score"], 1.75)
        self.assertTrue(row["qwen_answerability_scores_numeric"])
        self.assertTrue(row["raw_modality_scores_numeric"])
        self.assertTrue(row["candidates"][0]["exact_identifier_match"])
        self.assertNotIn("content", row["candidates"][0])

    def test_qwen_infeasible_calibration_does_not_insert_or_query_holdout(self) -> None:
        dataset = build_c86_dataset()
        events: list[str] = []
        calibration_rows: list[dict[str, Any]] = []

        def retrieve(case: dict[str, Any]) -> dict[str, Any]:
            events.append(case["split"])
            positive = bool(case["answerable"])
            cross_score = 2.0 if positive else -2.0
            candidate = {
                "score": 0.8, "rrf_score": 0.7, "cross_encoder_score": cross_score,
                "cross_encoder_score_numeric": True,
                "qwen_answerability_score": None,
                "qwen_answerability_score_numeric": False,
                "vector_rank": 1, "fts5_rank": 1, "vector_distance": 0.1,
                "fts5_rank_score": -0.4, "modality_evidence_present": True,
                "raw_modality_score_present": True, "raw_modality_scores_numeric": True,
                "exact_identifier_match": positive and case["category"] == "exact_identifier",
                "relevant": positive, "forbidden_markers_found": [],
            }
            row = {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_evidence_present": True, "modality_ranks_numeric": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "cross_encoder_scores_numeric": True, "qwen_answerability_scores_numeric": False,
                "fixture_ids_mapped": True, "result_count": 1, "candidates": [candidate],
                "latency_ms": 1.0, "candidate_chars": 24, "estimated_candidate_tokens": 6,
            }
            if case["split"] == "calibration":
                calibration_rows.append(row)
            return row

        def before_holdout() -> None:
            events.append("holdout-insert")

        calibration_cases = [case for case in dataset["cases"] if case["split"] == "calibration"]
        legacy_only = None
        report = run_calibration_phases(
            dataset, retrieve, qwen_only=True, before_holdout=before_holdout,
        )
        legacy_only = calibrate_signal(dataset, calibration_cases, calibration_rows, "cross_encoder_score")
        self.assertIsNotNone(legacy_only["best_feasible"])
        self.assertIsNone(report["selected_policy"])
        self.assertFalse(report["holdout_evaluated"])
        self.assertEqual(events, ["calibration"] * 150)
        self.assertNotIn("holdout-insert", events)
        self.assertFalse(report["qwen_answerability_scores_numeric"])

    def test_qwen_feasible_policy_inserts_then_queries_holdout_once(self) -> None:
        dataset = build_c86_dataset()
        events: list[str] = []

        def retrieve(case: dict[str, Any]) -> dict[str, Any]:
            events.append(case["split"])
            positive = bool(case["answerable"])
            candidate = {
                "score": 0.8, "rrf_score": 0.7, "cross_encoder_score": 0.5,
                "cross_encoder_score_numeric": True,
                "qwen_answerability_score": 2.0 if positive else -2.0,
                "qwen_answerability_score_numeric": True,
                "vector_rank": 1, "fts5_rank": 1, "vector_distance": 0.1,
                "fts5_rank_score": -0.4, "modality_evidence_present": True,
                "raw_modality_score_present": True, "raw_modality_scores_numeric": True,
                "exact_identifier_match": positive and case["category"] == "exact_identifier",
                "relevant": positive, "forbidden_markers_found": [],
            }
            return {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_evidence_present": True, "modality_ranks_numeric": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "cross_encoder_scores_numeric": True, "qwen_answerability_scores_numeric": True,
                "fixture_ids_mapped": True, "result_count": 1, "candidates": [candidate],
                "final_margin": 0.1, "rrf_margin": 0.1, "lexical_margin": 0.1,
                "vector_distance_margin": 0.1, "fts5_rank_score_margin": 0.1,
                "cross_encoder_margin": 0.1, "latency_ms": 1.0,
                "candidate_chars": 24, "estimated_candidate_tokens": 6,
            }

        def before_holdout() -> None:
            events.append("holdout-insert")

        report = run_calibration_phases(
            dataset, retrieve, qwen_only=True, before_holdout=before_holdout,
        )
        self.assertIsNotNone(report["selected_policy"])
        self.assertEqual(report["selected_policy"]["signal"], "qwen_identifier_guard")
        self.assertEqual([sweep["signal"] for sweep in report["threshold_sweeps"]], ["qwen_identifier_guard"])
        self.assertEqual(events[150], "holdout-insert")
        self.assertEqual(events[:150], ["calibration"] * 150)
        self.assertEqual(events[151:], ["holdout"] * 150)
        self.assertTrue(report["holdout_evaluated"])
        self.assertTrue(report["holdout_gates_pass"])
        self.assertTrue(report["qwen_answerability_scores_numeric"])

    def test_qwen_nonnumeric_holdout_fails_closed_after_single_gated_query(self) -> None:
        dataset = build_c86_dataset()
        events: list[str] = []

        def retrieve(case: dict[str, Any]) -> dict[str, Any]:
            events.append(case["split"])
            positive = bool(case["answerable"])
            qwen_score = (2.0 if positive else -2.0) if case["split"] == "calibration" else None
            exact = positive and case["category"] == "exact_identifier"
            candidate = {
                "score": 0.8, "rrf_score": 0.7, "cross_encoder_score": 0.5,
                "cross_encoder_score_numeric": True,
                "qwen_answerability_score": qwen_score,
                "qwen_answerability_score_numeric": _numeric(qwen_score),
                "vector_rank": 1, "fts5_rank": 1, "vector_distance": 0.1,
                "fts5_rank_score": -0.4, "modality_evidence_present": True,
                "raw_modality_score_present": True, "raw_modality_scores_numeric": True,
                "exact_identifier_match": exact, "relevant": positive,
                "forbidden_markers_found": [],
            }
            return {
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_evidence_present": True, "modality_ranks_numeric": True,
                "raw_modality_scores_available": True, "raw_modality_scores_numeric": True,
                "cross_encoder_scores_numeric": True,
                "qwen_answerability_scores_numeric": qwen_score is not None,
                "fixture_ids_mapped": True, "result_count": 1, "candidates": [candidate],
                "final_margin": 0.1, "rrf_margin": 0.1, "lexical_margin": 0.1,
                "vector_distance_margin": 0.1, "fts5_rank_score_margin": 0.1,
                "cross_encoder_margin": 0.1, "latency_ms": 1.0,
                "candidate_chars": 24, "estimated_candidate_tokens": 6,
            }

        def before_holdout() -> None:
            events.append("holdout-insert")

        report = run_calibration_phases(
            dataset, retrieve, qwen_only=True, before_holdout=before_holdout,
        )
        self.assertEqual(report["selected_policy"]["signal"], "qwen_identifier_guard")
        self.assertTrue(report["holdout_evaluated"])
        self.assertFalse(report["holdout_gates_pass"])
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["qwen_answerability_scores_numeric"])
        self.assertEqual(events[:150], ["calibration"] * 150)
        self.assertEqual(events[150], "holdout-insert")
        self.assertEqual(events[151:], ["holdout"] * 150)

    def test_frozen_holdout_passes_only_after_calibration_selected_rule(self) -> None:
        dataset = build_dataset()
        rows = []
        for case in dataset["cases"]:
            candidates = []
            if case["answerable"]:
                candidates = [{
                    "fixture_id": f"TARGET_{case['case_id']}", "rank": 1, "score": 0.9,
                    "rrf_score": 0.8, "temporal_boost": 0.0, "importance_boost": 0.0,
                    "vector_rank": 1, "fts5_rank": 1, "modality_evidence_present": True,
                    "both_modalities": True, "exact_identifier_match": False, "lexical_coverage": 0.75,
                    "type": "fact", "relevant": True, "forbidden_markers_found": [],
                }]
            rows.append({
                "case_id": case["case_id"], "scores_numeric": True,
                "modality_ranks_numeric": True, "modality_evidence_present": bool(candidates),
                "result_count": len(candidates), "candidates": candidates,
                "final_margin": 0.9 if candidates else None,
                "rrf_margin": 0.8 if candidates else None, "latency_ms": 1.0,
                "candidate_chars": 10, "estimated_candidate_tokens": 3,
            })
        case_splits = {case["case_id"]: case["split"] for case in dataset["cases"]}
        calibration_rows = [row for row in rows if case_splits[row["case_id"]] == "calibration"]
        holdout_rows = [row for row in rows if case_splits[row["case_id"]] == "holdout"]
        report = summarize(dataset, calibration_rows, holdout_rows)
        self.assertEqual(report["status"], "PASS")
        self.assertIsNotNone(report["selected_policy"])
        self.assertTrue(report["holdout_evaluated"])
        self.assertTrue(report["holdout_gates_pass"])

    def test_no_threshold_that_hits_gates_is_not_selected(self) -> None:
        dataset = {"gates": {"positive_recall_at_k_min": 0.9, "exact_identifier_recall_min": 1.0, "negative_false_positive_rate_max": 0.05, "namespace_leaks_max": 0, "forbidden_marker_leaks_max": 0}}
        cases = [
            {"case_id": "P", "answerable": True, "category": "exact_identifier"},
            {"case_id": "N", "answerable": False, "category": "negative_query"},
        ]
        rows = [
            {"case_id": "P", "candidates": [{"score": 0.8, "rrf_score": 0.8, "relevant": True, "forbidden_markers_found": []}], "final_margin": 0.8, "rrf_margin": 0.8},
            {"case_id": "N", "candidates": [{"score": 0.9, "rrf_score": 0.9, "relevant": False, "forbidden_markers_found": []}], "final_margin": 0.9, "rrf_margin": 0.9},
        ]
        for signal in SIGNALS:
            result = calibrate_signal(dataset, cases, rows, signal)
            self.assertIsNone(result["best_feasible"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "self-test", "e2e"), required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--container", default="nyawa-abstention-calibration")
    parser.add_argument("--fixtures", type=Path, default=FIXTURE_DEFAULT)
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
        return 0
    if args.mode == "self-test":
        try:
            dataset = load_dataset(args.fixtures) if args.fixtures.exists() else build_dataset()
            print(json.dumps({"fixture_validation": "PASS", "records": len(dataset["records"]), "cases": len(dataset["cases"])}))
        except Exception as error:
            print(f"synthetic fixture validation FAILED: {error}", file=sys.stderr)
            return 1
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C81CalibrationTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        report = run_e2e(args.container, args.image, args.fixtures)
        print(json.dumps(report, indent=2, sort_keys=True))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            stage = {SCHEMA: "C8.1", SCHEMA_C83: "C8.3", SCHEMA_C84: "C8.4", SCHEMA_C85: "C8.5", SCHEMA_C86: "C8.6"}[report["schema_version"]]
            summary_report = report
            if report["schema_version"] == SCHEMA_C86:
                summary_keys = (
                    "status", "schema_version", "protocol", "gates", "raw_retrieval",
                    "threshold_sweeps", "joint_threshold_sweeps", "selected_policy",
                    "holdout_evaluated", "holdout_policy_metrics", "holdout_gates_pass",
                    "numeric_score_and_rrf_fields", "fixture_ids_mapped",
                    "modality_evidence_available", "modality_ranks_numeric",
                    "raw_modality_scores_available", "raw_modality_scores_numeric",
                    "cross_encoder_scores_numeric", "qwen_answerability_scores_numeric",
                    "source_counts", "latency_median_ms",
                    "latency_p95_ms", "candidate_chars_total", "estimated_candidate_tokens_total",
                    "queries_run", "synthetic_records_inserted", "vector_indexed", "result_limit_k",
                )
                summary_report = {key: report[key] for key in summary_keys if key in report}
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write(f"## {stage} Abstention Calibration\n\n```json\n")
                stream.write(json.dumps(summary_report, indent=2, sort_keys=True))
                stream.write("\n```\n\nSynthetic-only; calibration selects policy, holdout is evaluated only after a feasible calibration policy. No provider/Hermes call or context injection.\n")
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        print(f"abstention evaluation FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

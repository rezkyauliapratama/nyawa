#!/usr/bin/env python3
"""Run the isolated C8.7 synthetic-development evaluation against local Nyawa."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from typing import Any

from c81_abstention_calibration import HTTP_HELPER, cleanup, docker, inside_http, preflight, verify_container
from c87_fixture_generator import (
    SCHEMA as SCHEMA_C87,
    SCHEMA_V2 as SCHEMA_C87_V2,
    build_dataset,
    build_development_dataset_v2,
    validate_development_dataset_v2,
)

IMAGE_C87 = "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@sha256:bcbc858b5df17d734858eb5db7c1faa521a327576d8070e97bc248653eff5246"
LABEL_KEY = "org.nyawa.abstention-calibration"
POLICY_KS = (1, 3, 5, 12)
TYPE_MAP = {"artifact": "reference", "constraint": "procedure", "decision": "decision", "fact": "fact"}
# Raw forbidden/superseded exposures remain diagnostics; namespace leaks retain a raw hard gate.
MIN_GATES = {
    "recall_at_5": 0.90,
    "exact_identifier_recall_at_5": 1.0,
    "active_decision_recall_at_5": 0.90,
    "hard_constraint_recall_at_5": 1.0,
    "supersession_recall_at_5": 0.90,
    "namespace_isolation_recall_at_5": 0.90,
}
MAX_GATES = {
    "negative_false_positive_rate_at_k": 0.0,
    "raw_namespace_leak_count": 0,
    "selected_namespace_leak_count": 0,
    "selected_forbidden_candidate_exposure_count": 0,
    "superseded_selected_candidate_count": 0,
    "raw_unmapped_candidate_count": 0,
}


def field(mapping: dict[str, Any], name: str, default: Any = None) -> Any:
    """Read API fields without depending on JSON capitalization."""
    if name in mapping:
        return mapping[name]
    wanted = name.casefold()
    return next((value for key, value in mapping.items() if str(key).casefold() == wanted), default)


def load_dataset(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != SCHEMA_C87_V2:
        raise RuntimeError(f"unexpected schema: {dataset.get('schema_version')!r}")
    validate_development_dataset_v2(dataset)
    return dataset


def supported_mcp_type(fixture_type: str) -> str:
    try:
        return TYPE_MAP[fixture_type]
    except KeyError as error:
        raise RuntimeError(f"unsupported fixture type: {fixture_type}") from error


def eligible_candidates(raw: list[dict[str, Any]], case: dict[str, Any],
                        anchor_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Select active, same-namespace calibration candidates using metadata only."""
    eligible = []
    for row in raw:
        meta = anchor_map.get(str(row.get("fixture_id", "")))
        if not meta:
            continue
        if meta.get("namespace") != case.get("namespace"):
            continue
        if meta.get("split") != "calibration" or meta.get("status") != "active":
            continue
        eligible.append(row)
    return eligible


def threshold_grid(scored: dict[str, list[dict[str, Any]]]) -> list[float]:
    values = sorted({float(row["score"]) for rows in scored.values() for row in rows})
    return values + [math.nextafter(values[-1], math.inf)] if values else [0.0]


def _top_scored(rows: list[dict[str, Any]], threshold: float, k: int) -> list[dict[str, Any]]:
    return [row for row in rows if float(row["score"]) >= threshold][:k]


def evaluate_policy(cases: list[dict[str, Any]], scored: dict[str, list[dict[str, Any]]],
                    threshold: float, k: int,
                    raw_scored: dict[str, list[dict[str, Any]]] | None = None,
                    anchor_map: dict[str, dict[str, Any]] | None = None) -> dict[str, float | int]:
    raw_scored = raw_scored if raw_scored is not None else scored
    anchor_map = anchor_map or {}
    positives = [case for case in cases if case.get("answerable") is True]
    negatives = [case for case in cases if case.get("answerable") is False]
    recall_limit = min(k, 5)
    hits: dict[str, list[int]] = {}
    for category in sorted({case["category"] for case in positives}):
        group = [case for case in positives if case["category"] == category]
        hits[category] = [
            int(bool(set(case.get("expected_fixture_ids", [])) & {
                row["fixture_id"] for row in _top_scored(scored.get(case["case_id"], []), threshold, recall_limit)
            })) for case in group
        ]
    total_hits = sum(sum(values) for values in hits.values())
    category_recall = {
        f"{category}_recall_at_5": sum(values) / len(values) if values else 0.0
        for category, values in hits.items()
    }
    false_positive_queries = sum(
        bool(_top_scored(scored.get(case["case_id"], []), threshold, k)) for case in negatives
    )

    raw_namespace_leaks = raw_forbidden = raw_superseded = raw_unmapped = 0
    selected_namespace_leaks = selected_forbidden = superseded_selected = 0
    for case in cases:
        case_id = case["case_id"]
        forbidden = set(case.get("forbidden_fixture_ids", []))
        raw_rows = raw_scored.get(case_id, [])
        raw_forbidden += sum(row.get("fixture_id") in forbidden for row in raw_rows)
        raw_unmapped += sum(row.get("fixture_id") is None for row in raw_rows)
        raw_namespace_leaks += sum(
            row.get("fixture_id") in anchor_map
            and anchor_map[row["fixture_id"]].get("namespace") != case.get("namespace")
            for row in raw_rows
        )
        raw_superseded += sum(
            row.get("fixture_id") in anchor_map
            and anchor_map[row["fixture_id"]].get("status") != "active"
            for row in raw_rows
        )
        selected_rows = _top_scored(scored.get(case_id, []), threshold, k)
        selected_forbidden += sum(row.get("fixture_id") in forbidden for row in selected_rows)
        for row in selected_rows:
            metadata = anchor_map.get(str(row.get("fixture_id", "")), {})
            if metadata and metadata.get("namespace") != case.get("namespace"):
                selected_namespace_leaks += 1
            if metadata and metadata.get("status") != "active":
                superseded_selected += 1

    result: dict[str, float | int] = {
        "positive_case_count": len(positives),
        "negative_case_count": len(negatives),
        "recall_at_5": total_hits / len(positives) if positives else 0.0,
        "false_positive_queries_at_k": false_positive_queries,
        "negative_false_positive_rate_at_k": false_positive_queries / len(negatives) if negatives else 0.0,
        "raw_namespace_leak_count": raw_namespace_leaks,
        "selected_namespace_leak_count": selected_namespace_leaks,
        "raw_forbidden_candidate_exposure_count": raw_forbidden,
        "raw_superseded_candidate_count": raw_superseded,
        "raw_unmapped_candidate_count": raw_unmapped,
        "selected_forbidden_candidate_exposure_count": selected_forbidden,
        "superseded_selected_candidate_count": superseded_selected,
    }
    result.update(category_recall)
    return result


def policy_feasible(metrics: dict[str, float | int]) -> bool:
    """Fail closed when any frozen gate is absent or non-finite/non-numeric."""
    for gates, compare in ((MIN_GATES, lambda value, limit: value >= limit),
                           (MAX_GATES, lambda value, limit: value <= limit)):
        for gate, limit in gates.items():
            value = metrics.get(gate)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)) or not compare(float(value), limit):
                return False
    return True


def policy_sweep(cases: list[dict[str, Any]], scored: dict[str, list[dict[str, Any]]],
                 raw_scored: dict[str, list[dict[str, Any]]] | None = None,
                 anchor_map: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    thresholds = threshold_grid(raw_scored if raw_scored is not None else scored)
    sweep = []
    for threshold in thresholds:
        for k in POLICY_KS:
            metrics = evaluate_policy(cases, scored, threshold, k, raw_scored, anchor_map)
            sweep.append({"threshold": threshold, "k": k,
                          "feasible": policy_feasible(metrics), "metrics": metrics})
    feasible = [row for row in sweep if row["feasible"]]
    selected = max(feasible, key=lambda row: (row["metrics"]["recall_at_5"], -row["k"], row["threshold"]), default=None)
    return {
        "case_count": len(cases), "score_threshold_count": len(thresholds),
        "policy_sweep_count": len(sweep), "feasible_policy_count": len(feasible),
        "selected_policy": ({"threshold": selected["threshold"], "k": selected["k"],
                             "calibration_metrics": selected["metrics"]} if selected else None),
        "sweep": sweep,
    }


def _result_items(body: Any, case_id: str) -> list[dict[str, Any]]:
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        items = field(body, "results")
        if items is None:
            items = field(body, "memories")
    else:
        items = None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise RuntimeError(f"recall results malformed for {case_id}")
    return items


def _numeric_score(item: dict[str, Any], case_id: str) -> float:
    value = field(item, "score")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeError(f"recall score missing or non-numeric for {case_id}")
    return float(value)


def _preserve_recall_candidates(items: list[dict[str, Any]],
                                memory_to_fixture: dict[str, str],
                                case_id: str) -> tuple[list[dict[str, Any]], list[dict[str, float | int]]]:
    """Retain rank and score for every result, but never retain candidate content or unknown IDs."""
    rows: list[dict[str, Any]] = []
    unmapped: list[dict[str, float | int]] = []
    for rank, item in enumerate(items, 1):
        memory_id = field(item, "id")
        fixture_id = memory_to_fixture.get(str(memory_id)) if memory_id is not None else None
        score = _numeric_score(item, case_id)
        rows.append({"fixture_id": fixture_id, "score": score, "rank": rank})
        if fixture_id is None:
            unmapped.append({"score": score, "rank": rank})
    return rows, unmapped


def _vector_indexed(body: Any) -> int:
    if not isinstance(body, dict):
        raise RuntimeError("stats response is not an object")
    store = field(body, "store", body)
    if not isinstance(store, dict):
        raise RuntimeError("stats store response is not an object")
    value = field(store, "vector_indexed")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("stats response has no valid vector_indexed count")
    return value


def _validated_hash_fixture_ids(dataset: dict[str, Any]) -> set[str]:
    """Identify only the ten canonical v2 raw SHA-256 records, without exposing digests."""
    validate_development_dataset_v2(dataset)
    records = {row["fixture_id"]: row for row in dataset["records"]}
    cases = [row for row in dataset["cases"] if row.get("group") == "exact_hash"]
    expected_ids = {f"D-HASH-{index:02d}" for index in range(1, 11)}
    if len(cases) != 10:
        raise RuntimeError("development-v2 exact-hash case set is unexpected")
    actual_ids: set[str] = set()
    for case in cases:
        expected = case.get("expected_fixture_ids")
        if (case.get("expected_safety") != "answer" or not isinstance(expected, list)
                or len(expected) != 1 or not isinstance(case.get("expected_answer"), str)):
            raise RuntimeError("development-v2 exact-hash labels are malformed")
        fixture_id = expected[0]
        record = records.get(fixture_id)
        if (fixture_id not in expected_ids or record is None
                or not re.fullmatch(r"[0-9a-f]{64}", case["expected_answer"])):
            raise RuntimeError("development-v2 exact-hash fixture identity is unexpected")
        digest_match = re.search(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", record["content"])
        if digest_match is None or digest_match.group(0) != case["expected_answer"]:
            raise RuntimeError("development-v2 exact-hash fixture does not match its canonical label")
        actual_ids.add(fixture_id)
    if actual_ids != expected_ids:
        raise RuntimeError("development-v2 exact-hash fixture IDs are incomplete or unexpected")
    return actual_ids


SAFE_PROVIDER_REASONS = frozenset({
    "OpenAI Key", "Anthropic Key", "AWS Access Key", "AWS Secret Key", "Bearer Token",
    "GitHub Token", "GitHub PAT", "JWT Token", "PostgreSQL URL", "RSA Private Key",
    "EC Private Key", "ENV Variable",
})


def _safe_provider_reason(reason: Any) -> str:
    """Expose only static reason labels known from the filter; redact everything else."""
    return reason if isinstance(reason, str) and reason in SAFE_PROVIDER_REASONS else "unrecognized (redacted)"


def _classify_store_response(record: dict[str, Any], response: Any,
                             hash_fixture_ids: set[str]) -> tuple[str | None, str | None]:
    """Return (memory ID, blocked reason) for success or the one validated synthetic block."""
    fixture_id = record["fixture_id"]
    if not isinstance(response, dict):
        raise RuntimeError(f"store response malformed for {fixture_id}")
    status = response.get("status")
    if isinstance(status, bool) or not isinstance(status, int):
        raise RuntimeError(f"store response status malformed for {fixture_id}")
    body = response.get("body")
    if status == 201:
        if not isinstance(body, dict):
            raise RuntimeError(f"store response body malformed for {fixture_id}")
        memory_id = field(body, "id")
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise RuntimeError(f"store response omitted or malformed id for {fixture_id}")
        return memory_id, None
    if status == 403:
        if not isinstance(body, dict):
            raise RuntimeError(f"malformed HTTP 403 response for {fixture_id}; response body redacted")
        reason = field(body, "reason")
        safe_reason = _safe_provider_reason(reason)
        if fixture_id not in hash_fixture_ids:
            raise RuntimeError(f"unexpected HTTP 403 for non-hash fixture {fixture_id}; provider reason {safe_reason}")
        if reason != "AWS Secret Key":
            raise RuntimeError(f"unexpected HTTP 403 reason for {fixture_id}; provider reason {safe_reason}")
        return None, "AWS Secret Key"
    raise RuntimeError(f"store failed for {fixture_id}: HTTP {status}")


def quality_outcome(metrics: dict[str, Any], requested_count: int,
                    accepted_count: int, blocked_count: int) -> tuple[bool, bool, bool]:
    """Keep measured policy metrics separate from the hard full-coverage quality gate."""
    metrics_pass = policy_feasible(metrics)
    coverage_complete = blocked_count == 0 and accepted_count == requested_count
    return metrics_pass, coverage_complete, metrics_pass and coverage_complete


def retrieval_comparability(coverage_complete: bool) -> dict[str, str | None]:
    """Distinguish full-corpus metrics from partial-ingestion diagnostics."""
    if coverage_complete:
        return {"status": "FULL_CORPUS", "caveat": None}
    return {
        "status": "PARTIAL_CORPUS_DIAGNOSTIC_ONLY",
        "caveat": (
            "retrieval metrics are diagnostic on a partial corpus; top-K/ranking and negatives may differ "
            "from full 160-record corpus; not comparable to a full-corpus run; blocked hash cases remain "
            "in denominator and quality cannot pass"
        ),
    }


def evaluate_api_workflow(dataset: dict[str, Any],
                          request: Any) -> dict[str, Any]:
    """Store, attest indexing, and query every original case through an injected transport."""
    validate_development_dataset_v2(dataset)
    hash_fixture_ids = _validated_hash_fixture_ids(dataset)
    records, cases = dataset["records"], dataset["cases"]
    metadata = {row["fixture_id"]: {key: row[key] for key in ("namespace", "split", "status", "supersedes")}
                for row in records}
    memory_to_fixture: dict[str, str] = {}
    blocked_records: list[dict[str, Any]] = []
    for row in records:
        response = request("POST", "/v1/memories", {
            "content": row["content"], "namespace": row["namespace"],
            "type": supported_mcp_type(row["type"]),
        })
        memory_id, reason = _classify_store_response(row, response, hash_fixture_ids)
        if reason is not None:
            blocked_records.append({"fixture_id": row["fixture_id"], "status": 403, "reason": reason})
            continue
        assert memory_id is not None
        if memory_id in memory_to_fixture:
            raise RuntimeError("store API returned a duplicate memory ID; candidate mapping is ambiguous")
        memory_to_fixture[memory_id] = row["fixture_id"]

    stats_response = request("GET", "/v1/stats", None)
    if not isinstance(stats_response, dict) or stats_response.get("status") != 200:
        status = stats_response.get("status") if isinstance(stats_response, dict) else "malformed"
        raise RuntimeError(f"stats returned HTTP {status}")
    vector_indexed = _vector_indexed(stats_response.get("body"))
    if vector_indexed != len(memory_to_fixture):
        raise RuntimeError(f"vector index count mismatch: {vector_indexed}/{len(memory_to_fixture)} accepted records")

    raw: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {}
    latency_ms: dict[str, float] = {}
    for case in cases:
        payload = {"query": case["query"], "limit": 12, "min_score": 0.0}
        if case["namespace"] is not None:
            payload["namespace"] = case["namespace"]
        started = time.monotonic()
        response = request("POST", "/v1/recall", payload)
        latency_ms[case["case_id"]] = round((time.monotonic() - started) * 1000, 2)
        if not isinstance(response, dict) or response.get("status") != 200:
            status = response.get("status") if isinstance(response, dict) else "malformed"
            raise RuntimeError(f"recall failed for {case['case_id']}: HTTP {status}")
        items = _result_items(response.get("body"), case["case_id"])
        candidates, unmapped = _api_candidates(items, memory_to_fixture, case["case_id"])
        if unmapped:
            raise RuntimeError(f"recall returned unexpected memory IDs for {case['case_id']}")
        candidate_ids = [row["fixture_id"] for row in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RuntimeError(f"recall returned duplicate fixture IDs for {case['case_id']}")
        raw[case["case_id"]] = candidates
        sidecar = _sidecar_for_candidates(candidates, metadata)
        # Labels remain outside inference; only query, explicit scope, API candidates, and sidecar enter.
        predictions[case["case_id"]] = frozen_decision(case["query"], case["namespace"], candidates, sidecar)
        diagnostics[case["case_id"]] = {
            "nyawa_response_candidate_count": len(candidates),
            "sidecar_candidate_count": len(sidecar),
            "unmapped_candidate_count": len(unmapped), "unmapped_candidates": unmapped,
            "raw_superseded_count": sum(metadata[row["fixture_id"]]["status"] != "active"
                                        for row in candidates if row["fixture_id"] in metadata),
            "raw_cross_namespace_count": sum(metadata[row["fixture_id"]]["namespace"] != case["namespace"]
                                             for row in candidates if row["fixture_id"] in metadata
                                             and case["namespace"] is not None),
        }

    metrics = score_predictions(cases, raw, predictions, metadata)
    metrics_pass, coverage_complete, quality_pass = quality_outcome(
        metrics, len(records), len(memory_to_fixture), len(blocked_records))
    latencies = sorted(latency_ms.values())
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
    return {
        "status": "SYNTHETIC_QUALITY_PASS" if quality_pass else "SYNTHETIC_QUALITY_FAIL",
        "process_status": "COMPLETED", "ingestion_status": "FULL" if coverage_complete else "PARTIAL",
        "quality_status": "PASS" if quality_pass else "FAIL",
        "quality_gates_pass": quality_pass, "observed_policy_metrics_pass": metrics_pass,
        "coverage_complete": coverage_complete,
        "ingestion": {"requested_count": len(records),
                      "requested_fixture_ids": [row["fixture_id"] for row in records],
                      "accepted_count": len(memory_to_fixture),
                      "accepted_fixture_ids": list(memory_to_fixture.values()),
                      "blocked_count": len(blocked_records), "blocked_records": blocked_records,
                      "vector_indexed": vector_indexed},
        "filter_check_diagnostics": {
            "canonical_raw_sha256_fixture_count": len(hash_fixture_ids),
            "blocked_fixture_ids": [row["fixture_id"] for row in blocked_records],
            "provider_reasons": sorted({row["reason"] for row in blocked_records}),
            "native_limitation": (
                f"Observed Nyawa rejecting {len(blocked_records)} raw synthetic SHA-256 record(s) with reason "
                "'AWS Secret Key'; those cases remain in the original scoring denominator."
                if blocked_records else
                "No expected raw synthetic SHA-256 rejection was observed in this run; no native limitation is claimed."
            ),
        },
        "quality_metrics": metrics, "raw_retrieval_diagnostics": diagnostics,
        "retrieval_results": {case["case_id"]: [{"fixture_id": row["fixture_id"], "score": row["score"],
                             "rank": row["rank"]} for row in raw[case["case_id"]]] for case in cases},
        "predictions": predictions, "latency_ms": latency_ms,
        "latency_median_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
        "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
        "queries_run": len(latency_ms), "fixture_counts": {
            "synthetic_development_cases": len(cases), "synthetic_development_records": len(records)},
    }


FROZEN_THRESHOLD = 0.99
FROZEN_POLICY_PROVENANCE = "snapshot of offline prototype.py inspected 2026-11-11; no tuning on v2 corpus"
_POLICY_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.I)
# Keep exact frozen prototype boundaries: underscore-adjacent IDs are not boundary matches,
# while Unicode word and digit semantics follow Python's original ``re.I`` / ``\d`` rules.
_POLICY_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", re.I)
_POLICY_NUMBER_RE = re.compile(r"\b\d+\b")
_POLICY_STOP = set("a an and are as at be does for from how in is it many name of on or the to what which who why with find lookup record synthetic path identifier current latest updated rule governs applies policy window cache refresh often describe topic".split())


def _policy_tokens(text: str) -> set[str]:
    return {token.casefold() for token in _POLICY_TOKEN_RE.findall(text)}


def _policy_metadata_ok(fid: str, sidecar: dict[str, dict[str, Any]]) -> bool:
    meta = sidecar.get(fid)
    return (isinstance(meta, dict) and all(k in meta for k in ("namespace", "split", "status", "supersedes"))
            and isinstance(meta["namespace"], str) and bool(meta["namespace"])
            and meta["split"] == "synthetic_development" and meta["status"] in {"active", "superseded"}
            and isinstance(meta["supersedes"], list)
            and all(isinstance(x, str) and x for x in meta["supersedes"]))


def _policy_lineage_ok(sidecar: dict[str, dict[str, Any]]) -> bool:
    graph = {}
    for fid, meta in sidecar.items():
        if not _policy_metadata_ok(fid, sidecar):
            return False
        graph[fid] = list(meta["supersedes"])
        if len(graph[fid]) != len(set(graph[fid])):
            return False
        for old_id in graph[fid]:
            old = sidecar.get(old_id)
            if (old is None or not _policy_metadata_ok(old_id, sidecar)
                    or old["status"] != "superseded" or old["namespace"] != meta["namespace"]):
                return False
    visiting, visited = set(), set()

    def visit(fid: str) -> bool:
        if fid in visiting:
            return False
        if fid in visited:
            return True
        visiting.add(fid)
        if not all(visit(old) for old in graph[fid]):
            return False
        visiting.remove(fid)
        visited.add(fid)
        return True

    if not all(visit(fid) for fid in graph):
        return False
    descendants = {fid: set() for fid in graph}

    def ancestors(fid: str) -> set[str]:
        result = set(graph[fid])
        for prior in graph[fid]:
            result.update(ancestors(prior))
        return result

    for fid, meta in sidecar.items():
        if meta["status"] == "active":
            for old in ancestors(fid):
                descendants[old].add(fid)
    return all(len(items) <= 1 for items in descendants.values())


def frozen_resolve(query: str, namespace: str, candidates: list[dict[str, Any]],
                   sidecar: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Frozen prototype algorithm; only query, scope, API candidates and metadata enter."""
    if not isinstance(query, str) or not query.strip() or not isinstance(namespace, str) or not namespace:
        return None
    if not candidates or not _policy_lineage_ok(sidecar):
        return None
    usable = []
    seen = set()
    for row in candidates:
        if not isinstance(row, dict):
            return None
        fid, content = row.get("fixture_id"), row.get("content")
        if (not isinstance(fid, str) or not fid or fid in seen or not isinstance(content, str)
                or not _policy_metadata_ok(fid, sidecar) or "score" not in row or "rank" not in row):
            return None
        seen.add(fid)
        score, rank = row["score"], row["rank"]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            return None
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            return None
        meta = sidecar[fid]
        if meta["namespace"] != namespace:
            return None
        if meta["status"] == "active":
            usable.append(row)
    if not usable:
        return None
    identifiers = {x.casefold() for x in _POLICY_ID_RE.findall(query)}
    if identifiers:
        matches = [row for row in usable if identifiers & {x.casefold() for x in _POLICY_ID_RE.findall(row["content"])}]
        return {"fixture_id": matches[0]["fixture_id"], "confidence": 1.0, "method": "exact_identifier"} if len(matches) == 1 else None
    numbers = set(_POLICY_NUMBER_RE.findall(query))
    query_terms = _policy_tokens(query) - _POLICY_STOP - numbers
    state_request = bool({"current", "latest", "updated"} & _policy_tokens(query))
    ranked = []
    for row in usable:
        content = row["content"]
        if numbers and not numbers.issubset(set(_POLICY_NUMBER_RE.findall(content))):
            continue
        overlap = query_terms & _policy_tokens(content)
        if query_terms and not overlap:
            continue
        if not overlap and not numbers:
            continue
        coverage = len(overlap) / max(1, len(query_terms))
        confidence = 0.85 + 0.14 * coverage if numbers else 0.70 + 0.29 * coverage
        meta = sidecar[row["fixture_id"]]
        priority = int(state_request and bool(meta["supersedes"]))
        ranked.append((priority, confidence, float(row["score"]), -row["rank"], row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[:4], reverse=True)
    best = ranked[0]
    if sum(item[:3] == best[:3] for item in ranked) != 1:
        return None
    return {"fixture_id": best[4]["fixture_id"], "confidence": best[1], "method": "content_anchor"}


def frozen_decision(query: str, request_namespace: str | None,
                    candidates: list[dict[str, Any]],
                    sidecar: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Label-blind adapter; only trusted explicit request scope can authorize an answer."""
    if not isinstance(request_namespace, str) or not request_namespace:
        return {"decision": "abstain", "reason": "missing_request_namespace"}
    resolved = frozen_resolve(query, request_namespace, candidates, sidecar)
    if resolved is None:
        return {"decision": "abstain", "reason": "frozen_policy_abstain"}
    confidence = resolved.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or confidence < FROZEN_THRESHOLD:
        return {"decision": "abstain", "reason": "below_fixed_threshold", "confidence": confidence}
    return {"decision": "answer", "fixture_id": resolved["fixture_id"],
            "confidence": confidence, "method": resolved.get("method")}


def _api_candidates(items: list[dict[str, Any]], memory_to_fixture: dict[str, str], case_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates, unmapped = [], []
    for index, item in enumerate(items, 1):
        memory_id = field(item, "id")
        fixture_id = memory_to_fixture.get(str(memory_id)) if memory_id is not None else None
        score = _numeric_score(item, case_id)
        content = field(item, "content", "")
        rank = field(item, "rank", index)
        if not isinstance(content, str):
            raise RuntimeError(f"recall content malformed for {case_id}")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise RuntimeError(f"recall rank missing or invalid for {case_id}")
        row = {"fixture_id": fixture_id, "content": content, "score": score, "rank": rank}
        candidates.append(row)
        if fixture_id is None:
            unmapped.append({"rank": rank, "score": score})
    return candidates, unmapped


def _sidecar_for_candidates(candidates: list[dict[str, Any]],
                            metadata: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build relevant simulated metadata, including same-scope alternate successors.

    Nyawa does not provide these state/lineage fields. In the synthetic harness, trusted
    fixture metadata can reveal a competing successor omitted from retrieval top-K; this
    sidecar is not evidence of equivalent native Nyawa behavior.
    """
    sidecar: dict[str, dict[str, Any]] = {}
    seeds = {row["fixture_id"] for row in candidates
             if isinstance(row, dict) and isinstance(row.get("fixture_id"), str)}
    # Expand the lineage-connected component of retrieved entities, not every item in
    # a namespace. This finds omitted competing successors while unrelated malformed
    # state does not poison the whole namespace.
    neighbors: dict[str, set[str]] = {fid: set() for fid in metadata}
    for fid, source in metadata.items():
        if not isinstance(source, dict):
            continue
        supersedes = source.get("supersedes")
        if not isinstance(supersedes, list):
            continue
        for prior in supersedes:
            old = metadata.get(prior)
            if (isinstance(prior, str) and isinstance(old, dict)
                    and old.get("namespace") == source.get("namespace")):
                neighbors.setdefault(fid, set()).add(prior)
                neighbors.setdefault(prior, set()).add(fid)
    pending = list(seeds)
    relevant: set[str] = set()
    while pending:
        fid = pending.pop()
        if fid in relevant:
            continue
        relevant.add(fid)
        pending.extend(neighbors.get(fid, set()) - relevant)
    for fid in relevant:
        source = metadata.get(fid)
        if not isinstance(source, dict):
            continue
        # Preserve absent/malformed required values; resolver validation then abstains.
        sidecar[fid] = {key: source[key] for key in ("namespace", "split", "status", "supersedes")
                        if key in source}
    return sidecar


def score_predictions(cases: list[dict[str, Any]], raw: dict[str, list[dict[str, Any]]],
                     predictions: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positives = [case for case in cases if case["expected_safety"] == "answer"]
    negatives = [case for case in cases if case["expected_safety"] == "abstain"]
    categories = ("exact_identifier", "active_decision", "hard_constraint", "supersession", "namespace_isolation")
    selected_hits: dict[str, int] = {category: 0 for category in categories}
    raw_hits: dict[str, int] = {category: 0 for category in categories}
    selected_source_count = exact_answers = 0

    def top_five(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if isinstance(row.get("rank"), int)
                and not isinstance(row.get("rank"), bool) and 1 <= row["rank"] <= 5]

    raw_namespace_leaks = raw_forbidden = raw_superseded = raw_unmapped = 0
    selected_namespace_leaks = selected_forbidden = selected_superseded = 0
    false_positive_queries = 0
    for case in cases:
        case_id = case["case_id"]
        expected_ns = case.get("namespace")
        rows = raw[case_id]
        raw_top_five = top_five(rows)
        expected_ids = set(case.get("expected_fixture_ids", []))
        prediction = predictions[case_id]
        selected_id = prediction.get("fixture_id") if prediction.get("decision") == "answer" else None
        selected_rows = [row for row in rows if row.get("fixture_id") == selected_id] if selected_id else []
        selected_row = selected_rows[0] if selected_rows else None
        selected_meta = metadata.get(selected_id, {}) if selected_id else {}
        selected_scope_ok = (selected_meta.get("namespace") == expected_ns and expected_ns is not None
                             and selected_meta.get("split") == "synthetic_development"
                             and selected_meta.get("status") == "active")
        answer_text = case.get("expected_answer", "")
        selected_content_ok = (selected_row is not None and isinstance(answer_text, str) and bool(answer_text)
                               and answer_text.casefold() in str(selected_row.get("content", "")).casefold())
        correct_source = bool(selected_id and selected_id in expected_ids and selected_scope_ok and selected_content_ok)
        if correct_source:
            selected_source_count += 1
            exact_answers += 1

        if case in positives:
            if any(row.get("fixture_id") in expected_ids for row in raw_top_five):
                raw_hits[case["category"]] = raw_hits.get(case["category"], 0) + 1
            # Recall gates evaluate the frozen one-source output, and @5 means its
            # actual Nyawa rank must be within the first five returned candidates.
            if correct_source and selected_row is not None and selected_row in raw_top_five:
                selected_hits[case["category"]] = selected_hits.get(case["category"], 0) + 1

        if case in negatives and prediction.get("decision") == "answer":
            false_positive_queries += 1
        raw_unmapped += sum(row.get("fixture_id") is None for row in rows)
        raw_forbidden += sum(row.get("fixture_id") in case.get("forbidden_fixture_ids", []) for row in rows)
        raw_namespace_leaks += sum(
            row.get("fixture_id") in metadata and expected_ns is not None
            and metadata[row["fixture_id"]].get("namespace") != expected_ns for row in rows
        )
        raw_superseded += sum(
            row.get("fixture_id") in metadata and metadata[row["fixture_id"]].get("status") != "active"
            for row in rows
        )
        if selected_id:
            selected_namespace_leaks += int(not selected_scope_ok)
            selected_forbidden += int(selected_id in case.get("forbidden_fixture_ids", []))
            selected_superseded += int(selected_meta.get("status") != "active"
                                        or selected_meta.get("split") != "synthetic_development")

    category_recall = {
        f"{category}_recall_at_5": selected_hits.get(category, 0) /
        sum(case["category"] == category for case in positives)
        if any(case["category"] == category for case in positives) else 0.0
        for category in categories
    }
    raw_category_recall = {
        f"raw_retrieval_{category}_recall_at_5": raw_hits.get(category, 0) /
        sum(case["category"] == category for case in positives)
        if any(case["category"] == category for case in positives) else 0.0
        for category in categories
    }
    raw_retrieval_hits = sum(raw_hits.values())
    selected_retrieval_hits = sum(selected_hits.values())
    metrics: dict[str, Any] = {
        "positive_case_count": len(positives), "negative_case_count": len(negatives),
        "recall_at_5": selected_retrieval_hits / len(positives) if positives else 0.0,
        "raw_retrieval_recall_at_5": raw_retrieval_hits / len(positives) if positives else 0.0,
        "frozen_policy_correct_source_count": selected_source_count,
        "frozen_policy_exact_answer_count": exact_answers,
        "negative_false_positive_queries_at_k": false_positive_queries,
        "negative_false_positive_rate_at_k": false_positive_queries / len(negatives) if negatives else 0.0,
        "raw_namespace_leak_count": raw_namespace_leaks,
        "selected_namespace_leak_count": selected_namespace_leaks,
        "raw_forbidden_candidate_exposure_count": raw_forbidden,
        "raw_superseded_candidate_count": raw_superseded,
        "raw_unmapped_candidate_count": raw_unmapped,
        "unknown_id_candidate_count": raw_unmapped,
        "unknown_id_case_count": sum(any(row.get("fixture_id") is None for row in raw[case["case_id"]])
                                      for case in cases),
        "selected_forbidden_candidate_exposure_count": selected_forbidden,
        "superseded_selected_candidate_count": selected_superseded,
    }
    metrics.update(category_recall)
    metrics.update(raw_category_recall)
    return metrics


def run_e2e(container: str, image: str, fixture_path: Path) -> dict[str, Any]:
    dataset = load_dataset(fixture_path)
    if image != IMAGE_C87:
        raise RuntimeError("C8.7 requires its pinned immutable BGE runtime image")
    preflight()
    docker(["pull", image], timeout=300)
    image_info = json.loads(docker(["image", "inspect", image]).stdout)[0]
    if image not in image_info.get("RepoDigests", []):
        raise RuntimeError("pulled image digest mismatch")
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("container already exists")
    db_path = "/scratch/c87-synthetic-development-v2.db"
    try:
        docker(["run", "--detach", "--name", container, "--label", f"{LABEL_KEY}=true",
                "--network=none", "--read-only", "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--memory=4g", "--memory-swap=4g",
                "--cpus=2", "--pids-limit=128", "--restart=no", "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true", "--env", "MEMORY_EMBEDDER=local",
                "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model", "--env", "HOME=/tmp",
                "--env", "TMPDIR=/tmp", image, "serve", db_path], timeout=60)
        verify_container(container, image)
        deadline, last_error = time.monotonic() + 180, "not ready"
        while time.monotonic() < deadline:
            state = json.loads(docker(["inspect", "--format", "{{json .State}}", container]).stdout)
            if not state.get("Running"):
                logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
                raise RuntimeError(f"Nyawa exited: {logs}")
            try:
                root, health = inside_http(container, "GET", "/"), inside_http(container, "GET", "/v1/health")
                if root["status"] == 200 and health["status"] == 200:
                    break
                last_error = f"root={root['status']} health={health['status']}"
            except Exception as error:
                last_error = str(error)
            time.sleep(2)
        else:
            logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
            raise RuntimeError(f"Nyawa timeout: {last_error}; {logs}")

        # The injected transport keeps the same local HTTP calls while making partial ingestion
        # and its original-denominator scoring independently testable without Docker or network.
        result = evaluate_api_workflow(dataset, lambda method, path, payload:
                                       inside_http(container, method, path, payload))
        return {
            "schema": "c8.7.synthetic-development-e2e-report/v2",
            **result,
            "infrastructure_status": "PASS", "scope": "corrected synthetic development only",
            "ci_exit_behavior": "quality or corpus coverage failure exits 2 after printing the full JSON report; infrastructure failure exits 1; pass exits 0",
            "retrieval_comparability": retrieval_comparability(result["coverage_complete"]),
            "classification": dataset["classification"], "sealed_holdout_evaluated": False,
            "holdout_generated": False, "actual_nyawa": True,
            "backend": {"requested": "local embedder", "runtime_model": "not attested",
                        "runtime_dimension": "not attested"},
            "vector_indexed": result["ingestion"]["vector_indexed"],
            "vector_indexing_attestation": {
                "evidence": "Nyawa /v1/stats vector_indexed count only",
                "vector_indexed": result["ingestion"]["vector_indexed"],
                "model_identity_verified": False,
                "dimension_verified": False,
            },
            "frozen_policy": FROZEN_POLICY_PROVENANCE,
            "threshold": FROZEN_THRESHOLD, "threshold_tuned_on_this_corpus": False,
            "policy_gates": {"minimum": MIN_GATES, "maximum": MAX_GATES},
            "eligibility": {"inference_inputs": ["query", "request_namespace", "nyawa_response_candidates", "simulated_metadata_sidecar"],
                            "inference_labels_used": False, "scoring_labels_used_after_inference": True,
                            "raw_retrieval_preserved": True,
                            "native_metadata_limitation": "Nyawa HTTP store exposes content, namespace and type; status/supersedes are a simulated fixture sidecar, not native API metadata.",
                            "lineage_coverage_limitation": "The synthetic sidecar checks the lineage-connected component of returned candidates, including known omitted alternative successors in that component. Unlinked or unavailable alternatives omitted by top-K cannot be detected; no global Nyawa lineage-conflict guarantee is claimed."},
            "fixture_validation": {
                "validator": "fixed v2 group-specific evidence/query contract",
                "semantic_truth_attested": False,
                "human_sample_review_required": True,
            },
            "answer_generation": False,
            "scoring_semantics": "scorer selects a source and checks expected-text containment; it does not generate an answer",
            "runtime_attestation": {"mode": "disposable local container", "network": "none",
                                    "database": "tmpfs only", "container": container, "host_mounts": False},
            "image": image, "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        }
    finally:
        cleanup(container, image)


class C87Tests(unittest.TestCase):
    def test_fixture_is_deterministic_and_calibration_only(self) -> None:
        first, second = build_dataset(), build_dataset()
        self.assertEqual(first, second)
        self.assertEqual(len(first["cases"]), 60)
        self.assertEqual(len(first["records"]), 40)
        self.assertEqual({row["split"] for row in first["cases"] + first["records"]}, {"calibration"})
        self.assertEqual(sum(case["answerable"] for case in first["cases"]), 50)
        namespace_cases = [case for case in first["cases"] if case["category"] == "namespace_isolation"]
        self.assertEqual(sum(case["answerable"] is False for case in namespace_cases), 5)
        self.assertEqual(sum(case["answerable"] is True for case in namespace_cases), 10)
        self.assertTrue(all(case["forbidden_fixture_ids"] for case in namespace_cases if not case["answerable"]))

    def test_schema_type_mapping_does_not_change_fixture_categories(self) -> None:
        self.assertEqual({key: supported_mcp_type(key) for key in TYPE_MAP}, {
            "artifact": "reference", "constraint": "procedure", "decision": "decision", "fact": "fact",
        })
        self.assertEqual({row["type"] for row in build_dataset()["records"]}, set(TYPE_MAP))
        self.assertEqual({case["category"] for case in build_dataset()["cases"]}, {
            "exact_identifier", "active_decision", "hard_constraint", "supersession",
            "namespace_isolation", "negative_confusable",
        })

    def test_metadata_filter_keeps_active_superseding_record(self) -> None:
        anchors = {
            "active": {"namespace": "ns", "split": "calibration", "status": "active", "supersedes": ["old"]},
            "stale": {"namespace": "ns", "split": "calibration", "status": "superseded", "supersedes": []},
            "foreign": {"namespace": "elsewhere", "split": "calibration", "status": "active"},
            "other_split": {"namespace": "ns", "split": "other", "status": "active"},
        }
        raw = [{"fixture_id": key, "score": 1.0} for key in anchors]
        self.assertEqual([row["fixture_id"] for row in eligible_candidates(raw, {"namespace": "ns"}, anchors)], ["active"])

    def test_recall_at_five_category_denominators_and_zero_fpr(self) -> None:
        cases = [
            {"case_id": "p1", "answerable": True, "category": "artifact", "expected_fixture_ids": ["hit"]},
            {"case_id": "p2", "answerable": True, "category": "constraint", "expected_fixture_ids": ["fifth"]},
            {"case_id": "n1", "answerable": False, "category": "negative_confusable", "expected_fixture_ids": []},
        ]
        scored = {
            "p1": [{"fixture_id": f"x{i}", "score": 0.9} for i in range(4)] + [{"fixture_id": "hit", "score": 0.8}],
            "p2": [{"fixture_id": f"z{i}", "score": 0.9} for i in range(4)] + [{"fixture_id": "fifth", "score": 0.8}],
            "n1": [],
        }
        result = evaluate_policy(cases, scored, 0.0, 5)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["artifact_recall_at_5"], 1.0)
        self.assertEqual(result["constraint_recall_at_5"], 1.0)
        self.assertEqual(result["negative_false_positive_rate_at_k"], 0.0)
        self.assertEqual(evaluate_policy(cases, scored, 0.0, 5)["negative_case_count"], 1)

    def test_policy_sweep_recall_at_five_respects_policy_k(self) -> None:
        cases = [{"case_id": "p", "answerable": True, "category": "exact_identifier",
                  "expected_fixture_ids": ["gold"]}]
        scored = {"p": [{"fixture_id": f"x{i}", "score": 0.5} for i in range(4)]
                       + [{"fixture_id": "gold", "score": 0.5}]}
        result = policy_sweep(cases, scored)
        at_threshold = [row for row in result["sweep"] if row["threshold"] == 0.5]
        recall_by_k = {row["k"]: row["metrics"]["recall_at_5"] for row in at_threshold}
        self.assertEqual(recall_by_k[1], 0.0)
        self.assertEqual(recall_by_k[5], 1.0)

    def test_filtered_raw_stale_forbidden_is_diagnostic_not_selected_gate(self) -> None:
        cases = [{"case_id": "supersession", "answerable": True, "category": "supersession",
                  "namespace": "local", "expected_fixture_ids": ["current"],
                  "forbidden_fixture_ids": ["old"]}]
        raw = {"supersession": [
            {"fixture_id": "old", "score": 0.9}, {"fixture_id": "current", "score": 0.8},
        ]}
        eligible = {"supersession": [{"fixture_id": "current", "score": 0.8}]}
        anchors = {
            "old": {"namespace": "local", "split": "calibration", "status": "superseded"},
            "current": {"namespace": "local", "split": "calibration", "status": "active", "supersedes": ["old"]},
        }
        metrics = evaluate_policy(cases, eligible, 0.0, 1, raw, anchors)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["raw_forbidden_candidate_exposure_count"], 1)
        self.assertEqual(metrics["raw_superseded_candidate_count"], 1)
        self.assertEqual(metrics["selected_forbidden_candidate_exposure_count"], 0)
        self.assertEqual(metrics["superseded_selected_candidate_count"], 0)
        gate_check = {**MIN_GATES, **MAX_GATES, **metrics}
        self.assertTrue(policy_feasible(gate_check))
        self.assertNotIn("raw_forbidden_candidate_exposure_count", MAX_GATES)

    def test_selected_namespace_forbidden_and_superseded_candidates_fail_gates(self) -> None:
        cases = [
            {"case_id": "cross", "answerable": False, "category": "namespace_isolation",
             "namespace": "local", "forbidden_fixture_ids": ["foreign"]},
            {"case_id": "stale", "answerable": True, "category": "supersession",
             "namespace": "local", "expected_fixture_ids": ["current"],
             "forbidden_fixture_ids": ["old"]},
        ]
        raw = {
            "cross": [{"fixture_id": "foreign", "score": 0.9}],
            "stale": [{"fixture_id": "old", "score": 0.9}],
        }
        # Deliberately pass unsafe rows as eligible to verify selected-output gates.
        eligible = {"cross": raw["cross"], "stale": raw["stale"]}
        anchors = {
            "foreign": {"namespace": "other", "split": "calibration", "status": "active"},
            "old": {"namespace": "local", "split": "calibration", "status": "superseded"},
        }
        metrics = evaluate_policy(cases, eligible, 0.0, 1, raw, anchors)
        self.assertEqual(metrics["raw_namespace_leak_count"], 1)
        self.assertEqual(metrics["selected_namespace_leak_count"], 1)
        self.assertEqual(metrics["raw_forbidden_candidate_exposure_count"], 2)
        self.assertEqual(metrics["selected_forbidden_candidate_exposure_count"], 2)
        self.assertEqual(metrics["superseded_selected_candidate_count"], 1)
        gate_check = {**MIN_GATES, **metrics}
        self.assertFalse(policy_feasible(gate_check))
        for gate in ("raw_namespace_leak_count", "selected_namespace_leak_count",
                     "selected_forbidden_candidate_exposure_count", "superseded_selected_candidate_count"):
            self.assertIn(gate, MAX_GATES)
        self.assertNotIn("raw_forbidden_candidate_exposure_count", MAX_GATES)
        for gate in MAX_GATES:
            violating_metrics = {**MIN_GATES, **MAX_GATES}
            violating_metrics[gate] = MAX_GATES[gate] + 1
            self.assertFalse(policy_feasible(violating_metrics), gate)

    def test_selected_safety_metrics_honor_policy_k(self) -> None:
        cases = [{"case_id": "p", "answerable": True, "category": "supersession",
                  "namespace": "local", "expected_fixture_ids": ["current"],
                  "forbidden_fixture_ids": ["old"]}]
        eligible = {"p": [
            {"fixture_id": "current", "score": 0.9}, {"fixture_id": "old", "score": 0.8},
        ]}
        anchors = {
            "current": {"namespace": "local", "status": "active"},
            "old": {"namespace": "local", "status": "superseded"},
        }
        at_one = evaluate_policy(cases, eligible, 0.0, 1, eligible, anchors)
        at_two = evaluate_policy(cases, eligible, 0.0, 2, eligible, anchors)
        self.assertEqual(at_one["selected_forbidden_candidate_exposure_count"], 0)
        self.assertEqual(at_one["superseded_selected_candidate_count"], 0)
        self.assertEqual(at_two["selected_forbidden_candidate_exposure_count"], 1)
        self.assertEqual(at_two["superseded_selected_candidate_count"], 1)

    def test_metric_misses_gold_outside_top_five_and_counts_false_positive(self) -> None:
        cases = [
            {"case_id": "p", "answerable": True, "category": "exact_identifier", "expected_fixture_ids": ["gold"]},
            {"case_id": "n", "answerable": False, "category": "negative_confusable", "expected_fixture_ids": []},
        ]
        scored = {
            "p": [{"fixture_id": f"x{i}", "score": 1.0 - i / 10} for i in range(5)] + [{"fixture_id": "gold", "score": 0.1}],
            "n": [{"fixture_id": "decoy", "score": 0.8}],
        }
        result = evaluate_policy(cases, scored, 0.0, 5)
        self.assertEqual(result["recall_at_5"], 0.0)
        self.assertEqual(result["negative_false_positive_rate_at_k"], 1.0)

    def test_api_field_case_handling_and_vector_index_count(self) -> None:
        self.assertEqual(field({"ID": "m1"}, "id"), "m1")
        self.assertEqual(field({"SCORE": 0.75}, "score"), 0.75)
        self.assertEqual(_result_items({"Results": [{"ID": "m1"}]}, "case"), [{"ID": "m1"}])
        self.assertEqual(_vector_indexed({"Store": {"Vector_Indexed": 40}}), 40)
        self.assertEqual(_numeric_score({"Score": 0.5}, "case"), 0.5)

    def test_unmapped_recall_candidates_preserve_safe_metadata_and_fail_closed(self) -> None:
        items = [
            {"id": "known-id", "score": 0.91, "content": "fixture text"},
            {"id": "unknown-id", "score": 0.83, "content": "must not be retained"},
        ]
        rows, unmapped = _preserve_recall_candidates(items, {"known-id": "known-fixture"}, "case")
        self.assertEqual(rows, [
            {"fixture_id": "known-fixture", "score": 0.91, "rank": 1},
            {"fixture_id": None, "score": 0.83, "rank": 2},
        ])
        self.assertEqual(unmapped, [{"score": 0.83, "rank": 2}])
        self.assertNotIn("content", str(rows) + str(unmapped))
        cases = [{"case_id": "case", "answerable": True, "category": "exact_identifier",
                  "namespace": "local", "expected_fixture_ids": ["known-fixture"]}]
        scored = {"case": [rows[0]]}
        anchors = {"known-fixture": {"namespace": "local", "split": "calibration", "status": "active"}}
        clean_metrics = evaluate_policy(cases, scored, 0.0, 1, {"case": [rows[0]]}, anchors)
        clean_gate_metrics = {**MIN_GATES, **MAX_GATES, **clean_metrics}
        self.assertEqual(clean_metrics["raw_unmapped_candidate_count"], 0)
        self.assertTrue(policy_feasible(clean_gate_metrics))
        metrics = evaluate_policy(cases, scored, 0.0, 1, {"case": rows}, anchors)
        self.assertEqual(metrics["raw_unmapped_candidate_count"], 1)
        gate_metrics = {**MIN_GATES, **MAX_GATES, **metrics}
        self.assertFalse(policy_feasible(gate_metrics))
        self.assertEqual(MAX_GATES["raw_unmapped_candidate_count"], 0)

    def test_report_candidate_projection_excludes_content(self) -> None:
        projected = {"fixture_id": "fid", "score": 0.9, "rank": 1}
        self.assertEqual(set(projected), {"fixture_id", "score", "rank"})


class C87DevelopmentV2Tests(unittest.TestCase):
    def test_v2_is_deterministic_development_only_and_evidence_supported(self) -> None:
        first, second = build_development_dataset_v2(), build_development_dataset_v2()
        self.assertEqual(first, second)
        self.assertEqual(first["schema_version"], SCHEMA_C87_V2)
        self.assertEqual(first["sealed_holdout"], False)
        self.assertEqual(first["classification"], "synthetic_development_not_independent_not_holdout_not_production_evidence")
        self.assertEqual((len(first["cases"]), len(first["records"])), (120, 160))
        self.assertEqual(first["corrections"]["underspecified_identifier_items"], 10)
        self.assertEqual(first["corrections"]["underspecified_hash_items"], 10)
        by_id = {row["fixture_id"]: row for row in first["records"]}
        for case in first["cases"]:
            if case["expected_safety"] == "answer":
                self.assertIn(case["expected_answer"].casefold(), by_id[case["expected_fixture_ids"][0]]["content"].casefold())
        self.assertEqual({case["language"] for case in first["cases"]}, {"en", "id"})
        self.assertTrue(all("candidate_document_ids" not in case for case in first["cases"]))

    def test_inference_is_label_blind_and_label_mutations_do_not_change_decision(self) -> None:
        dataset = build_development_dataset_v2()
        case = next(row for row in dataset["cases"] if row["group"] == "exact_identifier")
        record = next(row for row in dataset["records"] if row["fixture_id"] == case["expected_fixture_ids"][0])
        candidates = [{"fixture_id": record["fixture_id"], "content": record["content"], "score": 0.37, "rank": 1}]
        sidecar = _sidecar_for_candidates(candidates, {
            record["fixture_id"]: {key: record[key] for key in ("namespace", "split", "status", "supersedes")}
        })
        first = frozen_decision(case["query"], case["namespace"], candidates, sidecar)
        mutated = dict(case, expected_safety="abstain", expected_answer="MUTATED", expected_fixture_ids=[],
                       forbidden_fixture_ids=[record["fixture_id"]], group="mutated", category="mutated")
        second = frozen_decision(mutated["query"], mutated["namespace"], candidates, sidecar)
        self.assertEqual(first, second)
        self.assertEqual(first["decision"], "answer")
        import inspect
        self.assertEqual(list(inspect.signature(frozen_decision).parameters),
                         ["query", "request_namespace", "candidates", "sidecar"])
        inference_source = inspect.getsource(frozen_decision) + inspect.getsource(frozen_resolve)
        for forbidden in ("expected_answer", "expected_fixture_ids", "expected_safety", "forbidden_fixture_ids", "group", "language"):
            self.assertNotIn(forbidden, inference_source)

    def test_api_candidates_preserve_real_response_rank_score_content_and_unknown_ids(self) -> None:
        api_response = [
            {"ID": "nyawa-id-7", "Score": 0.912, "Content": "response content one", "Rank": 4},
            {"id": "unmapped-id", "score": 0.811, "content": "unknown result", "rank": 7},
        ]
        rows, unmapped = _api_candidates(api_response, {"nyawa-id-7": "D-1"}, "case")
        self.assertEqual(rows, [
            {"fixture_id": "D-1", "content": "response content one", "score": 0.912, "rank": 4},
            {"fixture_id": None, "content": "unknown result", "score": 0.811, "rank": 7},
        ])
        self.assertEqual(unmapped, [{"rank": 7, "score": 0.811}])

    def test_metadata_sidecar_is_simulated_and_missing_or_cross_scope_fails_closed(self) -> None:
        sidecar = {
            "new": {"namespace": "ns-a", "split": "synthetic_development", "status": "active", "supersedes": ["old"]},
            "old": {"namespace": "ns-a", "split": "synthetic_development", "status": "superseded", "supersedes": []},
        }
        candidate = [{"fixture_id": "new", "content": "Exact ref ERR-2026-1001", "score": 0.7, "rank": 1}]
        self.assertEqual(frozen_decision("Find ERR-2026-1001", "ns-a", candidate, sidecar)["decision"], "answer")
        self.assertEqual(sidecar["new"]["supersedes"], ["old"])
        missing = {"new": {"namespace": "ns-a", "split": "synthetic_development", "status": None, "supersedes": []}}
        self.assertEqual(frozen_decision("Find ERR-2026-1001", "ns-a", candidate, missing)["decision"], "abstain")
        foreign = {"new": {"namespace": "ns-b", "split": "synthetic_development", "status": "active", "supersedes": []}}
        self.assertEqual(frozen_decision("Find ERR-2026-1001", "ns-a", candidate, foreign)["decision"], "abstain")
        ambiguous = dict(sidecar, foreign={"namespace": "ns-b", "split": "synthetic_development", "status": "active", "supersedes": []})
        cross = candidate + [{"fixture_id": "foreign", "content": "Other", "score": 0.6, "rank": 2}]
        self.assertEqual(frozen_decision("Which alias?", None, cross, ambiguous)["reason"], "missing_request_namespace")

    def test_fixed_policy_answer_abstention_and_infeasible_quality_are_distinguished(self) -> None:
        sidecar = {"good": {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": []},
                   "decoy": {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": []}}
        positive = frozen_decision("Find ERR-2026-1001", "ns", [
            {"fixture_id": "good", "content": "Reference ERR-2026-1001", "score": 0.91, "rank": 1}], sidecar)
        negative = frozen_decision("Find ERR-2026-1099", "ns", [
            {"fixture_id": "decoy", "content": "Reference ERR-2026-1001", "score": 0.99, "rank": 1}], sidecar)
        self.assertEqual(positive["decision"], "answer")
        self.assertEqual(negative["decision"], "abstain")
        cases = [
            {"case_id": "p", "expected_safety": "answer", "category": "exact_identifier",
             "expected_fixture_ids": ["good"], "forbidden_fixture_ids": [], "expected_answer": "ERR-2026-1001", "namespace": "ns"},
            {"case_id": "n", "expected_safety": "abstain", "category": "negative_confusable",
             "expected_fixture_ids": [], "forbidden_fixture_ids": ["decoy"], "expected_answer": "", "namespace": "ns"},
        ]
        raw = {"p": [{"fixture_id": "good", "content": "Reference ERR-2026-1001", "score": 0.91, "rank": 1}],
               "n": [{"fixture_id": "decoy", "content": "Reference ERR-2026-1001", "score": 0.99, "rank": 1}]}
        metrics = score_predictions(cases, raw, {"p": positive, "n": negative}, sidecar)
        self.assertEqual(metrics["negative_false_positive_rate_at_k"], 0.0)
        self.assertEqual(metrics["recall_at_5"], 1.0)
        # A single positive category cannot satisfy the unchanged full C8.7 gates.
        self.assertFalse(policy_feasible({**MIN_GATES, **MAX_GATES, **metrics}))
        self.assertEqual("SYNTHETIC_QUALITY_FAIL" if not policy_feasible({**MIN_GATES, **MAX_GATES, **metrics}) else "SYNTHETIC_QUALITY_PASS",
                         "SYNTHETIC_QUALITY_FAIL")

    def test_all_abstain_with_perfect_raw_retrieval_fails_selected_output_gates(self) -> None:
        categories = ("exact_identifier", "active_decision", "hard_constraint", "supersession", "namespace_isolation")
        cases, raw, predictions, metadata = [], {}, {}, {}
        for index, category in enumerate(categories):
            case_id, fixture_id, answer = f"p{index}", f"gold{index}", f"answer-{index}"
            cases.append({"case_id": case_id, "expected_safety": "answer", "category": category,
                          "expected_fixture_ids": [fixture_id], "forbidden_fixture_ids": [],
                          "expected_answer": answer, "namespace": "ns"})
            raw[case_id] = [{"fixture_id": fixture_id, "content": f"evidence {answer}", "score": 0.9, "rank": 1}]
            predictions[case_id] = {"decision": "abstain", "reason": "below_fixed_threshold"}
            metadata[fixture_id] = {"namespace": "ns", "split": "synthetic_development",
                                    "status": "active", "supersedes": []}
        metrics = score_predictions(cases, raw, predictions, metadata)
        self.assertEqual(metrics["raw_retrieval_recall_at_5"], 1.0)
        self.assertEqual(metrics["recall_at_5"], 0.0)
        self.assertTrue(all(metrics[f"raw_retrieval_{category}_recall_at_5"] == 1.0 for category in categories))
        self.assertFalse(policy_feasible(metrics))

    def test_selected_source_and_content_drive_recall_and_all_gates(self) -> None:
        categories = ("exact_identifier", "active_decision", "hard_constraint", "supersession", "namespace_isolation")
        cases, raw, predictions, metadata = [], {}, {}, {}
        for index, category in enumerate(categories):
            case_id, fixture_id, answer = f"p{index}", f"gold{index}", f"answer-{index}"
            cases.append({"case_id": case_id, "expected_safety": "answer", "category": category,
                          "expected_fixture_ids": [fixture_id], "forbidden_fixture_ids": [],
                          "expected_answer": answer, "namespace": "ns"})
            raw[case_id] = [{"fixture_id": fixture_id, "content": f"evidence {answer}", "score": 0.9, "rank": 1}]
            predictions[case_id] = {"decision": "answer", "fixture_id": fixture_id}
            metadata[fixture_id] = {"namespace": "ns", "split": "synthetic_development",
                                    "status": "active", "supersedes": []}
        cases.append({"case_id": "negative", "expected_safety": "abstain", "category": "negative_confusable",
                      "expected_fixture_ids": [], "forbidden_fixture_ids": [], "expected_answer": "", "namespace": "ns"})
        raw["negative"] = []
        predictions["negative"] = {"decision": "abstain"}
        metrics = score_predictions(cases, raw, predictions, metadata)
        self.assertTrue(policy_feasible(metrics))
        self.assertEqual(metrics["recall_at_5"], 1.0)
        self.assertEqual(metrics["frozen_policy_correct_source_count"], 5)

        wrong_source_raw = {key: list(rows) for key, rows in raw.items()}
        wrong_source_raw["p0"].append({"fixture_id": "wrong", "content": "unrelated", "score": 0.8, "rank": 2})
        metadata["wrong"] = {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": []}
        wrong_source = dict(predictions, p0={"decision": "answer", "fixture_id": "wrong"})
        wrong_source_metrics = score_predictions(cases, wrong_source_raw, wrong_source, metadata)
        self.assertEqual(wrong_source_metrics["recall_at_5"], 0.8)
        self.assertEqual(wrong_source_metrics["exact_identifier_recall_at_5"], 0.0)
        self.assertFalse(policy_feasible(wrong_source_metrics))

        wrong_content_raw = {key: list(rows) for key, rows in raw.items()}
        wrong_content_raw["p0"] = [{"fixture_id": "gold0", "content": "different content", "score": 0.9, "rank": 1}]
        wrong_content_metrics = score_predictions(cases, wrong_content_raw, predictions, metadata)
        self.assertEqual(wrong_content_metrics["recall_at_5"], 0.8)
        self.assertFalse(policy_feasible(wrong_content_metrics))

    def test_missing_gate_metric_fails_closed(self) -> None:
        complete = {**MIN_GATES, **MAX_GATES}
        self.assertTrue(policy_feasible(complete))
        del complete["raw_unmapped_candidate_count"]
        self.assertFalse(policy_feasible(complete))
        self.assertFalse(policy_feasible({**MIN_GATES, **MAX_GATES, "recall_at_5": float("nan")}))

    def test_unscoped_single_candidate_abstains_but_explicit_scope_answers(self) -> None:
        sidecar = {"only": {"namespace": "ns", "split": "synthetic_development",
                             "status": "active", "supersedes": []}}
        candidate = [{"fixture_id": "only", "content": "Reference ERR-2026-1001",
                      "score": 0.99, "rank": 1}]
        unscoped = frozen_decision("Find ERR-2026-1001", None, candidate, sidecar)
        scoped = frozen_decision("Find ERR-2026-1001", "ns", candidate, sidecar)
        self.assertEqual(unscoped["decision"], "abstain")
        self.assertEqual(unscoped["reason"], "missing_request_namespace")
        self.assertEqual(scoped["decision"], "answer")

    def test_frozen_regex_and_tie_behavior_matches_prototype_boundaries(self) -> None:
        reference_id_re = re.compile(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+){2,}\b", re.I)
        reference_number_re = re.compile(r"\b\d+\b")
        samples = ("ORC-137-VQ", "_ORC-137-VQ_", "ITEM2", "ITEM-2", "ITEM 2",
                   "MÉR-١٢-X", "K-1-A", "CODE-१२-XX")
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(_POLICY_ID_RE.findall(sample), reference_id_re.findall(sample))
                self.assertEqual(_POLICY_NUMBER_RE.findall(sample), reference_number_re.findall(sample))
        self.assertEqual(_POLICY_ID_RE.findall("_ORC-137-VQ_"), [])
        self.assertEqual(_POLICY_ID_RE.findall("ITEM2"), [])
        self.assertEqual(_POLICY_NUMBER_RE.findall("ITEM2"), [])
        self.assertEqual(_POLICY_NUMBER_RE.findall("ITEM ٢"), ["٢"])
        sidecar = {fid: {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": []}
                   for fid in ("underscore", "item")}
        underscore = frozen_resolve("_ORC-137-VQ_", "ns", [
            {"fixture_id": "underscore", "content": "_ORC-137-VQ_", "score": 0.8, "rank": 1}], sidecar)
        item = frozen_resolve("ITEM2", "ns", [
            {"fixture_id": "item", "content": "ITEM2", "score": 0.8, "rank": 1}], sidecar)
        self.assertIsNotNone(underscore)
        self.assertIsNotNone(item)
        assert underscore is not None and item is not None
        self.assertEqual(underscore["method"], "content_anchor")
        self.assertEqual(item["method"], "content_anchor")

        tie_sidecar = {fid: dict(sidecar["item"]) for fid in ("first", "second")}
        tied = frozen_resolve("alpha replacement", "ns", [
            {"fixture_id": "first", "content": "alpha replacement", "score": 0.8, "rank": 1},
            {"fixture_id": "second", "content": "alpha replacement", "score": 0.8, "rank": 2},
        ], tie_sidecar)
        self.assertIsNone(tied)

    def test_sidecar_detects_omitted_alternative_successor_but_limits_scope(self) -> None:
        metadata = {
            "old": {"namespace": "ns", "split": "synthetic_development", "status": "superseded", "supersedes": []},
            "current": {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": ["old"]},
            "alternate": {"namespace": "ns", "split": "synthetic_development", "status": "active", "supersedes": ["old"]},
            "unrelated-bad": {"namespace": "ns", "split": "synthetic_development", "status": None, "supersedes": []},
        }
        candidate = [{"fixture_id": "current", "content": "Reference ERR-2026-1001", "score": 0.9, "rank": 1}]
        sidecar = _sidecar_for_candidates(candidate, metadata)
        self.assertEqual(set(sidecar), {"old", "current", "alternate"})
        result = frozen_decision("Find ERR-2026-1001", "ns", candidate, sidecar)
        self.assertEqual(result["decision"], "abstain")
        self.assertNotIn("unrelated-bad", sidecar)

    def test_frozen_threshold_and_original_c87_gate_values_are_unchanged(self) -> None:
        self.assertEqual(FROZEN_THRESHOLD, 0.99)
        self.assertEqual(MIN_GATES, {"recall_at_5": 0.90, "exact_identifier_recall_at_5": 1.0,
                                     "active_decision_recall_at_5": 0.90, "hard_constraint_recall_at_5": 1.0,
                                     "supersession_recall_at_5": 0.90, "namespace_isolation_recall_at_5": 0.90})
        self.assertEqual(MAX_GATES["negative_false_positive_rate_at_k"], 0.0)
        self.assertEqual(MAX_GATES["raw_unmapped_candidate_count"], 0)
        self.assertFalse(policy_feasible({**MIN_GATES, **MAX_GATES, "recall_at_5": 0.0}))

    def test_generator_has_no_scratch_or_holdout_side_effect(self) -> None:
        source = Path(__file__).with_name("c87_fixture_generator.py").read_text(encoding="utf-8")
        self.assertNotIn("c87-independent-validation-20261111", source)
        dataset = build_development_dataset_v2()
        self.assertFalse(dataset["sealed_holdout"])
        self.assertTrue(all(row["split"] == "synthetic_development" for row in dataset["cases"] + dataset["records"]))

    def test_development_validator_rejects_weak_answer_and_unrelated_query(self) -> None:
        from copy import deepcopy
        from c87_fixture_generator import validate_development_dataset_v2

        weak = deepcopy(build_development_dataset_v2())
        weak["cases"][0]["expected_answer"] = "item"
        with self.assertRaisesRegex(ValueError, "weak answer|weak exact identifier|gold labels"):
            validate_development_dataset_v2(weak)

        unrelated = deepcopy(build_development_dataset_v2())
        unrelated["cases"][0]["query"] = "Explain unrelated maintenance policy in namespace aurora_ops."
        with self.assertRaisesRegex(ValueError, "group-specific evidence anchor|query/language/gold labels"):
            validate_development_dataset_v2(unrelated)

    def test_development_validator_rejects_duplicate_active_source_at_fixed_count(self) -> None:
        from copy import deepcopy
        from c87_fixture_generator import validate_development_dataset_v2

        dataset = deepcopy(build_development_dataset_v2())
        source = next(row for row in dataset["records"] if row["fixture_id"] == "D-ID-01")
        duplicate_slot = next(row for row in dataset["records"] if row["fixture_id"] == "D-ABS-01")
        duplicate_slot["content"] = source["content"]
        self.assertEqual(len(dataset["records"]), 160)
        with self.assertRaisesRegex(ValueError, "ambiguous duplicate active evidence"):
            validate_development_dataset_v2(dataset)

    def test_development_validator_rejects_bilingual_label_query_swap(self) -> None:
        from copy import deepcopy
        from c87_fixture_generator import validate_development_dataset_v2

        dataset = deepcopy(build_development_dataset_v2())
        case = next(row for row in dataset["cases"] if row["group"] == "bilingual_semantic_transfer"
                    and row["language"] == "id")
        case["language"] = "en"
        case["query"] = "Where is Osprey marker 2 located in namespace cinder_archive?"
        with self.assertRaisesRegex(ValueError, "bilingual query/label language mismatch|query/language/gold labels"):
            validate_development_dataset_v2(dataset)

    def test_development_validator_rejects_impossible_conflicting_labels(self) -> None:
        from copy import deepcopy
        from c87_fixture_generator import validate_development_dataset_v2

        dataset = deepcopy(build_development_dataset_v2())
        case = next(row for row in dataset["cases"] if row["group"] == "conflicting_state")
        case["expected_safety"] = "answer"
        with self.assertRaisesRegex(ValueError, "impossible safety/category label"):
            validate_development_dataset_v2(dataset)

    def test_runtime_backend_report_does_not_overclaim_model_or_dimension(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        report_source = source.split("class C87Tests", 1)[0]
        self.assertNotIn('"backend": "bge-onnx-local"', report_source)
        self.assertIn('"requested": "local embedder"', report_source)
        self.assertIn('"runtime_model": "not attested"', source)
        self.assertIn('"runtime_dimension": "not attested"', source)
        self.assertIn('"model_identity_verified": False', source)
        self.assertIn('"dimension_verified": False', source)
        self.assertIn('"vector_indexed": vector_indexed', source)
        self.assertIn('"semantic_truth_attested": False', source)
        self.assertIn('"human_sample_review_required": True', source)
        self.assertIn('"answer_generation": False', source)

    def test_store_response_continues_only_for_canonical_hash_reason(self) -> None:
        dataset = build_development_dataset_v2()
        hash_ids = _validated_hash_fixture_ids(dataset)
        hash_record = next(row for row in dataset["records"] if row["fixture_id"] == "D-HASH-01")
        plain_record = next(row for row in dataset["records"] if row["fixture_id"] == "D-PATH-01")
        self.assertEqual(_classify_store_response(hash_record,
                         {"status": 403, "body": {"error": "content blocked", "reason": "AWS Secret Key"}},
                         hash_ids), (None, "AWS Secret Key"))
        for response in (
            {"status": 403, "body": {"reason": "unexpected-provider-secret"}},
            {"status": 403, "body": "not-json"},
        ):
            with self.assertRaises(RuntimeError) as caught:
                _classify_store_response(hash_record, response, hash_ids)
            self.assertNotIn("unexpected-provider-secret", str(caught.exception))
            self.assertNotIn(hash_record["content"], str(caught.exception))
        with self.assertRaisesRegex(RuntimeError, "provider reason GitHub Token"):
            _classify_store_response(hash_record, {"status": 403, "body": {"reason": "GitHub Token"}}, hash_ids)
        with self.assertRaisesRegex(RuntimeError, "non-hash fixture.*provider reason AWS Secret Key"):
            _classify_store_response(plain_record, {"status": 403, "body": {"reason": "AWS Secret Key"}}, hash_ids)
        for status in (400, 401, 404, 429, 500, 503):
            with self.subTest(status=status), self.assertRaisesRegex(RuntimeError, f"HTTP {status}"):
                _classify_store_response(hash_record, {"status": status, "body": {"raw": hash_record["content"]}}, hash_ids)
        with self.assertRaisesRegex(RuntimeError, "status malformed"):
            _classify_store_response(hash_record, {"status": True, "body": {}}, hash_ids)
        with self.assertRaisesRegex(RuntimeError, "malformed id"):
            _classify_store_response(hash_record, {"status": 201, "body": {"id": {"unexpected": "object"}}}, hash_ids)

    def test_partial_transport_runs_all_120_cases_and_keeps_original_denominators(self) -> None:
        dataset = build_development_dataset_v2()
        records = {row["fixture_id"]: row for row in dataset["records"]}
        cases_by_query = {row["query"]: row for row in dataset["cases"]}
        fixture_to_memory: dict[str, str] = {}
        memory_to_fixture: dict[str, str] = {}
        query_payloads: list[dict[str, Any]] = []
        blocked_ids = {f"D-HASH-{index:02d}" for index in range(1, 11)}

        def fake_request(method: str, path: str, payload: Any) -> dict[str, Any]:
            if path == "/v1/memories":
                row = next(record for record in records.values() if record["content"] == payload["content"])
                if row["fixture_id"] in blocked_ids:
                    return {"status": 403, "body": {"error": "content blocked", "reason": "AWS Secret Key"}}
                memory_id = f"mem-{row['fixture_id']}"
                fixture_to_memory[row["fixture_id"]] = memory_id
                memory_to_fixture[memory_id] = row["fixture_id"]
                return {"status": 201, "body": {"id": memory_id}}
            if path == "/v1/stats":
                return {"status": 200, "body": {"store": {"vector_indexed": len(fixture_to_memory)}}}
            if path == "/v1/recall":
                query_payloads.append(payload)
                case = cases_by_query[payload["query"]]
                expected = case["expected_fixture_ids"]
                if case["expected_safety"] == "answer" and expected and expected[0] not in blocked_ids:
                    fid = expected[0]
                    return {"status": 200, "body": {"results": [{
                        "id": fixture_to_memory[fid], "score": 0.99,
                        "content": records[fid]["content"], "rank": 1,
                    }]}}
                return {"status": 200, "body": {"results": []}}
            raise AssertionError(f"unexpected transport path: {path}")

        report = evaluate_api_workflow(dataset, fake_request)
        self.assertEqual(report["process_status"], "COMPLETED")
        self.assertEqual(report["ingestion_status"], "PARTIAL")
        self.assertEqual(report["quality_status"], "FAIL")
        self.assertFalse(report["quality_gates_pass"])
        self.assertEqual(report["ingestion"]["requested_count"], 160)
        self.assertEqual(set(report["ingestion"]["requested_fixture_ids"]), set(records))
        self.assertEqual(report["ingestion"]["accepted_count"], 150)
        self.assertEqual(set(report["ingestion"]["accepted_fixture_ids"]), set(records) - blocked_ids)
        self.assertEqual(report["ingestion"]["blocked_count"], 10)
        self.assertEqual(report["ingestion"]["vector_indexed"], 150)
        self.assertTrue(all(set(item) == {"fixture_id", "status", "reason"}
                            for item in report["ingestion"]["blocked_records"]))
        self.assertEqual(set(report["filter_check_diagnostics"]["blocked_fixture_ids"]), blocked_ids)
        self.assertEqual(set(report["filter_check_diagnostics"]["provider_reasons"]), {"AWS Secret Key"})
        self.assertEqual(report["queries_run"], 120)
        self.assertEqual(len(query_payloads), 120)
        self.assertTrue(all(set(payload) <= {"query", "namespace", "limit", "min_score"} for payload in query_payloads))
        query_set = {payload["query"] for payload in query_payloads}
        self.assertTrue(all(case["query"] in query_set for case in dataset["cases"]
                            if case.get("group") == "exact_hash"))
        metrics = report["quality_metrics"]
        self.assertEqual(report["fixture_counts"], {"synthetic_development_cases": 120, "synthetic_development_records": 160})
        self.assertEqual(metrics["positive_case_count"] + metrics["negative_case_count"], 120)
        self.assertEqual(metrics["positive_case_count"], 70)
        self.assertEqual(metrics["negative_case_count"], 50)
        self.assertEqual(metrics["hard_constraint_recall_at_5"], 0.0)
        hash_case = next(row for row in dataset["cases"] if row["expected_fixture_ids"] == ["D-HASH-01"])
        self.assertEqual(report["predictions"][hash_case["case_id"]]["decision"], "abstain")
        self.assertEqual(report["retrieval_results"][hash_case["case_id"]], [])
        self.assertNotIn(records["D-HASH-01"]["content"], json.dumps(report))
        self.assertNotIn("sha256", json.dumps(report["ingestion"]))

    def test_any_rejection_forces_quality_failure_even_with_perfect_metrics(self) -> None:
        perfect_metrics = {**MIN_GATES, **MAX_GATES, "raw_retrieval_recall_at_5": 1.0}
        metrics_pass, coverage_complete, quality_pass = quality_outcome(perfect_metrics, 160, 159, 1)
        self.assertTrue(metrics_pass)
        self.assertFalse(coverage_complete)
        self.assertFalse(quality_pass)
        self.assertEqual(quality_outcome(perfect_metrics, 160, 160, 0), (True, True, True))

    def test_fake_transport_fails_closed_for_index_mismatch_and_duplicate_ids(self) -> None:
        dataset = build_development_dataset_v2()

        def wrong_index(method: str, path: str, payload: Any) -> dict[str, Any]:
            if path == "/v1/memories":
                return {"status": 201, "body": {"id": "same-id"}}
            if path == "/v1/stats":
                return {"status": 200, "body": {"store": {"vector_indexed": 1}}}
            raise AssertionError("recalls must not begin after failed indexing")

        with self.assertRaisesRegex(RuntimeError, "duplicate memory ID"):
            evaluate_api_workflow(dataset, wrong_index)

        memory_counter = 0
        def under_indexed(method: str, path: str, payload: Any) -> dict[str, Any]:
            nonlocal memory_counter
            if path == "/v1/memories":
                memory_counter += 1
                return {"status": 201, "body": {"id": f"id-{memory_counter}"}}
            if path == "/v1/stats":
                return {"status": 200, "body": {"store": {"vector_indexed": 0}}}
            raise AssertionError("recalls must not begin after failed indexing")

        with self.assertRaisesRegex(RuntimeError, "vector index count mismatch"):
            evaluate_api_workflow(dataset, under_indexed)

        accepted = 0
        def unexpected_recall_id(method: str, path: str, payload: Any) -> dict[str, Any]:
            nonlocal accepted
            if path == "/v1/memories":
                accepted += 1
                return {"status": 201, "body": {"id": f"id-{accepted}"}}
            if path == "/v1/stats":
                return {"status": 200, "body": {"store": {"vector_indexed": accepted}}}
            return {"status": 200, "body": {"results": [{"id": "not-stored-here", "score": 0.9,
                                                             "content": "candidate text", "rank": 1}]}}

        with self.assertRaisesRegex(RuntimeError, "unexpected memory IDs"):
            evaluate_api_workflow(dataset, unexpected_recall_id)

    def test_partial_corpus_report_marks_metrics_noncomparable(self) -> None:
        partial = retrieval_comparability(False)
        self.assertEqual(partial["status"], "PARTIAL_CORPUS_DIAGNOSTIC_ONLY")
        self.assertEqual(
            partial["caveat"],
            "retrieval metrics are diagnostic on a partial corpus; top-K/ranking and negatives may differ "
            "from full 160-record corpus; not comparable to a full-corpus run; blocked hash cases remain "
            "in denominator and quality cannot pass",
        )
        self.assertEqual(retrieval_comparability(True), {"status": "FULL_CORPUS", "caveat": None})

    def test_cli_exit_statuses_and_report_output(self) -> None:
        base_report = {
            "quality_gates_pass": True,
            "quality_status": "PASS",
            "coverage_complete": True,
            "ingestion_status": "FULL",
            "quality_metrics": {"recall_at_5": 1.0},
        }

        def invoke(report: dict[str, Any] | None = None,
                   error: Exception | None = None) -> tuple[int, str, str]:
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(sys, "argv", ["c87_corrected_calibration.py", "--mode", "e2e",
                                             "--fixtures", str(Path(__file__))]):
                with patch(__name__ + ".run_e2e", side_effect=error if error else None,
                           return_value=report or base_report):
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        exit_code = main()
            return exit_code, stdout.getvalue(), stderr.getvalue()

        pass_code, pass_stdout, pass_stderr = invoke(base_report)
        self.assertEqual(pass_code, 0)
        self.assertEqual(json.loads(pass_stdout), base_report)
        self.assertEqual(pass_stderr, "")

        failed_report = {
            "quality_gates_pass": False,
            "quality_status": "FAIL",
            "coverage_complete": False,
            "ingestion_status": "PARTIAL",
            "quality_metrics": {"recall_at_5": 0.5},
        }
        fail_code, fail_stdout, fail_stderr = invoke(failed_report)
        self.assertEqual(fail_code, 2)
        serialized_report = json.dumps(failed_report, indent=2, sort_keys=True) + "\n"
        self.assertTrue(fail_stdout.startswith(serialized_report))
        self.assertEqual(json.loads(serialized_report), failed_report)
        self.assertIn("::error::C8.7 QUALITY/COVERAGE FAILURE", fail_stdout)
        self.assertIn("quality_status=FAIL", fail_stdout)
        self.assertIn("ingestion_status=PARTIAL", fail_stdout)
        self.assertEqual(fail_stderr, "")

        infra_code, infra_stdout, infra_stderr = invoke(error=RuntimeError("simulated infrastructure crash"))
        self.assertEqual(infra_code, 1)
        self.assertEqual(infra_stdout, "")
        self.assertIn("C8.7 INFRASTRUCTURE_FAILURE: simulated infrastructure crash", infra_stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "synthetic-self-test", "e2e"), required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--image", default=IMAGE_C87)
    parser.add_argument("--container", default="nyawa-c87-eval")
    args = parser.parse_args()
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C87Tests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.mode == "synthetic-self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C87DevelopmentV2Tests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.mode == "e2e":
        if args.fixtures is None or not args.fixtures.is_file():
            print("error: --fixtures is required for e2e mode", file=sys.stderr)
            return 1
        try:
            report = run_e2e(args.container, args.image, args.fixtures)
        except Exception as error:
            print(f"C8.7 INFRASTRUCTURE_FAILURE: {error}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["quality_gates_pass"]:
            print(
                "::error::C8.7 QUALITY/COVERAGE FAILURE: quality gates or corpus coverage failed; "
                f"quality_status={report.get('quality_status')} "
                f"ingestion_status={report.get('ingestion_status')}. Full JSON report is printed above."
            )
            return 2
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

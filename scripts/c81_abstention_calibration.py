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

from c81_fixture_generator import SCHEMA, build_dataset, validate_dataset

IMAGE = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:61da25caf922c0925f4fd8e3af1d811f6fc9d3a9df21beb44138c7300d47efb6"
)
LABEL_KEY = "org.nyawa.c81-abstention-calibration"
MEMORY_LIMIT = 4 * 1024**3
MIN_AVAILABLE_MEMORY = 5 * 1024**3
FIXTURE_DEFAULT = Path(__file__).resolve().parents[1] / "tests/fixtures/c81-abstention-calibration-fixtures.json"
SIGNALS = (
    "score", "rrf_score", "final_margin", "rrf_margin", "lexical_coverage", "lexical_margin",
    "vector_rank", "fts5_rank", "both_modalities", "exact_identifier_match",
)
LEXICAL_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it of on or the to was what when where which who why with did does do this that these those find give tell me fictional synthetic fixture marker c8 c81 synth id dan yang apa berapa untuk dari pada ini itu adalah sebagai dengan bagaimana siapa kapan dimana mengapa antrean kebijakan mode berapa sebuah sebuahnya".split()
)
SYNTHETIC_ID = re.compile(r"SYNTH-C81-ID-[0-9]{4}", re.IGNORECASE)

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
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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


def _exact_synthetic_identifier_match(query: str, content: str) -> bool:
    query_ids = {value.upper() for value in SYNTHETIC_ID.findall(query)}
    return bool(query_ids & {value.upper() for value in SYNTHETIC_ID.findall(content)})


def _margin(candidates: list[dict[str, Any]], field: str) -> float | None:
    values = [candidate[field] for candidate in candidates if _numeric(candidate.get(field))]
    if not values:
        return None
    values.sort(reverse=True)
    return float(values[0] - values[1]) if len(values) > 1 else float(values[0])


def score_case(
    case: dict[str, Any],
    items: list[dict[str, Any]],
    latency_ms: float,
    k: int,
    memory_to_fixture: dict[str, str],
) -> dict[str, Any]:
    relevant = case["relevant_markers"]
    forbidden = case["forbidden_markers"]
    candidates: list[dict[str, Any]] = []
    chars = 0
    numeric = True
    modality_numeric_all = True
    for rank, item in enumerate(items[:k], start=1):
        content = _content(item)
        score = _field(item, "Score", "score")
        rrf_score = _field(item, "RRFScore", "rrf_score")
        vector_rank = _field(item, "VectorRank", "vector_rank")
        fts5_rank = _field(item, "FTS5Rank", "fts5_rank")
        numeric = numeric and _numeric(score) and _numeric(rrf_score)
        modality_numeric = (
            isinstance(vector_rank, int) and not isinstance(vector_rank, bool) and vector_rank >= 0
            and isinstance(fts5_rank, int) and not isinstance(fts5_rank, bool) and fts5_rank >= 0
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
            "modality_evidence_present": modality_numeric,
            "both_modalities": bool(modality_numeric and vector_rank > 0 and fts5_rank > 0),
            "lexical_coverage": _lexical_coverage(case["query"], content),
            "exact_identifier_match": _exact_synthetic_identifier_match(case["query"], content),
            "type": str(item.get("Type", item.get("type", ""))),
            "relevant": bool(marker_hits),
            "forbidden_markers_found": forbidden_hits,
        }
        candidates.append(candidate)
        chars += len(content)
    relevant_candidates = [candidate for candidate in candidates if candidate["relevant"]]
    first_relevant_rank = relevant_candidates[0]["rank"] if relevant_candidates else None
    forbidden_hits = [marker for candidate in candidates for marker in candidate["forbidden_markers_found"]]
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
        "fixture_ids_mapped": all(candidate["fixture_id"] != "unknown-fixture" for candidate in candidates),
        "latency_ms": round(float(latency_ms), 2),
        "candidate_chars": chars,
        "estimated_candidate_tokens": math.ceil(chars / 4),
        "final_margin": _margin(candidates, "score"),
        "rrf_margin": _margin(candidates, "rrf_score"),
        "lexical_margin": _margin(candidates, "lexical_coverage"),
        "forbidden_markers_found": forbidden_hits,
        "candidates": candidates,
    }


def evaluate_policy(
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    signal: str | None = None,
    threshold: float | None = None,
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
        if signal in ("final_margin", "rrf_margin", "lexical_margin"):
            margin = row.get(signal)
            accepted = candidates if threshold is not None and _numeric(margin) and margin >= threshold else []
        elif signal == "both_modalities":
            accepted = [candidate for candidate in candidates if _both_modalities_present(candidate)]
        elif signal == "exact_identifier_match":
            accepted = [candidate for candidate in candidates if candidate.get("exact_identifier_match", False)]
        elif signal in ("vector_rank", "fts5_rank"):
            accepted = [candidate for candidate in candidates if _rank_within(candidate, signal, threshold)]
        elif signal is not None:
            accepted = [
                candidate for candidate in candidates
                if threshold is not None and _numeric(candidate.get(signal)) and candidate[signal] >= threshold
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
    )


def threshold_values(rows: list[dict[str, Any]], signal: str) -> list[float]:
    if signal == "both_modalities":
        return [1.0] if any(row.get("modality_evidence_present") for row in rows) else []
    if signal == "exact_identifier_match":
        return [1.0]
    values: set[float] = set()
    if signal in ("final_margin", "rrf_margin", "lexical_margin"):
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
    ordered.append(math.nextafter(ordered[-1], math.inf))
    return ordered


def calibrate_signal(
    dataset: dict[str, Any],
    cases: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    signal: str,
) -> dict[str, Any]:
    gates = dataset["gates"]
    points = []
    for threshold in threshold_values(rows, signal):
        metrics = evaluate_policy(cases, rows, signal, threshold)["metrics"]
        point = {"threshold": threshold, **metrics, "feasible": passes_gates(metrics, gates)}
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


def summarize(dataset: dict[str, Any], case_results: list[dict[str, Any]]) -> dict[str, Any]:
    cases = dataset["cases"]
    gates = dataset["gates"]
    calibration_cases = [case for case in cases if case["split"] == "calibration"]
    holdout_cases = [case for case in cases if case["split"] == "holdout"]
    by_id = {row["case_id"]: row for row in case_results}
    calibration_rows = [by_id[case["case_id"]] for case in calibration_cases]
    holdout_rows = [by_id[case["case_id"]] for case in holdout_cases]
    calibration_raw = evaluate_policy(calibration_cases, calibration_rows)["metrics"]
    holdout_raw = evaluate_policy(holdout_cases, holdout_rows)["metrics"]
    sweeps = [calibrate_signal(dataset, calibration_cases, calibration_rows, signal) for signal in SIGNALS]
    feasible_candidates = [
        {"signal": sweep["signal"], **sweep["best_feasible"]}
        for sweep in sweeps if sweep["best_feasible"] is not None
    ]
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
    holdout_pass = False
    if selected:
        holdout_policy_metrics = evaluate_policy(
            holdout_cases, holdout_rows, selected["signal"], selected["threshold"]
        )["metrics"]
        holdout_pass = passes_gates(holdout_policy_metrics, gates)
    all_numeric = all(row["scores_numeric"] for row in case_results)
    fixture_ids_mapped = all(row.get("fixture_ids_mapped", True) for row in case_results)
    returned_candidates = [candidate for row in case_results for candidate in row["candidates"]]
    modality_evidence_available = bool(returned_candidates) and all(candidate.get("modality_evidence_present", False) for candidate in returned_candidates)
    candidate_rows = [row for row in case_results if row.get("result_count", len(row.get("candidates", []))) > 0]
    modality_ranks_numeric = bool(candidate_rows) and all(row.get("modality_ranks_numeric", False) for row in candidate_rows)
    source_counts = {"vector_only": 0, "fts5_only": 0, "both": 0, "neither": 0}
    if modality_evidence_available:
        for candidate in returned_candidates:
            vector_hit = candidate["vector_rank"] > 0
            fts5_hit = candidate["fts5_rank"] > 0
            key = "both" if vector_hit and fts5_hit else "vector_only" if vector_hit else "fts5_only" if fts5_hit else "neither"
            source_counts[key] += 1
    status = "PASS" if selected and holdout_pass and all_numeric and fixture_ids_mapped else "FAIL"
    latencies = sorted(row["latency_ms"] for row in case_results)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0
    return {
        "status": status,
        "schema_version": SCHEMA,
        "protocol": "calibration-only threshold selection, frozen holdout evaluation",
        "gates": gates,
        "raw_retrieval": {
            "calibration": calibration_raw,
            "holdout": holdout_raw,
        },
        "threshold_sweeps": sweeps,
        "selected_policy": selected,
        "holdout_policy_metrics": holdout_policy_metrics,
        "holdout_gates_pass": holdout_pass,
        "numeric_score_and_rrf_fields": all_numeric,
        "fixture_ids_mapped": fixture_ids_mapped,
        "modality_evidence_available": modality_evidence_available,
        "modality_ranks_numeric": modality_ranks_numeric,
        "source_counts": source_counts,
        "latency_median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
        "candidate_chars_total": sum(row["candidate_chars"] for row in case_results),
        "estimated_candidate_tokens_total": sum(row["estimated_candidate_tokens"] for row in case_results),
        "case_results": case_results,
        "provider_called": False,
        "hermes_called": False,
        "context_injected": False,
    }


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
        raise RuntimeError("C8.1 container image/ownership mismatch")
    if host.get("NetworkMode") != "none" or not host.get("ReadonlyRootfs"):
        raise RuntimeError("C8.1 container is not network=none/read-only")
    if host.get("PortBindings") or host.get("Binds") or host.get("Mounts"):
        raise RuntimeError("C8.1 container unexpectedly has a port or host/volume mount")
    if host.get("Memory") != MEMORY_LIMIT or host.get("MemorySwap") != MEMORY_LIMIT or host.get("PidsLimit") != 128:
        raise RuntimeError("C8.1 container resource limits do not match the declared bounds")
    if any(mount.get("Type") != "tmpfs" for mount in inspected.get("Mounts", [])):
        raise RuntimeError("C8.1 container has a non-tmpfs mount")


def cleanup(container: str, image: str) -> None:
    result = docker(["inspect", container], timeout=15, check=False)
    if result.returncode:
        return
    inspected = json.loads(result.stdout)[0]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if labels.get(LABEL_KEY) != "true" or inspected["Config"].get("Image") != image:
        raise RuntimeError("refusing to remove a container not created by this C8.1 run")
    docker(["rm", "--force", container], timeout=30)
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("C8.1 cleanup readback failed: container still exists")


def run_e2e(container: str, image: str, fixture_path: Path) -> dict[str, Any]:
    if image != IMAGE:
        raise RuntimeError("image must equal the approved immutable BGE r1 digest")
    dataset = load_dataset(fixture_path)
    preflight()
    docker(["pull", image], timeout=300)
    image_info = json.loads(docker(["image", "inspect", image]).stdout)[0]
    if image not in image_info.get("RepoDigests", []):
        raise RuntimeError("pulled image digest does not match the pinned image")
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("container name already exists; refusing to replace it")
    try:
        docker([
            "run", "--detach", "--name", container,
            "--label", f"{LABEL_KEY}=true",
            "--network=none", "--read-only",
            "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
            "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
            "--restart=no", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
            "--env", "MEMORY_EMBEDDER=local",
            "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model",
            "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
            image, "serve", "/scratch/c81-abstention.db",
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
        for record in dataset["records"]:
            response = inside_http(container, "POST", "/v1/memories", {
                "content": record["content"], "namespace": record["namespace"], "type": record["type"]
            })
            if response["status"] != 201 or not response["body"].get("id"):
                raise RuntimeError(f"synthetic insert failed for {record['fixture_id']}: HTTP {response['status']}")
            inserted[record["fixture_id"]] = str(response["body"]["id"])
        reverse_ids = {memory_id: fixture_id for fixture_id, memory_id in inserted.items()}
        stats = inside_http(container, "GET", "/v1/stats")
        if stats["status"] != 200:
            raise RuntimeError(f"stats returned HTTP {stats['status']}")
        store_stats = stats["body"].get("store", stats["body"])
        vector_indexed = int(store_stats.get("vector_indexed", 0))
        if vector_indexed < len(dataset["records"]):
            raise RuntimeError(f"not all C8.1 synthetic records were vector-indexed: {store_stats}")

        per_case = []
        for case in dataset["cases"]:
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
            per_case.append(score_case(case, items, elapsed, int(dataset["k"]), reverse_ids))
        report = summarize(dataset, per_case)
        report.update({
            "image": image,
            "runtime_version": root_body["version"],
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "synthetic_records_inserted": len(inserted),
            "vector_indexed": vector_indexed,
            "queries_run": len(dataset["cases"]),
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
        report = summarize(dataset, rows)
        self.assertEqual(report["status"], "PASS")
        self.assertIsNotNone(report["selected_policy"])
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
    parser.add_argument("--container", default="nyawa-c81-abstention-calibration")
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
            print(f"C8.1 fixture validation FAILED: {error}", file=sys.stderr)
            return 1
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C81CalibrationTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        report = run_e2e(args.container, args.image, args.fixtures)
        print(json.dumps(report, indent=2, sort_keys=True))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("## C8.1 Abstention Calibration\n\n```json\n")
                stream.write(json.dumps(report, indent=2, sort_keys=True))
                stream.write("\n```\n\nSynthetic-only; calibration selects policy, holdout remains frozen. No provider/Hermes call or context injection.\n")
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        print(f"C8.1 evaluation FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

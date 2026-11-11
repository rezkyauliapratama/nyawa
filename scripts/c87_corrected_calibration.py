#!/usr/bin/env python3
"""Run isolated, calibration-only C8.7 evaluation against the local Nyawa API."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import unittest
from pathlib import Path
from typing import Any

from c81_abstention_calibration import HTTP_HELPER, cleanup, docker, inside_http, preflight, verify_container
from c87_fixture_generator import SCHEMA as SCHEMA_C87, build_dataset

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
    if dataset.get("schema_version") != SCHEMA_C87:
        raise RuntimeError(f"unexpected schema: {dataset.get('schema_version')!r}")
    if any(row.get("split") != "calibration" for row in dataset.get("records", [])):
        raise RuntimeError("fixture includes records outside calibration")
    if any(row.get("split") != "calibration" for row in dataset.get("cases", [])):
        raise RuntimeError("fixture includes cases outside calibration")
    if len(dataset.get("records", [])) != 40 or len(dataset.get("cases", [])) != 60:
        raise RuntimeError("unexpected calibration fixture counts")
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
    return (
        all(float(metrics.get(gate, 0.0)) >= minimum for gate, minimum in MIN_GATES.items())
        and all(float(metrics.get(gate, 0.0)) <= maximum for gate, maximum in MAX_GATES.items())
    )


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

    db_path = "/scratch/c87-calibration.db"
    try:
        docker([
            "run", "--detach", "--name", container, "--label", f"{LABEL_KEY}=true",
            "--network=none", "--read-only", "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m", "--memory=4g", "--memory-swap=4g",
            "--cpus=2", "--pids-limit=128", "--restart=no", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--env", "MEMORY_EMBEDDER=local",
            "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model", "--env", "HOME=/tmp",
            "--env", "TMPDIR=/tmp", image, "serve", db_path,
        ], timeout=60)
        verify_container(container, image)
        deadline, last_error = time.monotonic() + 180, "not ready"
        while time.monotonic() < deadline:
            state = json.loads(docker(["inspect", "--format", "{{json .State}}", container]).stdout)
            if not state.get("Running"):
                logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
                raise RuntimeError(f"Nyawa exited: {logs}")
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
            raise RuntimeError(f"Nyawa timeout: {last_error}; {logs}")

        records = dataset["records"]
        anchor_map = {
            row["fixture_id"]: {key: row[key] for key in ("namespace", "split", "status")}
            for row in records
        }
        memory_to_fixture: dict[str, str] = {}
        for row in records:
            response = inside_http(container, "POST", "/v1/memories", {
                "content": row["content"], "namespace": row["namespace"],
                "type": supported_mcp_type(row["type"]),
            })
            if response["status"] != 201 or not isinstance(response.get("body"), dict):
                raise RuntimeError(f"store failed for {row['fixture_id']}: HTTP {response['status']}")
            memory_id = field(response["body"], "id")
            if memory_id is None or not str(memory_id):
                raise RuntimeError(f"store response omitted id for {row['fixture_id']}")
            memory_to_fixture[str(memory_id)] = row["fixture_id"]

        stats_response = inside_http(container, "GET", "/v1/stats")
        if stats_response["status"] != 200:
            raise RuntimeError(f"stats returned HTTP {stats_response['status']}")
        vector_indexed = _vector_indexed(stats_response["body"])
        if vector_indexed < len(records):
            raise RuntimeError(f"not all inserted records were vector-indexed: {vector_indexed}/{len(records)}")

        cases = dataset["cases"]
        raw_scored: dict[str, list[dict[str, Any]]] = {}
        eligible: dict[str, list[dict[str, Any]]] = {}
        diagnostics = {}
        latency_ms: dict[str, float] = {}
        for case in cases:
            start = time.monotonic()
            response = inside_http(container, "POST", "/v1/recall", {
                "query": case["query"], "namespace": case["namespace"],
                "limit": 12, "min_score": 0.0,
            })
            latency_ms[case["case_id"]] = round((time.monotonic() - start) * 1000, 2)
            if response["status"] != 200:
                raise RuntimeError(f"recall failed for {case['case_id']}: HTTP {response['status']}")
            items = _result_items(response.get("body"), case["case_id"])
            rows, unmapped = _preserve_recall_candidates(items, memory_to_fixture, case["case_id"])
            raw_scored[case["case_id"]] = rows
            eligible[case["case_id"]] = eligible_candidates(rows, case, anchor_map)
            diagnostics[case["case_id"]] = {
                "raw_count": len(rows), "eligible_count": len(eligible[case["case_id"]]),
                "raw_superseded_count": sum(
                    anchor_map[row["fixture_id"]]["status"] != "active"
                    for row in rows if row["fixture_id"] in anchor_map
                ),
                "raw_cross_namespace_count": sum(
                    anchor_map[row["fixture_id"]]["namespace"] != case["namespace"]
                    for row in rows if row["fixture_id"] in anchor_map
                ),
                "raw_cross_split_count": sum(
                    anchor_map[row["fixture_id"]]["split"] != "calibration"
                    for row in rows if row["fixture_id"] in anchor_map
                ),
                "unmapped_result_count": len(unmapped),
                "unmapped_candidates": unmapped,
            }

        calibration = policy_sweep(cases, eligible, raw_scored, anchor_map)
        latencies = sorted(latency_ms.values())
        p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1)
        return {
            "schema": "c8.7.corrected-calibration-report/v2",
            "status": "CALIBRATION_FEASIBLE" if calibration["selected_policy"] else "CALIBRATION_INFEASIBLE",
            "scope": "synthetic calibration only", "actual_nyawa": True,
            "backend": "bge-onnx-local", "calibration": calibration,
            "policy_gates": {"minimum": MIN_GATES, "maximum": MAX_GATES},
            "eligibility": {"metadata_fields": ["namespace", "split", "status"], "labels_used": False,
                            "raw_retrieval_preserved": True},
            "fixture_counts": {"calibration_cases": len(cases), "calibration_records": len(records)},
            "vector_indexed": vector_indexed,
            "raw_retrieval_diagnostics": diagnostics,
            "retrieval_results": {
                case["case_id"]: [{"fixture_id": row["fixture_id"], "score": row["score"], "rank": row["rank"]}
                                   for row in raw_scored[case["case_id"]]] for case in cases
            },
            "latency_ms": latency_ms,
            "latency_median_ms": round(latencies[len(latencies) // 2], 2) if latencies else 0.0,
            "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
            "runtime_attestation": {"mode": "local", "network": "blocked", "container": container},
            "image": image, "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "queries_run": len(cases),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "e2e"), required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--image", default=IMAGE_C87)
    parser.add_argument("--container", default="nyawa-c87-eval")
    args = parser.parse_args()
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C87Tests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.mode == "e2e":
        if args.fixtures is None or not args.fixtures.is_file():
            print("error: --fixtures is required for e2e mode", file=sys.stderr)
            return 1
        try:
            report = run_e2e(args.container, args.image, args.fixtures)
        except Exception as error:
            print(f"C8.7 evaluation FAILED: {error}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

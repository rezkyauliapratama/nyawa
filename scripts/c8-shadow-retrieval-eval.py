#!/usr/bin/env python3
"""Run a bounded C8 retrieval-quality evaluation on synthetic Nyawa data only."""
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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IMAGE = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:61da25caf922c0925f4fd8e3af1d811f6fc9d3a9df21beb44138c7300d47efb6"
)
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/c8-shadow-retrieval-fixtures.json"
MEMORY_LIMIT = 4 * 1024**3
MIN_AVAILABLE_MEMORY = 5 * 1024**3
AWS_SECRET_SHAPED = re.compile(r"[A-Za-z0-9/+=]{40}")
QUICK_CASE_IDS = ("Q01_SHA", "Q06_DECISION", "Q11_SCOPE_01", "Q12_SCOPE_02", "Q16_ACTIVE_LEASE", "Q21_NO_HIT_LUNAR")
TRACE_FILE: Path | None = None


def trace_event(event: str, **fields: Any) -> None:
    """Emit content-free JSONL progress with run identity and UTC timestamp."""
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "event": event,
        **fields,
    }
    line = json.dumps(record, sort_keys=True, allow_nan=False)
    if TRACE_FILE is not None:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    print(line, file=sys.stderr, flush=True)


@contextmanager
def traced_stage(name: str, **fields: Any):
    started = time.monotonic()
    trace_event("stage", name=name, status="started", **fields)
    try:
        yield
    except Exception as error:
        trace_event("stage", name=name, status="failed", elapsed_ms=round((time.monotonic() - started) * 1000, 2), error_type=type(error).__name__, **fields)
        raise
    else:
        trace_event("stage", name=name, status="complete", elapsed_ms=round((time.monotonic() - started) * 1000, 2), **fields)


def select_scope(dataset: dict[str, Any], scope: str) -> dict[str, Any]:
    """Return a copy with either the fixed fast smoke or all fixture cases."""
    if scope == "full":
        return {**dataset, "cases": list(dataset["cases"])}
    if scope != "quick-smoke":
        raise ValueError(f"unsupported evaluation scope: {scope}")
    cases = {case["case_id"]: case for case in dataset["cases"]}
    missing = [case_id for case_id in QUICK_CASE_IDS if case_id not in cases]
    if missing:
        raise ValueError(f"quick-smoke fixture cases are missing: {missing}")
    return {**dataset, "cases": [cases[case_id] for case_id in QUICK_CASE_IDS]}


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


def validate_dataset(dataset: dict[str, Any]) -> None:
    if dataset.get("schema_version") != "c8.shadow-retrieval.v1":
        raise ValueError("unsupported C8 fixture schema")
    records = dataset.get("records")
    cases = dataset.get("cases")
    if not isinstance(records, list) or not records or not isinstance(cases, list) or not cases:
        raise ValueError("fixture dataset must contain records and query cases")
    record_ids = [str(record.get("fixture_id", "")) for record in records]
    if not all(record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("record fixture_id values must be present and unique")
    record_content_by_namespace: dict[str, list[str]] = {}
    all_record_content: list[str] = []
    for record in records:
        namespace = record.get("namespace", "")
        content = record.get("content", "")
        if not namespace.startswith("c8-synthetic-"):
            raise ValueError(f"non-synthetic namespace rejected: {namespace!r}")
        if not isinstance(content, str) or "Synthetic C8" not in content:
            raise ValueError(f"record {record['fixture_id']} is not clearly synthetic")
        if AWS_SECRET_SHAPED.search(content):
            raise ValueError(f"record {record['fixture_id']} resembles a blocked secret-shaped string")
        record_content_by_namespace.setdefault(namespace, []).append(content)
        all_record_content.append(content)
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if not all(case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("case_id values must be present and unique")
    available_namespaces = {record["namespace"] for record in records}
    for case in cases:
        if case.get("namespace") not in available_namespaces:
            raise ValueError(f"case {case['case_id']} uses an unseeded namespace")
        if not str(case.get("query", "")).strip():
            raise ValueError(f"case {case['case_id']} has an empty query")
        if not isinstance(case.get("relevant_markers"), list) or not isinstance(case.get("forbidden_markers"), list):
            raise ValueError(f"case {case['case_id']} must declare relevant and forbidden markers")
        if not all(isinstance(x, str) and x for x in case["relevant_markers"] + case["forbidden_markers"]):
            raise ValueError(f"case {case['case_id']} has an invalid marker")
        if not isinstance(case.get("exclude_types", []), list):
            raise ValueError(f"case {case['case_id']} exclude_types must be a list")
    required_categories = {
        "exact_identifier",
        "semantic_paraphrase",
        "namespace_isolation",
        "active_vs_retired",
        "negative_query",
    }
    category_counts: dict[str, int] = {}
    for case in cases:
        category = case.get("category", "")
        category_counts[category] = category_counts.get(category, 0) + 1
        namespace_text = "\n".join(record_content_by_namespace.get(case["namespace"], []))
        for marker in case["relevant_markers"]:
            if marker not in namespace_text:
                raise ValueError(f"relevant marker {marker!r} for {case['case_id']} is not seeded in its namespace")
        for marker in case["forbidden_markers"]:
            if not any(marker in content for content in all_record_content):
                raise ValueError(f"forbidden marker {marker!r} for {case['case_id']} is absent from the corpus")
    if not required_categories.issubset(category_counts):
        raise ValueError(f"fixture corpus is missing categories: {sorted(required_categories - category_counts.keys())}")
    underfilled = {name: category_counts.get(name, 0) for name in required_categories if category_counts.get(name, 0) < 5}
    if underfilled:
        raise ValueError(f"C8 corpus requires at least five cases per category: {underfilled}")


def load_dataset(path: Path = FIXTURES) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    validate_dataset(dataset)
    return dataset


def _content(item: dict[str, Any]) -> str:
    return str(item.get("Content", item.get("content", "")))


def _score(item: dict[str, Any]) -> Any:
    return item.get("Score", item.get("score"))


def score_case(case: dict[str, Any], items: list[dict[str, Any]], latency_ms: float, k: int) -> dict[str, Any]:
    candidates = items[:k]
    relevant = case["relevant_markers"]
    forbidden = case["forbidden_markers"]
    found_relevant = [marker for marker in relevant if any(marker in _content(item) for item in candidates)]
    first_relevant_rank = next(
        (rank for rank, item in enumerate(candidates, start=1) if any(marker in _content(item) for marker in relevant)),
        None,
    )
    relevant_candidate_count = sum(
        1 for item in candidates if any(marker in _content(item) for marker in relevant)
    )
    forbidden_hits = [marker for marker in forbidden if any(marker in _content(item) for item in candidates)]
    candidate_chars = sum(len(_content(item)) for item in candidates)
    result_count = len(candidates)
    return {
        "case_id": case["case_id"],
        "category": case["category"],
        "namespace": case["namespace"],
        "relevant_count": len(relevant),
        "relevant_found": len(found_relevant),
        "recall_at_k": (len(found_relevant) / len(relevant)) if relevant else None,
        "precision_at_k": (relevant_candidate_count / result_count) if relevant and result_count else (0.0 if relevant else None),
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": (1.0 / first_relevant_rank) if first_relevant_rank is not None else (0.0 if relevant else None),
        "forbidden_markers_found": forbidden_hits,
        "negative_false_positive": (not relevant and result_count > 0),
        "result_count": result_count,
        "scores_numeric": all(isinstance(_score(item), (int, float)) for item in candidates),
        "latency_ms": round(float(latency_ms), 2),
        "candidate_chars": candidate_chars,
        "estimated_candidate_tokens": math.ceil(candidate_chars / 4),
        "retrieved_fixture_ids": [str(item.get("ID", item.get("id", ""))) for item in candidates],
        "scores": [float(_score(item)) for item in candidates if isinstance(_score(item), (int, float))],
    }


def summarize(dataset: dict[str, Any], per_case: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["case_id"]: row for row in per_case}
    cases = dataset["cases"]
    positive_cases = [case for case in cases if case["relevant_markers"]]
    exact_cases = [case for case in positive_cases if case["category"] == "exact_identifier"]
    total_relevant = sum(by_id[case["case_id"]]["relevant_count"] for case in positive_cases)
    total_found = sum(by_id[case["case_id"]]["relevant_found"] for case in positive_cases)
    total_results = sum(by_id[case["case_id"]]["result_count"] for case in positive_cases)
    total_relevant_results = sum(
        by_id[case["case_id"]]["relevant_found"] for case in positive_cases
    )
    recall = total_found / total_relevant if total_relevant else 0.0
    precision = total_relevant_results / total_results if total_results else 0.0
    exact_total = sum(by_id[case["case_id"]]["relevant_count"] for case in exact_cases)
    exact_found = sum(by_id[case["case_id"]]["relevant_found"] for case in exact_cases)
    exact_recall = exact_found / exact_total if exact_total else 0.0
    rr_values = [by_id[case["case_id"]]["reciprocal_rank"] or 0.0 for case in positive_cases]
    mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
    scope_cases = [case for case in cases if case["category"] == "namespace_isolation"]
    scope_leaks = sum(bool(by_id[case["case_id"]]["forbidden_markers_found"]) for case in scope_cases)
    forbidden_leaks = sum(bool(row["forbidden_markers_found"]) for row in per_case)
    numeric_scores = all(row["scores_numeric"] for row in per_case)
    negative_cases = [case for case in cases if case["category"] == "negative_query"]
    negative_fp = sum(by_id[case["case_id"]]["negative_false_positive"] for case in negative_cases)
    latencies = sorted(row["latency_ms"] for row in per_case)
    p95_index = max(0, math.ceil(0.95 * len(latencies)) - 1) if latencies else 0
    gates = dataset["gates"]
    checks = {
        "positive_recall_at_k": recall >= float(gates["positive_recall_at_k_min"]),
        "exact_identifier_recall": exact_recall >= float(gates["exact_identifier_recall_min"]),
        "namespace_leaks_zero": scope_leaks <= int(gates["namespace_leaks_max"]),
        "forbidden_marker_leaks_zero": forbidden_leaks <= int(gates["forbidden_marker_leaks_max"]),
        "numeric_scores": numeric_scores if gates.get("require_numeric_scores", True) else True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "metric_version": "c8.metrics.v1",
        "baseline": {
            "B0_no_retrieval_recall_at_k": 0.0,
            "B1_nyawa_recall_at_k": round(recall, 4),
            "delta_candidate_recall": round(recall, 4),
            "interpretation": "candidate coverage only; not answer-quality or LLM comparison",
        },
        "metrics": {
            "recall_at_k": round(recall, 4),
            "precision_at_k": round(precision, 4),
            "mrr": round(mrr, 4),
            "exact_identifier_recall_at_k": round(exact_recall, 4),
            "namespace_leaks": scope_leaks,
            "forbidden_marker_leaks": forbidden_leaks,
            "negative_query_count": len(negative_cases),
            "negative_false_positive_queries": negative_fp,
            "negative_false_positive_rate": round(negative_fp / len(negative_cases), 4) if negative_cases else None,
            "latency_median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
            "latency_p95_ms": round(latencies[p95_index], 2) if latencies else 0.0,
            "candidate_chars_total": sum(row["candidate_chars"] for row in per_case),
            "estimated_candidate_tokens_total": sum(row["estimated_candidate_tokens"] for row in per_case),
            "score_values_numeric": numeric_scores,
        },
        "gates": checks,
        "calibration_only": ["precision_at_k", "negative_false_positive_rate", "latency_p95_ms", "estimated_candidate_tokens_total"],
        "case_results": per_case,
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
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count < 2:
        raise RuntimeError("runner preflight failed: at least 2 CPUs required; no container started")
    stat = os.statvfs(os.environ.get("RUNNER_TEMP", "/tmp"))
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 4 * 1024**3:
        raise RuntimeError("runner preflight failed: less than 4 GiB free disk; no container started")
    print(json.dumps({"preflight": "PASS", "mem_available_bytes": available, "cpu_count": os.cpu_count(), "runner_temp_free_bytes": free_bytes}, sort_keys=True))


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
    if inspected["Config"].get("Image") != image or labels.get("org.nyawa.c8-shadow-eval") != "true":
        raise RuntimeError("container image/ownership mismatch")
    if host.get("NetworkMode") != "none" or not host.get("ReadonlyRootfs"):
        raise RuntimeError("container isolation is not network=none/read-only")
    if host.get("PortBindings") or host.get("Binds") or host.get("Mounts"):
        raise RuntimeError("container unexpectedly has a port or host/volume mount")
    if host.get("Memory") != MEMORY_LIMIT or host.get("MemorySwap") != MEMORY_LIMIT or host.get("PidsLimit") != 128:
        raise RuntimeError("container resource limits do not match the C8 bounds")
    if any(mount.get("Type") != "tmpfs" for mount in inspected.get("Mounts", [])):
        raise RuntimeError("container has a non-tmpfs mount")


def cleanup(container: str, image: str) -> None:
    result = docker(["inspect", container], timeout=15, check=False)
    if result.returncode:
        return
    inspected = json.loads(result.stdout)[0]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if labels.get("org.nyawa.c8-shadow-eval") != "true" or inspected["Config"].get("Image") != image:
        raise RuntimeError("refusing to remove a container not created by this C8 run")
    docker(["rm", "--force", container], timeout=30)
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("C8 cleanup readback failed: container still exists")


def run_e2e(container: str, image: str, fixture_path: Path, scope: str, report_path: Path | None) -> dict[str, Any]:
    if image != IMAGE:
        raise RuntimeError("image must equal the approved immutable BGE r1 digest")
    full_dataset = load_dataset(fixture_path)
    dataset = select_scope(full_dataset, scope)
    fixture_sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    image_digest = image.split("@", 1)[1]
    trace_event("run_start", scope=scope, image_digest=image_digest, fixture_sha256=fixture_sha256,
                fixture_case_count=len(full_dataset["cases"]), selected_case_ids=[case["case_id"] for case in dataset["cases"]])
    with traced_stage("runner_preflight", scope=scope):
        preflight()
    with traced_stage("image_pull", image_digest=image_digest):
        docker(["pull", image], timeout=300)
    with traced_stage("image_digest_verify", image_digest=image_digest):
        image_info = json.loads(docker(["image", "inspect", image]).stdout)[0]
        if image not in image_info.get("RepoDigests", []):
            raise RuntimeError("pulled image digest does not match the pinned image")
    existing = docker(["inspect", container], timeout=15, check=False)
    if existing.returncode == 0:
        raise RuntimeError("container name already exists; refusing to replace it")
    try:
        with traced_stage("container_start", container=container, image_digest=image_digest):
            docker([
                "run", "--detach", "--name", container,
                "--label", "org.nyawa.c8-shadow-eval=true",
                "--network=none", "--read-only",
                "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
                "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
                "--restart=no", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--env", "MEMORY_EMBEDDER=local",
                "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model",
                "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
                image, "serve", "/scratch/c8-shadow.db",
            ], timeout=60)
        with traced_stage("container_isolation_verify", container=container):
            verify_container(container, image)
        with traced_stage("readiness", timeout_seconds=180):
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
        with traced_stage("runtime_identity", expected_version="1.2.4"):
            root_body = root["body"]
            if root_body.get("version") != "1.2.4" or root_body.get("status") != "running":
                raise RuntimeError(f"unexpected Nyawa identity: {root_body}")

        inserted: dict[str, str] = {}
        with traced_stage("seed_synthetic_records", record_count=len(dataset["records"])):
            for index, record in enumerate(dataset["records"], start=1):
                response = inside_http(container, "POST", "/v1/memories", {
                    "content": record["content"], "namespace": record["namespace"], "type": record["type"]
                })
                if response["status"] != 201 or not response["body"].get("id"):
                    raise RuntimeError(f"synthetic insert failed for {record['fixture_id']}: {response}")
                inserted[record["fixture_id"]] = str(response["body"]["id"])
                trace_event("record_inserted", fixture_id=record["fixture_id"], index=index, total=len(dataset["records"]))

        with traced_stage("vector_index_check", expected_records=len(dataset["records"])):
            stats = inside_http(container, "GET", "/v1/stats")
            if stats["status"] != 200:
                raise RuntimeError(f"stats returned HTTP {stats['status']}")
            store_stats = stats["body"].get("store", stats["body"])
            vector_indexed = int(store_stats.get("vector_indexed", 0))
            if vector_indexed < len(dataset["records"]):
                raise RuntimeError(f"not all synthetic records are vector-indexed: {store_stats}")
            trace_event("vector_indexed", count=vector_indexed, expected=len(dataset["records"]))

        per_case = []
        with traced_stage("recall_queries", scope=scope, query_count=len(dataset["cases"])):
            for index, case in enumerate(dataset["cases"], start=1):
                trace_event("recall_case", status="started", case_id=case["case_id"], index=index, total=len(dataset["cases"]))
                start = time.monotonic()
                response = inside_http(container, "POST", "/v1/recall", {
                    "query": case["query"],
                    "namespace": case["namespace"],
                    "limit": int(dataset["k"]),
                    "min_score": float(case.get("min_score", 0.0)),
                    "exclude_types": case.get("exclude_types", []),
                })
                elapsed = (time.monotonic() - start) * 1000
                if response["status"] != 200:
                    trace_event("recall_case", status="failed", case_id=case["case_id"], elapsed_ms=round(elapsed, 2), http_status=response["status"])
                    raise RuntimeError(f"recall failed for {case['case_id']}: HTTP {response['status']}")
                items = response["body"].get("results", [])
                if not isinstance(items, list):
                    trace_event("recall_case", status="failed", case_id=case["case_id"], error_type="malformed_results")
                    raise RuntimeError(f"recall results malformed for {case['case_id']}")
                per_case.append(score_case(case, items, elapsed, int(dataset["k"])))
                trace_event("recall_case", status="complete", case_id=case["case_id"], index=index, total=len(dataset["cases"]), elapsed_ms=round(elapsed, 2), candidate_count=len(items))

        with traced_stage("metric_summary", scope=scope, query_count=len(per_case)):
            report = summarize(dataset, per_case)
        report.update({
            "report_schema": "c8.shadow-evaluation-report.v1",
            "evaluation_scope": scope,
            "full_fixture_case_count": len(full_dataset["cases"]),
            "selected_case_ids": [case["case_id"] for case in dataset["cases"]],
            "image": image,
            "image_digest": image_digest,
            "runtime_version": root_body["version"],
            "fixture_sha256": fixture_sha256,
            "synthetic_records_inserted": len(inserted),
            "vector_indexed": vector_indexed,
            "queries_run": len(dataset["cases"]),
            "result_limit_k": int(dataset["k"]),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "source_sha": os.environ.get("GITHUB_SHA"),
            "source_ref": os.environ.get("GITHUB_REF"),
        })
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_bytes = json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
            report_path.write_bytes(report_bytes)
            trace_event("report_written", filename=report_path.name, bytes=len(report_bytes), status=report["status"])
        trace_event("evaluation_complete", scope=scope, status=report["status"], queries_run=len(per_case), fixture_sha256=fixture_sha256)
        return report
    finally:
        with traced_stage("container_cleanup", container=container):
            cleanup(container, image)


class C8MetricsTests(unittest.TestCase):
    def test_fixture_schema_is_valid_and_synthetic(self) -> None:
        dataset = load_dataset()
        self.assertEqual(len(dataset["cases"]), 25)
        for category in {case["category"] for case in dataset["cases"]}:
            self.assertGreaterEqual(sum(case["category"] == category for case in dataset["cases"]), 5)
        self.assertTrue(all(r["namespace"].startswith("c8-synthetic-") for r in dataset["records"]))
        self.assertFalse(any(AWS_SECRET_SHAPED.search(r["content"]) for r in dataset["records"]))

    def test_scope_selection_is_fixed_and_traceable(self) -> None:
        dataset = load_dataset()
        quick = select_scope(dataset, "quick-smoke")
        self.assertEqual(tuple(case["case_id"] for case in quick["cases"]), QUICK_CASE_IDS)
        self.assertEqual(len(select_scope(dataset, "full")["cases"]), len(dataset["cases"]))

    def test_fixture_rejects_secret_shaped_strings(self) -> None:
        dataset = load_dataset()
        dataset["records"][0]["content"] += " " + "A" * 40
        with self.assertRaisesRegex(ValueError, "secret-shaped"):
            validate_dataset(dataset)

    def test_recall_precision_and_mrr(self) -> None:
        case = {"case_id": "Q", "category": "semantic", "namespace": "c8-synthetic-main", "relevant_markers": ["GOLD"], "forbidden_markers": []}
        result = score_case(case, [{"Content": "GOLD text", "Score": 0.8}, {"Content": "decoy", "Score": 0.3}], 12, 5)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["precision_at_k"], 0.5)
        self.assertEqual(result["reciprocal_rank"], 1.0)

    def test_forbidden_marker_is_detected_as_leak(self) -> None:
        case = {"case_id": "Q", "category": "namespace_isolation", "namespace": "c8-synthetic-a", "relevant_markers": ["A"], "forbidden_markers": ["B"]}
        result = score_case(case, [{"Content": "A and B", "Score": 0.5}], 1, 5)
        self.assertEqual(result["forbidden_markers_found"], ["B"])

    def test_negative_query_false_positive_is_reported(self) -> None:
        case = {"case_id": "Q", "category": "negative_query", "namespace": "c8-synthetic-main", "relevant_markers": [], "forbidden_markers": []}
        result = score_case(case, [{"Content": "unrelated result", "Score": 0.1}], 2, 5)
        self.assertTrue(result["negative_false_positive"])
        self.assertIsNone(result["precision_at_k"])

    def test_gates_fail_closed_on_namespace_leak(self) -> None:
        dataset = {
            "cases": [{"case_id": "Q", "category": "namespace_isolation", "relevant_markers": ["A"]}],
            "gates": {"positive_recall_at_k_min": 0.9, "exact_identifier_recall_min": 1.0, "namespace_leaks_max": 0, "forbidden_marker_leaks_max": 0, "require_numeric_scores": True},
        }
        row = {"case_id": "Q", "category": "namespace_isolation", "relevant_count": 1, "relevant_found": 1, "recall_at_k": 1.0, "precision_at_k": 1.0, "reciprocal_rank": 1.0, "forbidden_markers_found": ["B"], "negative_false_positive": False, "result_count": 1, "scores_numeric": True, "latency_ms": 1, "candidate_chars": 1, "estimated_candidate_tokens": 1}
        report = summarize(dataset, [row])
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["gates"]["namespace_leaks_zero"])

    def test_perfect_synthetic_results_satisfy_all_hard_gates(self) -> None:
        dataset = load_dataset()
        rows = []
        for case in dataset["cases"]:
            if case["relevant_markers"]:
                items = [{"ID": case["case_id"], "Content": "Synthetic C8 " + " ".join(case["relevant_markers"]), "Score": 0.99}]
            else:
                items = []
            rows.append(score_case(case, items, 2.0, dataset["k"]))
        report = summarize(dataset, rows)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(report["metrics"]["exact_identifier_recall_at_k"], 1.0)
        self.assertEqual(report["metrics"]["namespace_leaks"], 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "self-test", "e2e"), required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--container", default="nyawa-c8-shadow-eval")
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--scope", choices=("quick-smoke", "full"), default="quick-smoke")
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--report-file", type=Path)
    args = parser.parse_args()
    global TRACE_FILE
    TRACE_FILE = args.trace_file
    if TRACE_FILE is not None:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        TRACE_FILE.touch(exist_ok=True)
    trace_event("harness_start", mode=args.mode, scope=args.scope)
    if args.mode == "preflight":
        try:
            with traced_stage("runner_preflight", scope=args.scope):
                preflight()
        except Exception as error:
            trace_event("preflight_result", status="FAILED", error_type=type(error).__name__)
            if args.report_file is not None:
                preflight_report = {
                    "report_schema": "c8.shadow-evaluation-report.v1",
                    "status": "PREFLIGHT_FAILED",
                    "retrieval_run": False,
                    "evaluation_scope": args.scope,
                    "failure_type": type(error).__name__,
                    "image_digest": args.image.split("@", 1)[1] if "@" in args.image else "unverified",
                    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                    "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                    "source_sha": os.environ.get("GITHUB_SHA"),
                }
                args.report_file.parent.mkdir(parents=True, exist_ok=True)
                args.report_file.write_text(json.dumps(preflight_report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
            print(f"C8 preflight FAILED: {error}", file=sys.stderr)
            return 1
        if args.report_file is not None:
            preflight_report = {
                "report_schema": "c8.shadow-evaluation-report.v1",
                "status": "PREFLIGHT_PASS",
                "retrieval_run": False,
                "evaluation_scope": args.scope,
                "image_digest": args.image.split("@", 1)[1] if "@" in args.image else "unverified",
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "source_sha": os.environ.get("GITHUB_SHA"),
            }
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(json.dumps(preflight_report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
        return 0
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C8MetricsTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        report = run_e2e(args.container, args.image, args.fixtures, args.scope, args.report_file)
        print(json.dumps(report, indent=2, sort_keys=True))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("Scope: " + args.scope + chr(10))
                stream.write("Run: " + os.environ.get("GITHUB_RUN_ID", "local") + "." + os.environ.get("GITHUB_RUN_ATTEMPT", "1") + chr(10))
                stream.write("Source SHA: " + os.environ.get("GITHUB_SHA", "local") + chr(10) + chr(10))
                stream.write("## C8 Shadow Retrieval Evaluation" + chr(10) + chr(10) + "```json" + chr(10))
                stream.write(json.dumps(report, indent=2, sort_keys=True))
                stream.write("\n```\n\nOnly synthetic records were used; no provider call or Hermes injection.\n")
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        trace_event("run_failed", scope=args.scope, error_type=type(error).__name__)
        if args.report_file is not None:
            failure_report = {
                "report_schema": "c8.shadow-evaluation-report.v1",
                "status": "FAILED",
                "evaluation_scope": args.scope,
                "failure_type": type(error).__name__,
                "image_digest": args.image.split("@", 1)[1] if "@" in args.image else "unverified",
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
                "source_sha": os.environ.get("GITHUB_SHA"),
            }
            args.report_file.parent.mkdir(parents=True, exist_ok=True)
            args.report_file.write_text(json.dumps(failure_report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
        print(f"C8 evaluation FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

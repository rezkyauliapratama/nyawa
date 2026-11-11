#!/usr/bin/env python3
"""C8.7 corrected calibration runner — synthetic-only, calibration-first, fail-closed holdout.
Uses the same Docker image infrastructure as existing C8.1–C8.6 workflows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any

# Reuse helpers from the existing shared calibration module.
from c81_abstention_calibration import (
    SCHEMA, SCHEMA_C83, SCHEMA_C84, SCHEMA_C85, SCHEMA_C86,
    HTTP_HELPER, inside_http, docker, preflight, verify_container, cleanup,
)
from c87_fixture_generator import SCHEMA as SCHEMA_C87

# ---------------------------------------------------------------------------
# Type map
# ---------------------------------------------------------------------------
TYPE_MAP = {"artifact": "reference", "constraint": "procedure",
            "decision": "decision", "fact": "fact"}

# ---------------------------------------------------------------------------
# Image pin
# ---------------------------------------------------------------------------
# Reuse the same BGE runtime image used by C8.4 (any recent bge-runtime works)
IMAGE_C87 = "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@sha256:bcbc858b5df17d734858eb5db7c1faa521a327576d8070e97bc248653eff5246"
LABEL_KEY = "org.nyawa.abstention-calibration"

# ---------------------------------------------------------------------------
# Hard gates (unchanged from C8)
# ---------------------------------------------------------------------------
MIN_GATES = {"recall_at_5": 0.90, "exact_identifier_recall_at_k": 1.0,
             "active_decision_recall_at_k": 0.90, "hard_constraint_recall_at_k": 1.0,
             "supersession_recall_at_k": 0.90}
MAX_GATES = {"false_retrieval_rate_at_k": 0.00}
POLICY_KS = (1, 3, 5, 12)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> dict:
    d = json.loads(path.read_bytes())
    if d["schema_version"] != SCHEMA_C87:
        raise RuntimeError(f"unexpected schema: {d['schema_version']}")
    return d


def supported_mcp_type(fixture_type: str) -> str:
    try:
        return TYPE_MAP[fixture_type]
    except KeyError:
        raise RuntimeError(f"unsupported fixture type: {fixture_type}")


def eligible_candidates(raw: list[dict], case: dict, anchor_map: dict[str, dict]) -> list[dict]:
    """Filter by anchor metadata only. No answerable/expected/forbidden labels used."""
    out = []
    for row in raw:
        aid = str(row.get("fixture_id", ""))
        meta = anchor_map.get(aid)
        if not meta:
            continue
        if meta.get("namespace") != case.get("namespace"):
            continue
        if meta.get("status") != "active" or meta.get("supersedes"):
            continue
        out.append(row)
    return out


def threshold_grid(scored: dict[str, list[dict]]) -> list[float]:
    vals = sorted({float(r["score"]) for rs in scored.values() for r in rs})
    return vals + [math.nextafter(vals[-1], math.inf)] if vals else [0.0]


def policy_sweep(cases: list[dict], anchors: dict[str, dict], eligible: dict[str, list[dict]],
                 b0: dict[str, list[str]]) -> dict:
    thresholds = threshold_grid(eligible)
    sweep = []
    for th in thresholds:
        for k in POLICY_KS:
            metrics = {"recall_at_5": 0.0, "exact_identifier_recall_at_k": 0.0,
                       "active_decision_recall_at_k": 0.0, "hard_constraint_recall_at_k": 0.0,
                       "supersession_recall_at_k": 0.0, "false_retrieval_rate_at_k": 0.0,
                       "forbidden_selected_count": 0, "superseded_selected_count": 0,
                       "namespace_leak_count": 0}
            n_answerable, n_found = 0, 0
            for case in cases:
                cat = case["category"]
                if cat == "negative_confusable":
                    continue
                n_answerable += 1
                ids = [str(r["fixture_id"]) for r in eligible.get(case["case_id"], [])
                       if r["score"] >= th][:k]
                if not ids:
                    continue
                exp = set(case.get("expected_fixture_ids", []))
                if ids[0] in exp:
                    n_found += 1
                    if cat == "exact_identifier":
                        metrics["exact_identifier_recall_at_k"] += 1.0
                    elif cat == "active_decision":
                        metrics["active_decision_recall_at_k"] += 1.0
                    elif cat == "hard_constraint":
                        metrics["hard_constraint_recall_at_k"] += 1.0
                    elif cat == "supersession":
                        metrics["supersession_recall_at_k"] += 1.0
                for fid in ids:
                    meta = anchors.get(fid)
                    if not meta:
                        continue
                    if fid in case.get("forbidden_fixture_ids", []):
                        metrics["forbidden_selected_count"] += 1
                    if meta.get("status") == "superseded":
                        metrics["superseded_selected_count"] += 1
                    if meta.get("namespace") != case.get("namespace"):
                        metrics["namespace_leak_count"] += 1
            if n_answerable:
                for kk in ("exact_identifier_recall_at_k", "active_decision_recall_at_k",
                           "hard_constraint_recall_at_k", "supersession_recall_at_k"):
                    metrics[kk] /= n_answerable / 5
                metrics["recall_at_5"] = n_found / n_answerable if n_answerable else 0.0
            neg_cases = [c for c in cases if c["category"] == "negative_confusable"]
            n_neg = sum(1 for c in neg_cases
                        if any(r["score"] >= th for r in eligible.get(c["case_id"], [])))
            metrics["false_retrieval_rate_at_k"] = n_neg / len(neg_cases) if neg_cases else 0.0

            feasible = True
            for gate, val in MIN_GATES.items():
                if metrics.get(gate, 0) < val:
                    feasible = False
                    break
            if feasible:
                for gate, val in MAX_GATES.items():
                    if metrics.get(gate, 0) > val:
                        feasible = False
                        break
            if feasible:
                for extra in ("forbidden_selected_count", "superseded_selected_count",
                             "namespace_leak_count"):
                    if metrics.get(extra, 0) > 0:
                        feasible = False
                        break
            sweep.append({"threshold": round(th, 6), "k": k, "feasible": feasible, "metrics": metrics})

    b0_metrics = {"recall_at_5": 0.0}
    n_ans_b0 = sum(1 for c in cases if c["category"] != "negative_confusable")
    n_found_b0 = sum(1 for c in cases if c["category"] != "negative_confusable"
                     and b0.get(c["case_id"], [])
                     and b0[c["case_id"]][0] in c.get("expected_fixture_ids", []))
    b0_metrics["recall_at_5"] = n_found_b0 / n_ans_b0 if n_ans_b0 else 0.0

    feasible = [p for p in sweep if p["feasible"]]
    selected = max(feasible, key=lambda p: (p["metrics"]["recall_at_5"], -p["k"], p["threshold"])) if feasible else None
    return {
        "case_count": len(cases),
        "b0": b0_metrics,
        "score_threshold_count": len(thresholds),
        "policy_sweep_count": len(sweep),
        "feasible_policy_count": len(feasible),
        "selected_policy": ({"threshold": selected["threshold"], "k": selected["k"],
                            "calibration_metrics": selected["metrics"]} if selected else None),
        "sweep": sweep,
    }


def run_e2e(container: str, image: str, fixture_path: Path) -> dict:
    dataset = load_dataset(fixture_path)
    raws = dataset["records"]
    cases = [c for c in dataset["cases"] if c["split"] == "calibration"]
    holdout_cases = [c for c in dataset["cases"] if c["split"] == "holdout"]

    anchor_map = {}
    for r in raws:
        anchor_map[r["fixture_id"]] = {
            "namespace": r["namespace"], "type": r["type"],
            "status": r["status"], "supersedes": r.get("supersedes", []),
            "split": r["split"],
        }

    preflight()
    docker(["pull", image], timeout=300)
    image_info = json.loads(docker(["image", "inspect", image]).stdout)[0]
    if image not in image_info.get("RepoDigests", []):
        raise RuntimeError("pulled image digest mismatch")
    if docker(["inspect", container], timeout=15, check=False).returncode == 0:
        raise RuntimeError("container already exists")

    db_path = "/scratch/c87-calibration.db"
    try:
        docker(["run", "--detach", "--name", container, "--label", f"{LABEL_KEY}=true",
                "--network=none", "--read-only",
                "--tmpfs", "/scratch:rw,nosuid,nodev,size=768m",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m",
                "--memory=4g", "--memory-swap=4g", "--cpus=2", "--pids-limit=128",
                "--restart=no", "--cap-drop=ALL", "--security-opt=no-new-privileges:true",
                "--env", "MEMORY_EMBEDDER=local",
                "--env", "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model",
                "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
                image, "serve", db_path], timeout=60)
        verify_container(container, image)

        deadline = time.monotonic() + 180
        root = health = None
        last_error = "not ready"
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
            except Exception as e:
                last_error = str(e)
            time.sleep(2)
        else:
            logs = docker(["logs", container], timeout=15, check=False).stdout[-1000:]
            raise RuntimeError(f"Nyawa timeout: {last_error}; {logs}")

        # Insert calibration-only synthetic records with type mapping
        cal_records = [r for r in raws if r["split"] == "calibration"]
        memory_to_fixture = {}
        for r in cal_records:
            mcp_type = supported_mcp_type(r["type"])
            response = inside_http(container, "POST", "/v1/memories", {
                "content": r["content"],
                "namespace": r["namespace"],
                "mem_type": mcp_type,
            })
            if response["status"] != 200:
                raise RuntimeError(f"store failed for {r['fixture_id']}")
            mid = response["body"].get("id", "")
            if mid:
                memory_to_fixture[mid] = r["fixture_id"]

        # Retrieve for each calibration case
        cal_scored = {}
        for case in cases:
            response = inside_http(container, "POST", "/v1/recall", {
                "query": case["query"],
                "namespace": case["namespace"],
                "limit": 12,
                "min_score": 0.0,
            })
            if response["status"] != 200:
                raise RuntimeError(f"recall failed for {case['case_id']}")
            items = response["body"].get("results", [])
            rows = []
            for rank, item in enumerate(items, 1):
                mid = item.get("id", "")
                fid = memory_to_fixture.get(mid)
                if fid:
                    rows.append({"fixture_id": fid, "score": item.get("score", 0.0), "rank": rank})
            cal_scored[case["case_id"]] = rows

        # Apply eligibility filter
        eligible = {}
        raw_diag = {}
        for case in cases:
            raw = cal_scored.get(case["case_id"], [])
            el = eligible_candidates(raw, case, anchor_map)
            eligible[case["case_id"]] = el
            raw_diag[case["case_id"]] = {
                "raw_count": len(raw),
                "eligible_count": len(el),
                "raw_superseded": sum(1 for r in raw if anchor_map.get(r["fixture_id"], {}).get("status") == "superseded"),
                "raw_cross_ns": sum(1 for r in raw if anchor_map.get(r["fixture_id"], {}).get("namespace") != case.get("namespace")),
            }

        b0 = {}
        for case in cases:
            b0[case["case_id"]] = case.get("expected_fixture_ids", [])[:3]

        calib = policy_sweep(cases, anchor_map, eligible, b0)

        # DB type readback
        type_counts = {}
        try:
            raw_types = docker(["exec", container, "python3", "-c",
                                f"import json,sqlite3; c=sqlite3.connect('file:{db_path}?mode=ro',uri=True); "
                                f"print(json.dumps(dict(c.execute('SELECT mem_type,count(*) FROM memories GROUP BY mem_type').fetchall()))); c.close()"],
                               timeout=15)
            type_counts = json.loads(raw_types.stdout)
        except Exception:
            type_counts = {"readback_failed": True}

        selected = calib["selected_policy"]
        status = "CALIBRATION_FEASIBLE_STOP_BEFORE_HOLDOUT" if selected else "CALIBRATION_INFEASIBLE_STOP_BEFORE_HOLDOUT"

        report = {
            "schema": "c8.7.corrected-calibration-report/v1",
            "status": status,
            "synthetic_only": True,
            "actual_nyawa": True,
            "backend": "bge-onnx-local",
            "calibration": calib,
            "policy_gates_unchanged": {"minimum": MIN_GATES, "maximum": MAX_GATES},
            "abstention": {"signal": "Nyawa recall score threshold", "margin_used": False},
            "eligibility": {"basis": ["namespace", "status", "supersedes"], "labels_used": False,
                            "raw_retrieval_retained": True},
            "fixture_counts": {"calibration_cases": len(cases), "calibration_records": len(cal_records),
                               "holdout_cases": len(holdout_cases)},
            "store_type_readback": type_counts,
            "raw_retrieval_diagnostics": raw_diag,
            "runtime_attestation": {"mode": "local", "network": "blocked", "container": container},
            "isolation": {"network": "docker --network=none", "production_access": False},
            "holdout_accessed": False, "holdout_rows_read": False, "holdout_db_created": False,
            "paths": {"fixture": str(fixture_path)},
            "image": image,
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "queries_run": len(cases),
        }
        return report
    finally:
        cleanup(container, image)


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
class C87Tests(unittest.TestCase):
    def test_fixture_counts(self):
        from c87_fixture_generator import build_dataset as bd
        ds = bd()
        self.assertEqual(len(ds["records"]), 80)
        self.assertEqual(len(ds["cases"]), 120)
        self.assertEqual(sum(1 for c in ds["cases"] if c["split"] == "calibration"), 60)

    def test_type_map(self):
        self.assertEqual(supported_mcp_type("artifact"), "reference")
        self.assertEqual(supported_mcp_type("constraint"), "procedure")
        self.assertEqual(supported_mcp_type("decision"), "decision")
        self.assertEqual(supported_mcp_type("fact"), "fact")
        with self.assertRaises(RuntimeError):
            supported_mcp_type("garbage")

    def test_eligibility(self):
        anchors = {
            "a1": {"namespace": "ns", "status": "active", "supersedes": []},
            "a2": {"namespace": "ns", "status": "superseded", "supersedes": []},
            "a3": {"namespace": "other", "status": "active", "supersedes": []},
            "a4": {"namespace": "ns", "status": "active", "supersedes": ["old"]},
        }
        case = {"namespace": "ns"}
        raw = [{"fixture_id": k, "score": 0.9} for k in anchors]
        el = eligible_candidates(raw, case, anchors)
        self.assertEqual([r["fixture_id"] for r in el], ["a1"])

    def test_policy_sweep(self):
        from c87_fixture_generator import build_dataset as bd
        ds = bd()
        cal = [c for c in ds["cases"] if c["split"] == "calibration"]
        anchors = {r["fixture_id"]: {"namespace": r["namespace"], "status": r["status"],
                                     "supersedes": r.get("supersedes", [])}
                   for r in ds["records"]}
        el = {}
        for case in cal:
            el[case["case_id"]] = [{
                "fixture_id": eid, "score": 0.9
            } for eid in case.get("expected_fixture_ids", [])[:3]]
        for case in cal:
            if case["category"] == "negative_confusable":
                el[case["case_id"]] = [{"fixture_id": "nonexistent", "score": 0.1}]
        b0 = {}
        result = policy_sweep(cal, anchors, el, b0)
        self.assertEqual(result["case_count"], 60)
        self.assertGreater(result["score_threshold_count"], 0)
        self.assertGreater(result["policy_sweep_count"], 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "e2e"), required=True)
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--image", default=IMAGE_C87)
    parser.add_argument("--container", default="nyawa-c87-eval")
    args = parser.parse_args()

    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(C87Tests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1

    if args.mode == "e2e":
        if not args.fixtures or not args.fixtures.is_file():
            print("error: --fixtures is required for e2e mode", file=sys.stderr)
            return 1
        report = run_e2e(args.container, args.image, args.fixtures)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

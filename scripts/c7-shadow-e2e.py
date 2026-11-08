#!/usr/bin/env python3
"""Ephemeral Nyawa/BGE retrieval check for C7 shadow mode.

This verifies real Nyawa API retrieval over synthetic records, namespace
isolation, and an observation-only result envelope. It does not load Hermes,
call a model provider, or claim to test a Hermes plugin integration.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any

IMAGE = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:61da25caf922c0925f4fd8e3af1d811f6fc9d3a9df21beb44138c7300d47efb6"
)
MEMORY_LIMIT = 4 * 1024**3
MIN_AVAILABLE_MEMORY = 5 * 1024**3
PORT = 3300

# Execute HTTP inside the isolated container: its network namespace is `none`,
# so no host port or bridge network is needed.
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


def observation_envelope(namespace: str, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a bounded observation record; it has no injection/provider field."""
    safe_candidates = []
    for item in candidates[:5]:
        content = str(item.get("Content", item.get("content", "")))
        score = item.get("Score", item.get("score"))
        safe_candidates.append(
            {
                "id": str(item.get("ID", item.get("id", ""))),
                "content": content[:500],
                "score": score if isinstance(score, (int, float)) else None,
            }
        )
    return {
        "mode": "shadow_observation",
        "namespace": namespace,
        "query": query,
        "injected": False,
        "candidates": safe_candidates,
    }


def preflight() -> None:
    meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
    available_kib = next(
        int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("MemAvailable:")
    )
    available = available_kib * 1024
    if available < MIN_AVAILABLE_MEMORY:
        raise RuntimeError(
            f"runner preflight failed: MemAvailable={available} bytes; "
            f"need at least {MIN_AVAILABLE_MEMORY} bytes; container not started"
        )
    if os.cpu_count() is None or os.cpu_count() < 2:
        raise RuntimeError("runner preflight failed: at least 2 CPUs required; container not started")
    stat = os.statvfs(os.environ.get("RUNNER_TEMP", "/tmp"))
    free_bytes = stat.f_bavail * stat.f_frsize
    if free_bytes < 4 * 1024**3:
        raise RuntimeError("runner preflight failed: less than 4 GiB free disk; container not started")
    print(
        json.dumps(
            {
                "preflight": "PASS",
                "mem_available_bytes": available,
                "cpu_count": os.cpu_count(),
                "runner_temp_free_bytes": free_bytes,
            },
            sort_keys=True,
        )
    )


def docker(args: list[str], timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["docker", *args], text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(
            f"docker {' '.join(args[:3])} failed ({result.returncode}): "
            f"{(result.stderr or result.stdout).strip()[-1200:]}"
        )
    return result


def inside_http(container: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = "-" if payload is None else json.dumps(payload, separators=(",", ":"))
    result = docker(
        ["exec", container, "python3", "-c", HTTP_HELPER, method, path, raw],
        timeout=45,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"in-container HTTP helper returned invalid JSON: {result.stdout[:300]!r}") from error


def verify_container(container: str, image: str) -> None:
    inspected = json.loads(docker(["inspect", container]).stdout)[0]
    host = inspected["HostConfig"]
    if inspected["Config"]["Image"] != image:
        raise RuntimeError("container image reference does not match the pinned digest")
    if host.get("NetworkMode") != "none":
        raise RuntimeError("container is not isolated with network=none")
    if not host.get("ReadonlyRootfs"):
        raise RuntimeError("container root filesystem is not read-only")
    if host.get("PortBindings"):
        raise RuntimeError("container unexpectedly has a published port")
    if host.get("Memory") != MEMORY_LIMIT or host.get("MemorySwap") != MEMORY_LIMIT:
        raise RuntimeError("container memory/swap limit does not match the 4 GiB bound")
    if host.get("PidsLimit") != 128:
        raise RuntimeError("container PID limit does not match the expected bound")
    if any(mount.get("Type") != "tmpfs" for mount in inspected.get("Mounts", [])):
        raise RuntimeError("container has a non-tmpfs mount")
    if host.get("Binds") or host.get("Mounts"):
        raise RuntimeError("container has a bind/volume mount")


def cleanup(container: str, image: str) -> None:
    check = docker(["inspect", container], timeout=15, check=False)
    if check.returncode != 0:
        return
    inspected = json.loads(check.stdout)[0]
    labels = inspected.get("Config", {}).get("Labels") or {}
    if labels.get("org.nyawa.c7-shadow-e2e") != "true" or inspected["Config"].get("Image") != image:
        raise RuntimeError("refusing to remove a container not created by this C7 run")
    docker(["rm", "--force", container], timeout=30)
    check = docker(["inspect", container], timeout=15, check=False)
    if check.returncode == 0:
        raise RuntimeError("cleanup verification failed: disposable container still exists")


def run_e2e(container: str, image: str) -> dict[str, Any]:
    if image != IMAGE:
        raise RuntimeError("image must equal the reviewed immutable BGE r1 digest")
    preflight()
    docker(["pull", image], timeout=300)
    # Ensure the pull resolved to the exact immutable digest before starting it.
    inspect = json.loads(docker(["image", "inspect", image]).stdout)[0]
    expected_repo_digest = image
    if expected_repo_digest not in inspect.get("RepoDigests", []):
        raise RuntimeError("pulled image RepoDigests do not contain the exact approved digest")
    existing = docker(["inspect", container], timeout=15, check=False)
    if existing.returncode == 0:
        raise RuntimeError("container name already exists; refusing to replace it")

    try:
        docker(
            [
                "run",
                "--detach",
                "--name",
                container,
                "--label",
                "org.nyawa.c7-shadow-e2e=true",
                "--network=none",
                "--read-only",
                "--tmpfs",
                "/scratch:rw,nosuid,nodev,size=768m",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=256m",
                "--memory=4g",
                "--memory-swap=4g",
                "--cpus=2",
                "--pids-limit=128",
                "--restart=no",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges:true",
                "--env",
                "MEMORY_EMBEDDER=local",
                "--env",
                "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model",
                "--env",
                "HOME=/tmp",
                "--env",
                "TMPDIR=/tmp",
                image,
                "serve",
                "/scratch/c7-shadow.db",
            ],
            timeout=60,
        )
        verify_container(container, image)

        deadline = time.monotonic() + 180
        root: dict[str, Any] | None = None
        health: dict[str, Any] | None = None
        last_error = "not yet ready"
        while time.monotonic() < deadline:
            state = json.loads(docker(["inspect", "--format", "{{json .State}}", container]).stdout)
            if not state.get("Running"):
                logs = docker(["logs", container], timeout=15, check=False).stdout[-1200:]
                raise RuntimeError(f"Nyawa container exited before readiness (exit={state.get('ExitCode')}): {logs}")
            try:
                root = inside_http(container, "GET", "/")
                health = inside_http(container, "GET", "/v1/health")
                if root["status"] == 200 and health["status"] == 200:
                    break
                last_error = f"root={root['status']} health={health['status']}"
            except Exception as error:  # readiness retry only; final timeout reports last reason
                last_error = str(error)
            time.sleep(2)
        else:
            logs = docker(["logs", container], timeout=15, check=False).stdout[-1200:]
            raise RuntimeError(f"Nyawa API did not become ready in 180s: {last_error}; logs={logs}")

        root_body = root["body"]
        if root_body.get("version") != "1.2.4" or root_body.get("status") != "running":
            raise RuntimeError(f"unexpected Nyawa runtime identity: {root_body}")

        # Store two synthetic records in isolated namespaces. The API and BGE
        # must index both; no real/user/production content is used.
        fixtures = [
            {
                "namespace": "c7-shadow-synthetic-chat-a",
                "marker": "C7_SYNTH_A_BLUE_CEDAR_384",
                "content": (
                    "Synthetic C7 shadow record C7_SYNTH_A_BLUE_CEDAR_384. "
                    "The synthetic note says the blue cedar vector belongs to chat A. "
                    "Exact fixture path /synthetic/c7/a.md and URL https://example.test/c7/a."
                ),
                "query": "C7_SYNTH_A_BLUE_CEDAR_384 blue cedar vector chat A",
            },
            {
                "namespace": "c7-shadow-synthetic-chat-b",
                "marker": "C7_SYNTH_B_AMBER_MAPLE_384",
                "content": (
                    "Synthetic C7 shadow record C7_SYNTH_B_AMBER_MAPLE_384. "
                    "The synthetic note says the amber maple vector belongs to chat B. "
                    "Exact fixture path /synthetic/c7/b.md and URL https://example.test/c7/b."
                ),
                "query": "C7_SYNTH_B_AMBER_MAPLE_384 amber maple vector chat B",
            },
        ]
        for fixture in fixtures:
            stored = inside_http(
                container,
                "POST",
                "/v1/memories",
                {"content": fixture["content"], "namespace": fixture["namespace"], "type": "fact"},
            )
            if stored["status"] != 201 or not stored["body"].get("id"):
                raise RuntimeError(f"synthetic memory store failed in {fixture['namespace']}: {stored}")

        stats = inside_http(container, "GET", "/v1/stats")
        if stats["status"] != 200:
            raise RuntimeError(f"Nyawa stats returned HTTP {stats['status']}")
        store_stats = stats["body"].get("store", stats["body"])
        vector_indexed = int(store_stats.get("vector_indexed", 0))
        if vector_indexed < 2:
            raise RuntimeError(f"BGE did not vector-index both synthetic records: {store_stats}")

        candidates_by_namespace: dict[str, list[dict[str, Any]]] = {}
        retrieval_ms_by_namespace: dict[str, float] = {}
        scores_by_namespace: dict[str, list[float]] = {}
        for fixture in fixtures:
            recall_started = time.monotonic()
            recalled = inside_http(
                container,
                "POST",
                "/v1/recall",
                {"query": fixture["query"], "namespace": fixture["namespace"], "limit": 5, "min_score": 0},
            )
            if recalled["status"] != 200:
                raise RuntimeError(f"recall failed in {fixture['namespace']}: HTTP {recalled['status']}")
            candidates = recalled["body"].get("results", [])
            if not any(fixture["marker"] in item.get("Content", item.get("content", "")) for item in candidates):
                raise RuntimeError(f"synthetic marker was not retrieved in {fixture['namespace']}")
            other_marker = next(x["marker"] for x in fixtures if x is not fixture)
            if any(other_marker in item.get("Content", item.get("content", "")) for item in candidates):
                raise RuntimeError(f"cross-namespace result leaked into {fixture['namespace']}")
            if not all(isinstance(item.get("Score", item.get("score")), (int, float)) for item in candidates):
                raise RuntimeError("recall candidates did not include numeric scores")
            retrieval_ms_by_namespace[fixture["namespace"]] = round(
                (time.monotonic() - recall_started) * 1000, 2
            )
            scores_by_namespace[fixture["namespace"]] = [
                float(item.get("Score", item.get("score"))) for item in candidates
            ]
            candidates_by_namespace[fixture["namespace"]] = candidates

        observations = [
            observation_envelope(f["namespace"], f["query"], candidates_by_namespace[f["namespace"]])
            for f in fixtures
        ]
        for observation in observations:
            if observation.get("injected") is not False or "context" in observation or "provider_request" in observation:
                raise RuntimeError("shadow observation envelope violates no-injection contract")

        logs = docker(["logs", container], timeout=15, check=False).stdout
        if "dim=384" not in logs:
            raise RuntimeError("Nyawa/BGE runtime log did not confirm dim=384")

        return {
            "status": "PASS",
            "scope": "synthetic Nyawa/BGE retrieval and namespace-scoped shadow observation only",
            "image": image,
            "runtime_version": root_body["version"],
            "embedding_dimension": 384,
            "vector_indexed": vector_indexed,
            "namespace_checks": len(fixtures),
            "synthetic_markers_retrieved": [f["marker"] for f in fixtures],
            "retrieval_ms_by_namespace": retrieval_ms_by_namespace,
            "scores_by_namespace": scores_by_namespace,
            "cross_namespace_leak": False,
            "shadow_injected": False,
            "provider_or_hermes_called": False,
        }
    finally:
        # Idempotent on errors, including startup/readiness failure.
        cleanup(container, image)


class ShadowEnvelopeTests(unittest.TestCase):
    def test_observation_envelope_is_not_context_injection(self) -> None:
        envelope = observation_envelope(
            "synthetic-ns",
            "synthetic query",
            [{"ID": "synthetic-id", "Content": "synthetic content", "Score": 0.75}],
        )
        self.assertEqual(envelope["mode"], "shadow_observation")
        self.assertIs(envelope["injected"], False)
        self.assertNotIn("context", envelope)
        self.assertNotIn("provider_request", envelope)
        self.assertEqual(envelope["candidates"][0]["score"], 0.75)

    def test_candidate_output_is_bounded(self) -> None:
        candidates = [{"ID": str(i), "Content": "x" * 1000, "Score": 0.5} for i in range(8)]
        envelope = observation_envelope("ns", "q", candidates)
        self.assertLessEqual(len(envelope["candidates"]), 5)
        self.assertTrue(all(len(item["content"]) <= 500 for item in envelope["candidates"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "self-test", "e2e"))
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--container", default="nyawa-c7-shadow-e2e")
    args = parser.parse_args()
    if args.mode == "preflight":
        preflight()
        return 0
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShadowEnvelopeTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        report = run_e2e(args.container, args.image)
        print(json.dumps(report, indent=2, sort_keys=True))
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as stream:
                stream.write("## C7 Nyawa/BGE Shadow E2E\n\n")
                stream.write("```json\n" + json.dumps(report, indent=2, sort_keys=True) + "\n```\n")
                stream.write("\nThis validates synthetic retrieval and namespace isolation, not Hermes plugin integration.\n")
        return 0
    except Exception as error:
        print(f"C7 E2E FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

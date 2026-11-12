#!/usr/bin/env python3
"""Manual C8.6 calibration with OpenRouter-hosted Qwen3 Reranker; synthetic data only."""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import c81_abstention_calibration as calibration
import c86_fixture_generator as fixture_generator

MODEL_ID = "qwen/qwen3-reranker-0.6b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/rerank"
C86_IMAGE = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:57fb872aca5dd9bf24547d33d920787fb6fef73a8f6e63087f8e785fc27c2fea"
)
ANSWERABILITY_INSTRUCTION = (
    "Decide whether the document contains sufficient explicit evidence to answer the query. "
    "Only rank a document highly when it supports a direct factual answer; topical similarity, "
    "an absent attribute, or a statement that a value is unknown is not sufficient."
)


def format_api_query(query: str) -> str:
    """Use Qwen's instruction-aware rerank endpoint for answerability scoring."""
    return f"<Instruct>: {ANSWERABILITY_INSTRUCTION}\n<Query>: {query}"


def _write_generated_fixture(path: Path) -> dict[str, Any]:
    """Write only the checked-in synthetic generator's dataset; no caller fixture input."""
    dataset = fixture_generator.build_dataset()
    fixture_generator.validate_dataset(dataset)
    records = dataset.get("records", [])
    cases = dataset.get("cases", [])
    if dataset.get("schema_version") != fixture_generator.SCHEMA_C86 or len(records) != 164 or len(cases) != 300:
        raise RuntimeError("generated C8.6 fixture provenance/count validation failed")
    if any(
        record.get("split") not in {"calibration", "holdout"}
        or not str(record.get("fixture_id", "")).startswith("C86_")
        or not str(record.get("namespace", "")).startswith("c86-synthetic-")
        or not str(record.get("content", "")).startswith("Synthetic C8.6 fixture. Marker C86_")
        for record in records
    ):
        raise RuntimeError("generated C8.6 records failed synthetic provenance validation")
    if any(
        case.get("split") not in {"calibration", "holdout"}
        or not str(case.get("case_id", "")).startswith("Q86_")
        or not str(case.get("namespace", "")).startswith("c86-synthetic-")
        for case in cases
    ):
        raise RuntimeError("generated C8.6 cases failed synthetic provenance validation")
    path.write_text(json.dumps(dataset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def _temporary_root() -> Path:
    """Use the hosted runner's temp directory or the configured local scratch directory."""
    value = os.environ.get("RUNNER_TEMP") or os.environ.get("C86_SCRATCH_DIR") or "/opt/data/cache/scratch"
    root = Path(value)
    if not root.is_dir():
        raise RuntimeError("temporary fixture root is unavailable; set RUNNER_TEMP or C86_SCRATCH_DIR")
    return root


_SAFE_STAGES = frozenset({"calibration_preflight", "run_e2e", "openrouter_rerank"})


def _write_stage_event(
    stage: str,
    event: str,
    elapsed_ms: float | None = None,
    batch_index: int | None = None,
    batch_size: int | None = None,
) -> None:
    """Emit only allowlisted stage names and aggregate timing/count metadata."""
    try:
        if stage not in _SAFE_STAGES or event not in {"start", "complete", "failed"}:
            return
        fields = [f"[qwen-stage] stage={stage}", f"event={event}"]
        if elapsed_ms is not None:
            fields.append(f"elapsed_ms={elapsed_ms:.1f}")
        if batch_index is not None:
            fields.append(f"request_index={batch_index}")
        if batch_size is not None:
            fields.append(f"document_count={batch_size}")
        print(" ".join(fields), file=sys.stderr, flush=True)
    except Exception:
        pass


@contextmanager
def _stage(stage: str, *, batch_index: int | None = None, batch_size: int | None = None) -> Iterator[None]:
    started = time.perf_counter()
    _write_stage_event(stage, "start", batch_index=batch_index, batch_size=batch_size)
    try:
        yield
    except BaseException:
        _write_stage_event(stage, "failed", (time.perf_counter() - started) * 1000, batch_index, batch_size)
        raise
    else:
        _write_stage_event(stage, "complete", (time.perf_counter() - started) * 1000, batch_index, batch_size)


class OpenRouterReranker:
    """Fail-closed OpenRouter client; logs never include key, query, document, or response body."""

    def __init__(
        self,
        api_key: str,
        allowed_queries: set[str],
        allowed_documents: set[str],
        timeout_seconds: int = 90,
        opener=None,
    ) -> None:
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required")
        if not allowed_queries or not allowed_documents:
            raise RuntimeError("refusing API use without a generated synthetic calibration allowlist")
        self._api_key = api_key
        self.allowed_queries = frozenset(allowed_queries)
        self.allowed_documents = frozenset(allowed_documents)
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self.request_count = 0
        self.total_tokens = 0

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        if query not in self.allowed_queries:
            raise RuntimeError("refusing to send query outside generated calibration fixture")
        if any(document not in self.allowed_documents for document in documents):
            raise RuntimeError("refusing to send candidate outside generated calibration fixture")
        body = json.dumps({
            "model": MODEL_ID,
            "query": format_api_query(query),
            "documents": documents,
            "top_n": len(documents),
            "provider": {"allow_fallbacks": False},
        }).encode("utf-8")
        request = urllib.request.Request(
            OPENROUTER_URL,
            data=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        self.request_count += 1
        with _stage("openrouter_rerank", batch_index=self.request_count, batch_size=len(documents)):
            try:
                opener = self._opener or urllib.request.urlopen
                with opener(request, timeout=self.timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"OpenRouter rerank API returned HTTP {error.code}") from None
            except urllib.error.URLError as error:
                raise RuntimeError(f"OpenRouter rerank request failed ({type(error.reason).__name__})") from None
            except (TimeoutError, OSError) as error:
                raise RuntimeError(f"OpenRouter rerank transport failed ({type(error).__name__})") from None
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise RuntimeError("OpenRouter rerank API returned invalid JSON") from None

        rows = result.get("results") if isinstance(result, dict) else None
        if not isinstance(rows, list) or len(rows) != len(documents):
            raise RuntimeError("OpenRouter rerank response did not score every candidate")
        scores: list[float | None] = [None] * len(documents)
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError("OpenRouter rerank response contains a malformed result")
            index = row.get("index")
            score = row.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(documents):
                raise RuntimeError("OpenRouter rerank response contains an invalid candidate index")
            if scores[index] is not None:
                raise RuntimeError("OpenRouter rerank response contains a duplicate candidate index")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RuntimeError("OpenRouter rerank response contains a nonnumeric score")
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise RuntimeError("OpenRouter rerank response contains a nonfinite score")
            scores[index] = numeric_score
        if any(score is None for score in scores):
            raise RuntimeError("OpenRouter rerank response omitted a candidate")
        usage = result.get("usage", {})
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            self.total_tokens += usage["total_tokens"]
        return [score for score in scores if score is not None]


class QwenPilotTests(unittest.TestCase):
    def test_query_contains_answerability_instruction_but_not_candidate_text(self) -> None:
        text = format_api_query("What is the recovery code?")
        self.assertIn(ANSWERABILITY_INSTRUCTION, text)
        self.assertIn("<Query>: What is the recovery code?", text)
        self.assertNotIn("PRIVATE_DOCUMENT", text)

    def test_api_scores_are_mapped_back_to_original_candidate_order(self) -> None:
        response = json.dumps({
            "model": MODEL_ID,
            "results": [
                {"index": 1, "relevance_score": 0.12},
                {"index": 0, "relevance_score": 0.91},
            ],
            "usage": {"total_tokens": 42},
        }).encode()
        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response

        def fake_open(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return FakeResponse()

        scorer = OpenRouterReranker(
            "test-secret",
            allowed_queries={"SYNTHETIC_QUERY_SENTINEL"},
            allowed_documents={"SYNTHETIC_DOC_A", "SYNTHETIC_DOC_B"},
            opener=fake_open,
        )
        scores = scorer("SYNTHETIC_QUERY_SENTINEL", ["SYNTHETIC_DOC_A", "SYNTHETIC_DOC_B"])
        self.assertEqual(scores, [0.91, 0.12])
        self.assertEqual(scorer.request_count, 1)
        self.assertEqual(scorer.total_tokens, 42)
        self.assertEqual(captured["url"], OPENROUTER_URL)
        self.assertEqual(captured["body"]["model"], MODEL_ID)
        self.assertEqual(captured["body"]["top_n"], 2)
        self.assertEqual(captured["body"]["documents"], ["SYNTHETIC_DOC_A", "SYNTHETIC_DOC_B"])
        self.assertFalse(captured["body"]["provider"]["allow_fallbacks"])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-secret")
        self.assertEqual(captured["timeout"], 90)

    def test_incomplete_api_scores_fail_closed(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"results":[{"index":0,"relevance_score":0.5}]}'

        scorer = OpenRouterReranker(
            "test-secret", allowed_queries={"synthetic"}, allowed_documents={"one", "two"},
            opener=lambda *a, **k: FakeResponse(),
        )
        with self.assertRaisesRegex(RuntimeError, "did not score every candidate"):
            scorer("synthetic", ["one", "two"])

    def test_non_allowlisted_query_or_candidate_never_calls_provider(self) -> None:
        calls = []
        scorer = OpenRouterReranker(
            "test-secret",
            allowed_queries={"synthetic calibration query"},
            allowed_documents={"Synthetic C8.6 fixture. Marker C86_CAL_001."},
            opener=lambda *a, **k: calls.append(a),
        )
        with self.assertRaisesRegex(RuntimeError, "query outside generated calibration fixture"):
            scorer("production-looking query", ["Synthetic C8.6 fixture. Marker C86_CAL_001."])
        with self.assertRaisesRegex(RuntimeError, "candidate outside generated calibration fixture"):
            scorer("synthetic calibration query", ["arbitrary candidate"])
        self.assertEqual(calls, [])

    def test_stage_markers_are_redacted_and_preserve_original_exceptions(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        sentinel = "PRIVATE_QUERY_DOCUMENT_SENTINEL"
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            with self.assertRaisesRegex(RuntimeError, sentinel):
                with _stage("openrouter_rerank", batch_index=2, batch_size=5):
                    raise RuntimeError(sentinel)
        trace = stderr.getvalue()
        self.assertIn("stage=openrouter_rerank event=start", trace)
        self.assertIn("request_index=2 document_count=5", trace)
        self.assertNotIn(sentinel, trace)
        self.assertEqual(stdout.getvalue(), "")

    def test_trace_output_failure_does_not_mask_stage_exception(self) -> None:
        class BrokenStream:
            def write(self, text: str) -> int:
                raise OSError("diagnostic stream unavailable")

            def flush(self) -> None:
                raise OSError("diagnostic stream unavailable")

        with mock.patch("sys.stderr", BrokenStream()):
            with self.assertRaisesRegex(RuntimeError, "original stage failure"):
                with _stage("run_e2e"):
                    raise RuntimeError("original stage failure")

    def test_trace_output_preserves_interruption_signals(self) -> None:
        for signal in (KeyboardInterrupt, SystemExit):
            with self.subTest(signal=signal.__name__):
                class InterruptingStream:
                    def write(self, text: str) -> int:
                        raise signal("diagnostic interruption")

                    def flush(self) -> None:
                        raise signal("diagnostic interruption")

                with mock.patch("sys.stderr", InterruptingStream()):
                    with self.assertRaisesRegex(signal, "diagnostic interruption"):
                        with _stage("run_e2e"):
                            self.fail("stage body should not run")

    def test_main_generates_fixture_internally_and_never_queries_holdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        report = {"status": "PASS", "calibration_policy_feasible": True}
        captured = {}
        runner_temp = os.environ.get("RUNNER_TEMP") or "/opt/data/cache/scratch"

        class FakeScorer:
            request_count = 3
            total_tokens = 42

        def fake_run(container, image, fixture_path, **kwargs):
            dataset = calibration.load_dataset(fixture_path)
            captured["path"] = Path(fixture_path)
            captured["dataset"] = dataset
            captured["kwargs"] = kwargs
            return dict(report)

        with (
            mock.patch.object(sys, "argv", ["c86_qwen_answerability.py", "--mode", "calibration-only"]),
            mock.patch.object(sys, "stdout", stdout),
            mock.patch.object(sys, "stderr", stderr),
            mock.patch.object(calibration, "preflight"),
            mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-secret", "RUNNER_TEMP": runner_temp}),
            mock.patch.object(sys.modules[__name__], "OpenRouterReranker", return_value=FakeScorer()),
            mock.patch.object(calibration, "run_e2e", side_effect=fake_run) as run_e2e,
        ):
            self.assertEqual(main(), 0)

        self.assertFalse(captured["kwargs"]["evaluate_holdout"])
        self.assertEqual(captured["dataset"]["schema_version"], fixture_generator.SCHEMA_C86)
        self.assertEqual(len(captured["dataset"]["cases"]), 300)
        self.assertEqual(sum(row["split"] == "calibration" for row in captured["dataset"]["cases"]), 150)
        self.assertFalse(captured["path"].exists(), "generated fixture temp directory should be removed")
        self.assertEqual(captured["path"].parent.parent, Path(runner_temp))
        output_report = json.loads(stdout.getvalue())
        self.assertTrue(output_report["provider_called"])
        self.assertEqual(output_report["openrouter_request_count"], 3)
        self.assertNotIn("test-secret", stdout.getvalue())
        self.assertIn("stage=calibration_preflight event=complete", stderr.getvalue())
        self.assertIn("stage=run_e2e event=complete", stderr.getvalue())
        self.assertEqual(run_e2e.call_count, 1)

    def test_cli_rejects_custom_fixture_path(self) -> None:
        with mock.patch.object(sys, "argv", [
            "c86_qwen_answerability.py", "--mode", "calibration-only", "--fixtures", "/unsafe.json",
        ]):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "calibration-only"), required=True)
    parser.add_argument("--image", default=C86_IMAGE)
    parser.add_argument("--container", default="nyawa-c86-qwen-eval")
    args = parser.parse_args()
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(QwenPilotTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("export OPENROUTER_API_KEY before running the manual C8.6 pilot")
        with _stage("calibration_preflight"):
            calibration.preflight()
        with tempfile.TemporaryDirectory(prefix="c86-openrouter-", dir=_temporary_root()) as temp_dir:
            fixture_path = Path(temp_dir) / "c86-qwen-calibration-fixtures.json"
            dataset = _write_generated_fixture(fixture_path)
            calibration_cases = [case for case in dataset["cases"] if case["split"] == "calibration"]
            calibration_documents = [record["content"] for record in dataset["records"] if record["split"] == "calibration"]
            scorer = OpenRouterReranker(
                api_key,
                allowed_queries={case["query"] for case in calibration_cases},
                allowed_documents=set(calibration_documents),
            )
            with _stage("run_e2e"):
                report = calibration.run_e2e(
                    args.container, args.image, fixture_path,
                    candidate_scorer=scorer, qwen_only=True, preflight_passed=True,
                    evaluate_holdout=False,
                )
            report.update({
                "answerability_model": MODEL_ID,
                "answerability_score_type": "OpenRouter rerank relevance_score",
                "answerability_instruction": ANSWERABILITY_INSTRUCTION,
                "qwen_policy_only": True,
                "provider_called": scorer.request_count > 0,
                "openrouter_request_count": scorer.request_count,
                "openrouter_total_tokens": scorer.total_tokens,
                "fixture_source": "checked-in c86_fixture_generator.py",
            })
            print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("calibration_policy_feasible", False) else 1
    except Exception as error:
        print(f"C8.6 OpenRouter Qwen rerank pilot FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

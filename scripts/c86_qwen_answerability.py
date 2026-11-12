#!/usr/bin/env python3
"""Local Qwen3 yes/no answerability scoring for the C8.6 calibration pilot."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import c81_abstention_calibration as calibration

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MODEL_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
C86_IMAGE = (
    "ghcr.io/rezkyauliapratama/nyawa/bge-runtime@"
    "sha256:57fb872aca5dd9bf24547d33d920787fb6fef73a8f6e63087f8e785fc27c2fea"
)
ANSWERABILITY_INSTRUCTION = (
    "Decide whether the document contains sufficient explicit evidence to answer the query. "
    "Answer yes only when a direct factual answer is supported by the document; topical similarity, "
    "an absent attribute, or a mention that a value is unknown is not sufficient."
)
SYSTEM_INSTRUCTION = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
)


def format_prompt(query: str, document: str) -> str:
    """Render the Qwen3-Reranker chat prompt with thinking disabled for yes/no logits."""
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_INSTRUCTION}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<Instruct>: {ANSWERABILITY_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def yes_no_margin(yes_logit: float, no_logit: float) -> float:
    """Return the unnormalized answerability score: yes logit minus no logit."""
    return float(yes_logit) - float(no_logit)


def last_token_column(sequence_length: int) -> int:
    """Qwen uses left padding; the final prompt token is the last padded column."""
    if sequence_length <= 0:
        raise ValueError("tokenized prompt must contain at least one token")
    return sequence_length - 1


_SAFE_STAGES = frozenset({
    "calibration_preflight",
    "tokenizer_from_pretrained",
    "model_from_pretrained",
    "model_to_device",
    "run_e2e",
    "inference_batch",
})


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
            fields.append(f"batch_index={batch_index}")
        if batch_size is not None:
            fields.append(f"batch_size={batch_size}")
        print(" ".join(fields), file=sys.stderr, flush=True)
    except Exception:
        # Ordinary diagnostic failures must not affect model execution.
        pass


@contextmanager
def _stage(
    stage: str,
    *,
    batch_index: int | None = None,
    batch_size: int | None = None,
) -> Iterator[None]:
    """Log privacy-safe start/finish markers without changing stage failures."""
    started = None
    try:
        started = time.perf_counter()
    except Exception:
        pass
    _write_stage_event(stage, "start", batch_index=batch_index, batch_size=batch_size)
    try:
        yield
    except BaseException:
        elapsed_ms = None
        try:
            if started is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception:
            pass
        _write_stage_event(stage, "failed", elapsed_ms, batch_index, batch_size)
        raise
    else:
        elapsed_ms = None
        try:
            if started is not None:
                elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception:
            pass
        _write_stage_event(stage, "complete", elapsed_ms, batch_index, batch_size)


class LocalQwenAnswerabilityScorer:
    """Batched local causal-LM scorer; no provider or remote inference path exists."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        torch_module: Any,
        device: str,
        batch_size: int = 8,
        max_length: int = 4096,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.torch = torch_module
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise ValueError("Qwen tokenizer must encode yes/no as single tokens")
        self.yes_token_id = int(yes_ids[0])
        self.no_token_id = int(no_ids[0])
        self.model.eval()

    @classmethod
    def from_pretrained(cls) -> "LocalQwenAnswerabilityScorer":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
        with _stage("tokenizer_from_pretrained"):
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False,
                padding_side="left",
            )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        with _stage("model_from_pretrained"):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                trust_remote_code=False,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        with _stage("model_to_device"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
        return cls(tokenizer, model, torch, device)

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        prompts = [format_prompt(query, document) for document in documents]
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_index = start // self.batch_size + 1
            batch_size = min(self.batch_size, len(prompts) - start)
            with _stage(
                "inference_batch", batch_index=batch_index, batch_size=batch_size,
            ):
                batch = self.tokenizer(
                    prompts[start : start + self.batch_size],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {name: value.to(self.device) for name, value in batch.items()}
                with self.torch.inference_mode():
                    logits = self.model(**inputs).logits
                    if not bool(self.torch.all(inputs["attention_mask"][:, -1] == 1)):
                        raise RuntimeError("left-padded Qwen batch has a masked final token")
                    positions = self.torch.full(
                        (logits.size(0),), last_token_column(logits.size(1)),
                        dtype=self.torch.long, device=self.device,
                    )
                    rows = self.torch.arange(logits.size(0), device=self.device)
                    yes_logits = logits[rows, positions, self.yes_token_id]
                    no_logits = logits[rows, positions, self.no_token_id]
                    yes_values = yes_logits.detach().float().cpu().tolist()
                    no_values = no_logits.detach().float().cpu().tolist()
                scores.extend(yes_no_margin(yes, no) for yes, no in zip(yes_values, no_values))
        return scores


class QwenPilotTests(unittest.TestCase):
    def test_prompt_has_answerability_instruction_and_pair_text(self) -> None:
        prompt = format_prompt("What is the recovery code?", "No recovery code is recorded.")
        self.assertIn(ANSWERABILITY_INSTRUCTION, prompt)
        self.assertIn("<Query>: What is the recovery code?", prompt)
        self.assertIn("<Document>: No recovery code is recorded.", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))

    def test_yes_no_margin_is_numeric_and_ordered(self) -> None:
        self.assertEqual(yes_no_margin(2.25, -0.5), 2.75)
        self.assertEqual(yes_no_margin(-1.0, 1.0), -2.0)

    def test_left_padding_scores_final_prompt_column(self) -> None:
        self.assertEqual(last_token_column(17), 16)
        with self.assertRaisesRegex(ValueError, "at least one token"):
            last_token_column(0)

    def test_fake_components_initialize_without_loading_model_weights(self) -> None:
        class FakeTokenizer:
            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return {"yes": [7], "no": [3]}[text]

        class FakeModel:
            evaluated = False

            def eval(self) -> "FakeModel":
                self.evaluated = True
                return self

        model = FakeModel()
        scorer = LocalQwenAnswerabilityScorer(FakeTokenizer(), model, object(), "cpu")
        self.assertEqual((scorer.yes_token_id, scorer.no_token_id), (7, 3))
        self.assertTrue(model.evaluated)
        with self.assertRaisesRegex(ValueError, "single tokens"):
            LocalQwenAnswerabilityScorer(
                type("MultiTokenTokenizer", (), {"encode": lambda self, text, add_special_tokens=False: [1, 2]})(),
                FakeModel(), object(), "cpu",
            )

    def test_stage_markers_are_redacted_flushed_to_stderr_and_keep_stdout_empty(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        secret = "PRIVATE_QUERY_DOCUMENT_SENTINEL"
        with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
            with self.assertRaisesRegex(RuntimeError, secret):
                with _stage("inference_batch", batch_index=2, batch_size=1):
                    raise RuntimeError(secret)
        trace = stderr.getvalue()
        self.assertIn("stage=inference_batch event=start", trace)
        self.assertIn("stage=inference_batch event=failed elapsed_ms=", trace)
        self.assertIn("batch_index=2 batch_size=1", trace)
        self.assertNotIn(secret, trace)
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

                body_ran = False
                with mock.patch("sys.stderr", InterruptingStream()):
                    with self.assertRaisesRegex(signal, "diagnostic interruption"):
                        with _stage("run_e2e"):
                            body_ran = True
                self.assertFalse(body_ran)

    def test_main_keeps_json_stdout_and_holdout_mode_gate(self) -> None:
        fake_scorer = object()
        for mode, expected_holdout in (("calibration-only", False), ("e2e", True)):
            stdout = io.StringIO()
            stderr = io.StringIO()
            report = {"status": "PASS", "calibration_policy_feasible": True}
            with (
                mock.patch.object(sys, "argv", ["c86_qwen_answerability.py", "--mode", mode]),
                mock.patch.object(sys, "stdout", stdout),
                mock.patch.object(sys, "stderr", stderr),
                mock.patch.object(calibration, "preflight"),
                mock.patch.object(LocalQwenAnswerabilityScorer, "from_pretrained", return_value=fake_scorer),
                mock.patch.object(calibration, "run_e2e", return_value=report) as run_e2e,
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(run_e2e.call_args.kwargs["evaluate_holdout"], expected_holdout)
            output_report = json.loads(stdout.getvalue())
            self.assertEqual(output_report["status"], "PASS")
            self.assertTrue(output_report["qwen_policy_only"])
            self.assertFalse(output_report["provider_called"])
            self.assertNotIn("qwen-stage", stdout.getvalue())
            self.assertIn("stage=calibration_preflight event=complete", stderr.getvalue())
            self.assertIn("stage=run_e2e event=complete", stderr.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "calibration-only", "e2e"), required=True)
    parser.add_argument("--image", default=C86_IMAGE)
    parser.add_argument("--container", default="nyawa-c86-qwen-eval")
    parser.add_argument(
        "--fixtures", type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/c86-abstention-calibration-fixtures.json",
    )
    args = parser.parse_args()
    if args.mode == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(QwenPilotTests)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    try:
        # Refuse small hosts before importing/loading Transformers or downloading weights.
        with _stage("calibration_preflight"):
            calibration.preflight()
        scorer = LocalQwenAnswerabilityScorer.from_pretrained()
        with _stage("run_e2e"):
            report = calibration.run_e2e(
                args.container, args.image, args.fixtures,
                candidate_scorer=scorer, qwen_only=True, preflight_passed=True,
                evaluate_holdout=args.mode == "e2e",
            )
        report.update({
            "answerability_model": MODEL_ID,
            "answerability_model_revision": MODEL_REVISION,
            "answerability_instruction": ANSWERABILITY_INSTRUCTION,
            "qwen_policy_only": True,
            "provider_called": False,
        })
        print(json.dumps(report, indent=2, sort_keys=True))
        if args.mode == "calibration-only":
            return 0 if report.get("calibration_policy_feasible", False) else 1
        return 0 if report["status"] == "PASS" else 1
    except Exception as error:
        print(f"C8.6 Qwen answerability pilot FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

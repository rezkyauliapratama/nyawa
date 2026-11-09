#!/usr/bin/env python3
"""Offline contract tests for the local ONNX cross-encoder JSONL adapter."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "internal/rag/reranker.py"
SPEC = importlib.util.spec_from_file_location("nyawa_local_reranker", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load local reranker module")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeEncoding:
    def __init__(self, text: str) -> None:
        self.ids = [1, len(text), 2]
        self.attention_mask = [1, 1, 1]
        self.type_ids = [0, 0, 0]


class FakeTokenizer:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.pad_settings: dict[str, Any] = {}
        self.truncation_length = 0

    def enable_truncation(self, max_length: int) -> None:
        self.truncation_length = max_length

    def token_to_id(self, token: str) -> int | None:
        return {"[PAD]": None, "<pad>": 0}.get(token)

    def enable_padding(self, **kwargs: Any) -> None:
        self.pad_settings = kwargs

    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[FakeEncoding]:
        self.pairs = pairs
        return [FakeEncoding(query + " " + passage) for query, passage in pairs]


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self) -> None:
        self.received: dict[str, Any] | None = None

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask"), FakeInput("token_type_ids")]

    def run(self, output_names: Any, inputs: dict[str, Any]) -> list[list[list[float]]]:
        self.received = inputs
        return [[[1.25], [-0.75]]]


def test_pair_encoding_and_numeric_scores() -> None:
    tokenizer, session = FakeTokenizer(), FakeSession()
    model = MODULE.LocalCrossEncoder(tokenizer=tokenizer, session=session, array_factory=lambda values: values)
    scores = model.rerank("query", ["relevant passage", "unrelated text"])
    assert scores == [1.25, -0.75], scores
    assert tokenizer.pairs == [("query", "relevant passage"), ("query", "unrelated text")]
    assert tokenizer.truncation_length == 384
    assert tokenizer.pad_settings["pad_id"] == 0
    assert session.received is not None and set(session.received) == {"input_ids", "attention_mask", "token_type_ids"}


def test_json_rpc_validates_and_returns_scores() -> None:
    model = MODULE.LocalCrossEncoder(tokenizer=FakeTokenizer(), session=FakeSession(), array_factory=lambda values: values)
    response = MODULE.handle_request({"id": 7, "method": "rerank", "params": {"query": "q", "texts": ["a", "b"]}}, model)
    assert response == {"jsonrpc": "2.0", "id": 7, "result": {"scores": [1.25, -0.75]}}, response
    invalid = MODULE.handle_request({"id": 8, "method": "rerank", "params": {"query": "q", "texts": "not-a-list"}}, model)
    assert invalid.get("error", {}).get("code") == -1, invalid
    unsupported = MODULE.handle_request({"id": 9, "method": "embed", "params": {}}, model)
    assert unsupported.get("error", {}).get("code") == -32601, unsupported


def test_empty_candidate_list_is_empty() -> None:
    model = MODULE.LocalCrossEncoder(tokenizer=FakeTokenizer(), session=FakeSession(), array_factory=lambda values: values)
    assert model.rerank("query", []) == []


if __name__ == "__main__":
    test_pair_encoding_and_numeric_scores()
    test_json_rpc_validates_and_returns_scores()
    test_empty_candidate_list_is_empty()
    print("local cross-encoder handshake: PASS (3 tests)")

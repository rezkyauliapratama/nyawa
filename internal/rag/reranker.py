#!/usr/bin/env python3
"""Offline JSONL RPC server for the pinned multilingual ONNX cross-encoder."""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

MODEL_NAME = "Horizon-Labs/multilingual-reranker-small"
DEFAULT_MODEL_DIR = Path("/opt/nyawa/model/reranker")
DEFAULT_MAX_LENGTH = 384
MAX_TEXT_CHARS = 16000
MAX_CANDIDATES = 50


class LocalCrossEncoder:
    def __init__(
        self,
        model_dir: Path | str | None = None,
        *,
        session: Any | None = None,
        tokenizer: Any | None = None,
        array_factory: Callable[[list[list[int]]], Any] | None = None,
    ) -> None:
        self.model_dir = Path(model_dir or os.environ.get("NYAWA_RERANKER_MODEL_DIR", DEFAULT_MODEL_DIR))
        self.max_length = int(os.environ.get("NYAWA_RERANKER_MAX_LENGTH", DEFAULT_MAX_LENGTH))
        if not 8 <= self.max_length <= 512:
            raise ValueError("NYAWA_RERANKER_MAX_LENGTH must be between 8 and 512")

        if tokenizer is None:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]

            tokenizer_path = self.model_dir / "tokenizer.json"
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"reranker tokenizer missing: {tokenizer_path}")
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        assert tokenizer is not None
        self.tokenizer = tokenizer
        self.tokenizer.enable_truncation(max_length=self.max_length)
        pad_token = "[PAD]"
        pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:
            pad_token = "<pad>"
            pad_id = self.tokenizer.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError("reranker tokenizer has no recognized pad token")
        self.tokenizer.enable_padding(direction="right", pad_id=pad_id, pad_token=pad_token)

        if session is None:
            import onnxruntime  # type: ignore[import-not-found]

            model_path = self.model_dir / "model_quantized.onnx"
            if not model_path.is_file():
                raise FileNotFoundError(f"reranker ONNX model missing: {model_path}")
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = onnxruntime.InferenceSession(str(model_path), options)
        assert session is not None
        self.session = session
        self.input_names = {item.name for item in self.session.get_inputs()}
        required = {"input_ids", "attention_mask"}
        if not required.issubset(self.input_names):
            raise ValueError(f"unsupported reranker ONNX inputs: {sorted(self.input_names)}")

        if array_factory is None:
            import numpy as np  # type: ignore[import-not-found]

            self.array_factory = lambda values: np.asarray(values, dtype=np.int64)
        else:
            self.array_factory = array_factory

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if not isinstance(query, str) or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("rerank requires a string query and a list of string passages")
        if len(texts) > MAX_CANDIDATES:
            raise ValueError(f"rerank candidate limit exceeded: {len(texts)} > {MAX_CANDIDATES}")
        if not texts:
            return []

        query = query[:MAX_TEXT_CHARS]
        pairs = [(query, text[:MAX_TEXT_CHARS]) for text in texts]
        encodings = self.tokenizer.encode_batch(pairs)
        model_inputs: dict[str, Any] = {
            "input_ids": self.array_factory([encoding.ids for encoding in encodings]),
            "attention_mask": self.array_factory([encoding.attention_mask for encoding in encodings]),
        }
        if "token_type_ids" in self.input_names:
            model_inputs["token_type_ids"] = self.array_factory([encoding.type_ids for encoding in encodings])
        if "position_ids" in self.input_names:
            model_inputs["position_ids"] = self.array_factory([
                list(range(len(encoding.ids))) for encoding in encodings
            ])
        if "global_attention_mask" in self.input_names:
            model_inputs["global_attention_mask"] = self.array_factory([
                [0] * len(encoding.ids) for encoding in encodings
            ])
        unknown = self.input_names - model_inputs.keys()
        if unknown:
            raise ValueError(f"unsupported reranker ONNX inputs: {sorted(unknown)}")

        output = self.session.run(None, model_inputs)[0]
        if hasattr(output, "reshape"):
            values = output.reshape(-1).tolist()
        else:
            values = [value for row in output for value in (row if isinstance(row, (list, tuple)) else [row])]
        if len(values) != len(texts):
            raise ValueError(f"reranker returned {len(values)} scores for {len(texts)} passages")
        scores = [float(value) for value in values]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("reranker returned a non-finite score")
        return scores


def handle_request(request: dict[str, Any], model: LocalCrossEncoder) -> dict[str, Any]:
    request_id = request.get("id")
    if request.get("method") != "rerank":
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "unknown method"}}
    params = request.get("params") or {}
    try:
        query = params.get("query")
        texts = params.get("texts")
        if not isinstance(query, str) or not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("rerank requires a string query and a list of string passages")
        scores = model.rerank(query, texts)
        return {"jsonrpc": "2.0", "id": request_id, "result": {"scores": scores}}
    except Exception as exc:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -1, "message": str(exc)}}


def self_test() -> int:
    model = LocalCrossEncoder()
    scores = model.rerank(
        "What is configured for the fictional Cedar Registry?",
        ["The Cedar Registry uses a four-entry rolling window.", "The ceramic bowl is blue."],
    )
    if len(scores) != 2 or not all(math.isfinite(score) for score in scores):
        raise RuntimeError("local cross-encoder smoke test returned invalid scores")
    print("Local cross-encoder inference smoke test: PASS")
    return 0


def main() -> int:
    try:
        model = LocalCrossEncoder()
    except Exception as exc:
        print(f"ERROR reranker unavailable: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"READY model={MODEL_NAME}", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = handle_request(request, model)
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    raise SystemExit(main())

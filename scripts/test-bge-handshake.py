#!/usr/bin/env python3
"""Exercise bge_server.py's real JSONL pipes with local fake model modules."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "internal/embedder/bge_server.py"

FAKES = {
    "numpy.py": '''
class _Array:
    def __init__(self, values=None):
        self.values = [0.1] * 384 if values is None else values
    def astype(self, _dtype): return self
    def __mul__(self, _other): return self
    def __truediv__(self, _other): return self
    def sum(self, axis=None): return self
    def clip(self, min=None): return self
    def __getitem__(self, _index): return self
    def tolist(self): return list(self.values)

int64 = object()
float32 = object()

def array(values, dtype=None): return _Array()
def zeros_like(values): return _Array()
def expand_dims(values, axis=-1): return _Array()

class _Linalg:
    @staticmethod
    def norm(values): return 1.0
linalg = _Linalg()
''',
    "onnxruntime.py": '''
class SessionOptions:
    def __init__(self):
        self.intra_op_num_threads = 0
        self.graph_optimization_level = None

class GraphOptimizationLevel:
    ORT_ENABLE_ALL = object()

class _Output:
    shape = [1, 2, 384]

class _Input:
    name = "input_ids"

class InferenceSession:
    def __init__(self, model_path, options): pass
    def get_inputs(self): return [_Input()]
    def get_outputs(self): return [_Output()]
    def run(self, _outputs, _inputs):
        from numpy import array
        return [array(None)]
''',
    "tokenizers.py": '''
class _Encoded:
    ids = [101, 102]
    attention_mask = [1, 1]

class Tokenizer:
    @classmethod
    def from_file(cls, path): return cls()
    def enable_truncation(self, max_length): self.max_length = max_length
    def encode(self, text): return _Encoded()
''',
}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="nyawa-bge-handshake-") as temp:
        root = Path(temp)
        fake_python = root / "fake_modules"
        fake_python.mkdir()
        for name, content in FAKES.items():
            (fake_python / name).write_text(content)
        model = root / "model"
        model.mkdir()
        (model / "model.onnx").write_bytes(b"test-model")
        (model / "tokenizer.json").write_text("{}")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(fake_python)
        env["NYAWA_MODEL_DIR"] = str(model)
        proc = subprocess.Popen(
            [sys.executable, str(SERVER)], cwd=ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None
            stdin = proc.stdin
            stdout = proc.stdout
            stderr = proc.stderr
            loaded = stderr.readline().strip()
            ready = stderr.readline().strip()
            assert loaded == "Model loaded: dim=384", loaded
            assert ready == "READY dim=384", ready
            stdin.write(json.dumps({
                "jsonrpc": "2.0", "id": 7, "method": "embed",
                "params": {"text": "handshake"},
            }) + "\n")
            stdin.flush()
            response = json.loads(stdout.readline())
            assert response == {
                "jsonrpc": "2.0", "id": 7,
                "result": {"embedding": [0.1] * 384, "dim": 384},
            }
            print("BGE stdin/stdout JSONL handshake: PASS (id=7, dim=384)")
        finally:
            proc.kill()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()

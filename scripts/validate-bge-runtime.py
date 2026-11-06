#!/usr/bin/env python3
"""No-container validation for the durable BGE runtime contract."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "sha256:6a2830f831b8aa8bcc6c82ed07b5ccbfd82882aa98c0bfb176e7a97a498b1ec0"
MODEL_SHA256 = "66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc"
TOKENIZER_SHA256 = "b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dockerfile = (ROOT / "runtime/Dockerfile").read_text()
    requirements = (ROOT / "runtime/requirements.lock").read_text()
    server = (ROOT / "internal/embedder/bge_server.py").read_text()
    embedder = (ROOT / "internal/embedder/py_embedder.go").read_text()

    assert DIGEST in dockerfile
    assert "io.nyawa.provenance=\"source-build\"" in dockerfile
    assert f"io.nyawa.approved-baseline-digest=\"{DIGEST}\"" in dockerfile
    assert "approved-nyawa-v124" not in dockerfile
    assert "COPY runtime/model /opt/nyawa/model" in dockerfile
    assert "COPY internal/embedder/bge_server.py" in dockerfile
    assert "NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--no-deps" in dockerfile
    assert "REPLACE_WITH_PUBLISHED_HASH" not in requirements
    packages = [
        match.group(1)
        for line in requirements.splitlines()
        if (match := re.match(r"^([A-Za-z0-9_.-]+)==", line))
    ]
    assert packages == ["numpy", "onnxruntime", "tokenizers"]
    for package in packages:
        assert f"{package}==" in requirements

    handshake = (ROOT / "scripts/test-bge-handshake.py").read_text()
    assert "subprocess.Popen" in handshake
    assert "stdin" in handshake and "stdout" in handshake
    assert '"method": "embed"' in handshake
    assert "result" in handshake and '"dim": 384' in handshake

    model_path = ROOT / "runtime/model/model.onnx"
    if model_path.exists():
        assert sha256(model_path) == MODEL_SHA256
    else:
        print("model.onnx absent in source checkout; publish workflow must fetch and verify it")
    assert sha256(ROOT / "runtime/model/tokenizer.json") == TOKENIZER_SHA256

    for forbidden in ("socket", "AF_UNIX", "threading", "serve_mode", "NYAWA_EMBED_PYTHON"):
        assert forbidden not in server, f"alternate transport remains: {forbidden}"
    for forbidden in ("net.Dial", "socketPathFor", "NYAWA_EMBED_PYTHON", "/opt/data/bge-venv", "/opt/hermes/.venv"):
        assert forbidden not in embedder, f"host dependency remains: {forbidden}"
    assert server.count("for line in sys.stdin") == 1
    assert embedder.count("exec.Command(pythonPath, scriptPath)") == 1
    assert "READY dim=" in server
    assert "exec.Command(pythonPath, scriptPath)" in embedder
    print("BGE runtime static contract: PASS")


if __name__ == "__main__":
    main()

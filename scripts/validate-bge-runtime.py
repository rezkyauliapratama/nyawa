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
RERANKER_MODEL_SHA256 = "f913154682468bfe92d6a7e02ed4e994e548c112fddc74452b573b17c949046e"
RERANKER_TOKENIZER_SHA256 = "7925f638b5444897efe3b4608eed56c1652f0ed39c2662c8302de1bc264c6088"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dockerfile = (ROOT / "runtime/Dockerfile").read_text()
    workflow = (ROOT / ".github/workflows/bge-runtime-publish.yml").read_text()
    requirements = (ROOT / "runtime/requirements.lock").read_text()
    server = (ROOT / "internal/embedder/bge_server.py").read_text()
    embedder = (ROOT / "internal/embedder/py_embedder.go").read_text()

    assert DIGEST in dockerfile
    assert "io.nyawa.provenance=\"source-build\"" in dockerfile
    assert f"io.nyawa.approved-baseline-digest=\"{DIGEST}\"" in dockerfile
    assert "approved-nyawa-v124" not in dockerfile
    assert "ARG RELEASE_VERSION=dev" in dockerfile
    assert 'org.opencontainers.image.version="${RELEASE_VERSION}"' in dockerfile
    assert '$(/usr/local/bin/nyawa version)" = "nyawa v${RELEASE_VERSION}' in dockerfile
    assert "name: Normalize image tag to runtime semver" in workflow
    assert 'release_version="${IMAGE_TAG#bge-v}"' in workflow
    assert "-r[0-9]+" in workflow
    assert "org.opencontainers.image.version=${{ steps.version.outputs.release_version }}" in workflow
    assert "RELEASE_VERSION=${{ steps.version.outputs.release_version }}" in workflow
    assert "RELEASE_VERSION=${{ env.IMAGE_TAG }}" not in workflow
    assert "COPY runtime/model /opt/nyawa/model" in dockerfile
    assert "COPY internal/embedder/bge_server.py" in dockerfile
    assert "COPY internal/rag/reranker.py" in dockerfile
    assert "INCLUDE_RERANKER" in dockerfile
    assert "reranker.py --self-test" in dockerfile
    assert "NYAWA_RERANKER_MODEL_DIR=/opt/nyawa/model/reranker" in dockerfile
    assert "include_reranker" in workflow
    assert RERANKER_MODEL_SHA256 in workflow and RERANKER_TOKENIZER_SHA256 in workflow
    assert "23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398" in workflow
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
    reranker_model = ROOT / "runtime/model/reranker/model_quantized.onnx"
    reranker_tokenizer = ROOT / "runtime/model/reranker/tokenizer.json"
    if reranker_model.exists() or reranker_tokenizer.exists():
        assert reranker_model.is_file() and reranker_tokenizer.is_file()
        assert sha256(reranker_model) == RERANKER_MODEL_SHA256
        assert sha256(reranker_tokenizer) == RERANKER_TOKENIZER_SHA256
    reranker = (ROOT / "internal/rag/reranker.py").read_text()
    assert "onnxruntime" in reranker and "tokenizers" in reranker
    for forbidden in ("urllib", "requests", "httpx", "from_pretrained", "huggingface.co"):
        assert forbidden not in reranker.lower(), f"runtime cross-encoder attempts network access: {forbidden}"
    assert "NYAWA_ENABLE_CROSS_ENCODER" in (ROOT / "internal/rag/reranker.go").read_text()
    handshake = (ROOT / "scripts/test-reranker-handshake.py").read_text()
    assert "FakeSession" in handshake and "jsonrpc" in handshake

    for forbidden in ("socket", "AF_UNIX", "threading", "serve_mode", "NYAWA_EMBED_PYTHON"):
        assert forbidden not in server, f"alternate transport remains: {forbidden}"
    for forbidden in ("net.Dial", "socketPathFor", "/opt/data/bge-venv", "/opt/hermes/.venv"):
        assert forbidden not in embedder, f"host dependency remains: {forbidden}"
    assert 'lookup("NYAWA_EMBED_PYTHON")' in embedder
    assert "filepath.IsAbs(path)" in embedder
    assert 'exec.Command(path, "-c", "import onnxruntime, numpy; print(\'ok\')")' in embedder
    assert "NYAWA_EMBED_PYTHON must be an absolute executable path" in embedder
    assert "NYAWA_EMBED_PYTHON failed the onnxruntime+numpy import probe" in embedder
    assert server.count("for line in sys.stdin") == 1
    assert embedder.count("exec.Command(pythonPath, scriptPath)") == 1
    assert "READY dim=" in server
    assert "exec.Command(pythonPath, scriptPath)" in embedder
    print("BGE runtime static contract: PASS")


if __name__ == "__main__":
    main()

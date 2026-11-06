# syntax=docker/dockerfile:1.7
# Durable BGE runtime. It compiles the reviewed Nyawa source and ships the
# interpreter, ONNX Runtime, tokenizer, and model in the same image.
#
# Provenance contract: this is a source build from this checkout. The approved
# Nyawa v1.2.4 digest is recorded as a baseline reference, not used as a stage
# or as the parent image of the patched binary.

FROM golang:1.23-bookworm AS nyawa-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
# Copy only the Go build sources; runtime/model stays out of the builder stage.
COPY cmd/nyawa ./cmd/nyawa
COPY internal ./internal
ARG RELEASE_VERSION=phase-c
RUN CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build \
    -tags sqlite_fts5 \
    -ldflags="-s -w -X github.com/rezkyauliapratama/nyawa/internal/version.Version=${RELEASE_VERSION}" \
    -o /nyawa ./cmd/nyawa/

FROM python:3.13-slim-bookworm
LABEL io.nyawa.provenance="source-build" \
      io.nyawa.approved-baseline-digest="sha256:6a2830f831b8aa8bcc6c82ed07b5ccbfd82882aa98c0bfb176e7a97a498b1ec0"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NYAWA_EMBED_MODEL_DIR=/opt/nyawa/model \
    NYAWA_MODEL_DIR=/opt/nyawa/model
WORKDIR /opt/nyawa
COPY runtime/requirements.lock /opt/nyawa/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --no-deps -r /opt/nyawa/requirements.lock
COPY --from=nyawa-builder /nyawa /usr/local/bin/nyawa
COPY internal/embedder/bge_server.py /opt/nyawa/internal/embedder/bge_server.py
COPY runtime/model /opt/nyawa/model
RUN test -s /usr/local/bin/nyawa \
    && test -s /opt/nyawa/internal/embedder/bge_server.py \
    && test -s /opt/nyawa/model/model.onnx \
    && test -s /opt/nyawa/model/tokenizer.json

ENTRYPOINT ["nyawa"]
CMD ["serve", "/opt/data/memory.db"]

# Deterministic Residual Packs for Compaction Checkpoints

**Status:** Proposal only. This document does not change Nyawa's runtime, schema, HTTP API, MCP API, or compaction output.

**Scope:** A bounded, inspectable sidecar for information omitted at a compaction boundary. It is additive to [context-compaction-design.md](context-compaction-design.md); when that historical design document and the checked-in Go code disagree about current behavior, the code is the source of truth.

## 1. Why a residual pack

[REMORY](https://arxiv.org/abs/2610.11287) pairs a readable summary with bounded learned soft-memory tokens. Nyawa cannot and should not claim to implement those learned, model-specific tensors with this proposal. Instead, a **residual pack** is a deterministic, versioned, textual/structured sidecar attached to a compaction checkpoint:

```text
source transcript
  -> existing prune + summary + retained tail
  -> summary checkpoint
  -> deterministic residual pack of selected omitted details
```

The summary remains the readable working-state checkpoint. The pack preserves a small, provenance-bearing set of details that deterministic pruning and summarization can omit, especially explicit tool failures and selected tool results. Exact raw history remains a separate archival concern; a residual pack is not an archive and must not silently become one.

## 2. Verified current baseline

| Surface | Implemented behavior today | Consequence for this proposal |
|---|---|---|
| HTTP compaction | `POST /v1/compact` is registered in `internal/server/server.go` and handled by `internal/server/compact_handler.go`. It accepts `messages`, optional focus/segmentation/pruning/session fields, and returns `summary_block`, `memory_ids`, `namespace`, and `stats`. | No checkpoint or residual-pack field exists in either request or response. |
| MCP compaction | `compact_context` is declared and handled in `internal/mcp/server.go`. Its `messages` argument is a JSON-array **string**, unlike the HTTP array value. | A future MCP extension must preserve this distinction; HTTP fields cannot be copied verbatim into the MCP schema. |
| Internal message model | `compact.Message` has only `Role` and `Content`. `FromOpenAI` copies only string `role` and string `content`. | Tool name, tool-call ID, arguments, exit code, structured content, and tool status are discarded before compaction. A reliable residual selector cannot recover them afterward. |
| Pruning | `PruneToolResults` replaces oversized `tool` role content with an ordinal placeholder such as `[tool result stored in nyawa:0]`. It returns only a count. | The current placeholder is not backed by separately persisted tool-result content or a durable identifier, and the pruned source indexes are unavailable to a pack builder. |
| Summary generation | `SummarizeSegments` uses an LLM when `NYAWA_LLM_API_KEY` is set and otherwise concatenates segment messages deterministically. | A residual pack can be deterministic for fixed normalized inputs, but it cannot make an upstream LLM summary deterministic. |
| Summary persistence | A successful non-dry-run compaction stores one `summary_<UnixNano>` memory with `Type: types.TypeContext`; it uses the requested namespace or `context`. | Current code does not store `mem_type=summary`, archived segments, residual entries, or a summary-to-archive relationship. Runtime-generated summary IDs must not participate in a deterministic pack identity. |
| Compaction log | `compaction_log` currently has only `session_key`, `old_session_id`, `new_session_id`, `summary_id`, and `created_at`; it has no primary key, foreign key, checkpoint ID, or query API. | A new checkpoint identity needs additive tables rather than treating an existing log row as a durable primary key. |
| No-op and dry-run | A short transcript returns the original messages with `noop: true`; `dry_run` avoids the summary memory and log write. | Neither case creates a persisted checkpoint today. A proposed pack preview must not be described as persisted. |
| Recall | `POST /v1/recall` takes `query`, `namespace`, `limit`, `min_score`, and `exclude_types`; results include memory IDs. Active superseded memories are excluded by the pipeline. | Existing recall can measure retained summary-memory retrieval, but it cannot retrieve a proposed dedicated residual table until an explicit integration is implemented. |

Two implementation details matter for compatibility work later:

- The current HTTP and MCP handlers apply `PruneToolResults` before deriving the retained tail. Therefore a large tool result in the tail may still be replaced; the current code does not provide a per-message prune manifest.
- `segment_overlap` is documented near the handlers as defaulting to `5`, but an omitted zero-valued request field currently reaches `Segment` as `0` because the handlers only substitute the default when the value is negative. This proposal does not alter that behavior.

## 3. Goals and non-goals

### Goals

1. **Deterministic output:** given the same normalized checkpoint input, policy, and summary bytes, generate byte-identical pack JSON and digest.
2. **Bounded context supplement:** retain a small amount of high-signal omitted information without copying a transcript into every checkpoint.
3. **Traceable provenance:** every item names its source message index and hashes the normalized original content.
4. **Explicit tool semantics:** classify failures and tool outputs from supplied metadata, not a fragile search for words such as `error` in text.
5. **Additive migration:** leave `memories` and the existing `compaction_log` readable during rollout.
6. **Evaluator-ready evidence:** emit stable IDs and digests needed by the compaction-aware evaluator in [compaction-evaluator-spec.md](compaction-evaluator-spec.md).

### Non-goals

- No learned soft tokens, model hidden states, training loop, or claim of REMORY-equivalence.
- No automatic prompt injection or agent behavior change.
- No replacement for a full raw-transcript/archive retention design.
- No heuristic semantic extraction from arbitrary message text in the first version.
- No migration, endpoint, MCP property, memory-type change, or production implementation in this slice.

## 4. Terminology

| Term | Definition |
|---|---|
| **checkpoint** | A durable record tying one compaction's source identity, summary identity, policy, and optional parent checkpoint together. It is distinct from the current unkeyed `compaction_log` row. |
| **residual pack** | A canonical JSON document of selected omitted details plus provenance. It is attached one-to-one to a checkpoint. |
| **source transcript** | The normalized message sequence before compaction mutation, plus future explicit tool-event and retention-hint metadata. |
| **omitted source** | A source message summarized away or explicitly pruned from the compacted output. Retained-tail content is not duplicated unless it was separately pruned. |
| **policy** | Versioned deterministic selection, redaction, truncation, and budget rules. Its full resolved configuration is fingerprinted. |
| **legacy metadata coverage** | `none`, `partial`, or `complete`, indicating whether the future caller supplied tool metadata for the source messages. It prevents an empty pack from being misread as proof that no useful detail existed. |

## 5. Proposed input contract and identity

### 5.1 Preserve structured tool facts before `compact.Message`

The current `compact.Message` representation cannot identify a tool call or determine whether its output is an error. A future compaction request must therefore add a named, versioned metadata field rather than relying on unknown keys inside `messages`.

For HTTP, the proposed field is `tool_events`. For MCP, it is a JSON-array string named `tool_events`, consistent with the existing string-shaped `messages` argument. Both are **proposal-only** fields.

Each event is tied to one source message:

```json
{
  "event_index": 42,
  "source_message_index": 17,
  "tool_name": "terminal",
  "tool_call_id": "optional-provider-id",
  "tool_input_sha256": "sha256:<lowercase-hex>",
  "outcome": "failure",
  "error_class": "nonzero_exit",
  "is_polling": false,
  "retain": true
}
```

Rules for version 1:

- `event_index` and `source_message_index` are non-negative integers and are strictly unique within their respective sequences.
- `source_message_index` must reference a source message whose normalized role is `tool`; otherwise validation fails.
- `tool_name` is required when metadata coverage is `complete`.
- `outcome` is one of `success`, `failure`, `tool_error`, `timeout`, `cancelled`, or `unknown`.
- `error_class` is optional and is descriptive only; error selection follows `outcome`, not textual parsing.
- `tool_input_sha256` is optional when the provider cannot expose inputs. Missing data stays explicit as `null`; it is never invented from an output.
- `retain` and future non-tool `residual_hints` are caller assertions. V1 accepts only explicit assertions; it does not infer decisions, constraints, or pending work with an LLM or keyword heuristic.

A legacy caller without this field remains compatible, but its pack must declare `metadata_coverage: "none"`. V1 must not infer a tool error by scanning its `Content` string.

### 5.2 Normalization

Before hashing or selection, the pack builder constructs a typed source envelope. It uses arrays and fixed structs, never maps, so field ordering is not data-dependent.

1. Convert strings to valid UTF-8, replacing an invalid byte sequence with one Unicode replacement character.
2. Replace `\r\n` and bare `\r` with `\n`. Do not trim whitespace, normalize Unicode, redact, reorder messages, or lower-case content.
3. Preserve source-message array order exactly. Tool events are ordered by `event_index`; retention hints are ordered by `source_message_index`, then declared hint ordinal.
4. Represent unavailable optional values as JSON `null`, not omitted fields, in the canonical form.
5. Encode the typed struct with Go `encoding/json.Marshal`, without indentation. The implementation must not use a map for a canonical object.

The redaction policy is part of the future resolved policy. If an implementation introduces deterministic redaction before persistence, it must record the redaction-policy version and hash in `policy_sha256`; it must not claim that the current compaction path applies such redaction.

### 5.3 Hashes and checkpoint ID

All digest values use SHA-256 over the exact canonical UTF-8 bytes and are rendered as lowercase hexadecimal with a `sha256:` prefix.

```text
source_sha256  = SHA-256(canonical normalized source envelope)
summary_sha256 = SHA-256(normalized summary text bytes)
policy_sha256  = SHA-256(canonical resolved residual-pack policy)
checkpoint_id  = "ccp_v1_" + hex(SHA-256(canonical checkpoint identity))
```

The checkpoint identity contains exactly:

```text
schema_version, session_key, old_session_id, new_session_id,
parent_checkpoint_id (or null), source_sha256, summary_sha256, policy_sha256
```

It excludes database timestamps, `summary_<UnixNano>` IDs, SQL rowids, and runtime-generated UUIDs. Replaying exactly the same normalized input and summary therefore produces the same checkpoint ID. A different LLM-generated summary produces a different `summary_sha256` and checkpoint ID; deterministic pack construction is conditional on a fixed summary, not a promise that an LLM is deterministic.

## 6. Proposed residual-pack wire format

`pack_sha256` is the SHA-256 of the canonical JSON bytes and is stored outside the JSON document to avoid a self-referential hash field.

```json
{
  "schema_version": 1,
  "checkpoint_id": "ccp_v1_<digest>",
  "metadata_coverage": "complete",
  "policy": {
    "name": "nyawa.residual-pack",
    "version": 1,
    "max_items": 16,
    "max_excerpt_bytes": 768,
    "max_pack_bytes": 16384
  },
  "source": {
    "source_sha256": "sha256:<digest>",
    "summary_sha256": "sha256:<digest>",
    "compacted_through_message_index": 29,
    "parent_checkpoint_id": null
  },
  "items": [
    {
      "entry_id": "rpi_v1_<digest>",
      "ordinal": 0,
      "source_message_index": 17,
      "kind": "tool_error",
      "tool_name": "terminal",
      "tool_call_id": null,
      "tool_input_sha256": "sha256:<digest>",
      "outcome": "failure",
      "content_sha256": "sha256:<digest>",
      "excerpt": "command failed with a non-zero exit status",
      "excerpt_truncated": false,
      "selection_reason": "explicit_tool_outcome"
    }
  ]
}
```

Field rules:

- `kind` is one of `tool_error`, `tool_result`, `decision`, `constraint`, `pending_action`, or `carry_forward`. V1 emits `decision`, `constraint`, and `pending_action` only from explicit caller hints.
- `entry_id` is `rpi_v1_` plus the SHA-256 of the canonical item fields excluding `ordinal`. It allows an evaluator to identify the same evidence even if an outer pack is rebuilt with a larger budget.
- `content_sha256` identifies the normalized pre-truncation source content. `excerpt` is a bounded, optionally redacted presentation field, not a replacement for the original content.
- `ordinal` is the zero-based order in the final pack after final source-order sorting.
- The pack has no `created_at`; storage timestamps are metadata, not pack content.
- An empty `items` array is valid. Its meaning must be interpreted with `metadata_coverage` and the source/selection audit, not as proof of absence of useful history.

The numeric values in `policy` are proposed defaults, not current Nyawa defaults.

## 7. Deterministic selection and bounding

### 7.1 Candidate eligibility

A future compactor must expose a prune/summary manifest containing source indexes that were omitted from the output. The current `PruneToolResults` return value is insufficient because it contains only a count.

V1 creates a candidate only when all of the following are true:

1. It has an omitted source index, or the manifest says it was pruned even though it was otherwise in the retained tail.
2. Its metadata is explicit (`tool_events` or an explicit residual hint).
3. It survives the configured deterministic redaction gate, if one is enabled.

This prevents the pack from duplicating retained context and prevents implicit classification of unstructured text.

### 7.2 Candidate kinds and admission priority

Candidates are deduplicated and admitted in this order:

1. `tool_error` for `failure`, `tool_error`, or `timeout` outcomes; failures are never merged away.
2. Explicit `constraint`, `decision`, or `pending_action` hints marked `retain: true`.
3. The latest distinct successful `tool_result` for a `(tool_name, tool_input_sha256, content_sha256)` tuple.
4. Explicit non-retained hints and `carry_forward` items, if a later policy adds parent-pack carry-forward.

For admission, sort by this stable key:

```text
priority ascending,
source_message_index descending,
event_index descending when present,
kind lexicographically,
content_sha256 lexicographically
```

The descending source index deliberately prefers the most recent usable state within a priority class. After admission, sort selected items by `source_message_index` ascending, then `event_index` ascending, then `entry_id` ascending; assign `ordinal` at that point. Thus selection preference is deterministic without presenting items in reverse causal order.

### 7.3 Budget rules

The builder applies the following hard limits in order:

1. Normalize and redact an eligible source excerpt.
2. Cut an excerpt at a valid UTF-8 rune boundary to `max_excerpt_bytes`; set `excerpt_truncated` when bytes were removed.
3. Admit at most `max_items` candidates by the stable admission order.
4. Serialize the whole typed pack canonically. If it exceeds `max_pack_bytes`, remove the lowest-priority last-admitted candidate and serialize again until it fits.
5. If required fixed metadata alone exceeds the budget, return a validation error rather than silently emitting an invalid or oversized pack.

The final pack is bounded by its canonical serialized byte length, not an approximate token estimate. The selection policy and all three resolved budget values are part of `policy_sha256`.

## 8. Proposed durable schema and API delta

### 8.1 Additive tables

The initial implementation should leave `compaction_log` unchanged and add the following tables through an additive migration:

```sql
CREATE TABLE compaction_checkpoints (
  checkpoint_id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  session_key TEXT NOT NULL,
  old_session_id TEXT NOT NULL,
  new_session_id TEXT NOT NULL,
  summary_id TEXT NOT NULL,
  parent_checkpoint_id TEXT,
  source_sha256 TEXT NOT NULL,
  summary_sha256 TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (summary_id) REFERENCES memories(id),
  FOREIGN KEY (parent_checkpoint_id) REFERENCES compaction_checkpoints(checkpoint_id),
  UNIQUE (session_key, old_session_id, new_session_id,
          source_sha256, summary_sha256, policy_sha256)
);
CREATE INDEX idx_compaction_checkpoints_session_created
  ON compaction_checkpoints(session_key, created_at);

CREATE TABLE compaction_residual_packs (
  checkpoint_id TEXT PRIMARY KEY,
  pack_version INTEGER NOT NULL,
  canonical_json BLOB NOT NULL,
  pack_sha256 TEXT NOT NULL UNIQUE,
  item_count INTEGER NOT NULL,
  byte_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (checkpoint_id) REFERENCES compaction_checkpoints(checkpoint_id)
);
```

SQLite foreign-key enforcement is connection-local. The current store connection setup does not enable it explicitly, so an implementation must either enable `PRAGMA foreign_keys = ON` for every relevant connection or validate both relationships in the same write transaction before commit. It must not assume the declarations alone enforce integrity.

`canonical_json` is a `BLOB` so its exact hashed byte sequence is retained. It is not inserted into `memories` by default; otherwise the existing recall pipeline may surface opaque JSON as ordinary context. A later explicit projection/search design can make selected entries retrievable without changing this invariant.

### 8.2 Transactional write sequence

For a non-noop, non-dry-run request, a future implementation should:

1. Build and validate the summary, omit/prune manifest, checkpoint identity, and pack before opening the write transaction.
2. Insert the summary memory and verify the exact `summary_id` exists.
3. Insert `compaction_checkpoints` and `compaction_residual_packs` in one transaction.
4. Insert the legacy `compaction_log` compatibility row in the same transaction when its session-key condition is met.
5. Commit before returning a checkpoint ID.

If any write fails, roll back all of these writes. This differs intentionally from the current best-effort path, which ignores errors from `LogCompaction` and can log a generated summary ID even when insertion did not succeed.

`dry_run` may return a proposed pack preview and its hashes, but must persist neither table nor a legacy log row. A `noop` response must have no checkpoint by default because no source was omitted; an explicit future `checkpoint_noop` policy would need a separate design.

### 8.3 Compatible response addition

A future HTTP response can add, without removing current fields:

```json
{
  "checkpoint": {
    "id": "ccp_v1_<digest>",
    "summary_id": "summary_<runtime-id>",
    "residual_pack_sha256": "sha256:<digest>",
    "residual_item_count": 3
  }
}
```

The future MCP result may expose the same object. It must retain existing `summary_block`, `memory_ids`, `namespace`, and `stats` shapes. New request fields and a read endpoint such as `GET /v1/compaction/checkpoints/:id` are proposed only; neither exists now.

## 9. Retrieval and active-context behavior

A residual pack is attached data, not automatically active prompt content. No current Nyawa component reads one, and the first implementation must preserve that behavior until a deliberate rehydration policy is reviewed.

A later rehydration API should return only the bounded pack associated with an explicit checkpoint. It should not scan arbitrary packs by text. Searchable residual recall, if desired, requires a separate projection with clear namespace, `TypeContext` semantics, provenance, deletion, and supersession rules. The current `POST /v1/recall` can still be used to retrieve the existing summary memory, but cannot retrieve a dedicated pack table by itself.

Because current compaction does not archive raw tool payloads, `content_sha256` proves provenance but does not make an omitted payload recoverable. A future archival feature must be separately authorized, retention-controlled, and evaluated; it is not implied by this proposal.

## 10. Acceptance criteria for a future implementation

| Area | Required proof |
|---|---|
| Determinism | Two runs over byte-identical normalized source, fixed summary, and fixed policy produce identical `checkpoint_id`, canonical JSON, item order, and hashes. |
| Sensitivity | Changing source content, summary bytes, source-session identity, or resolved policy changes the appropriate digest and checkpoint ID. |
| Bounds | Every emitted pack satisfies both item and serialized-byte limits; excerpts never end inside an invalid UTF-8 sequence. |
| Provenance | Every item points to an omitted source index and has a hash matching the normalized pre-truncation source content. |
| Metadata honesty | A legacy request emits `metadata_coverage: "none"` and creates no invented tool name, call ID, outcome, or error label. |
| Persistence | Failed summary/checkpoint/pack/log writes leave no partial checkpoint relationship. |
| Compatibility | Existing `/v1/compact` and `compact_context` clients work unchanged when the new fields are absent. |
| Prompt safety | A pack is never injected into `summary_block` or a later prompt unless a separately tested rehydration mode explicitly requests it. |

## 11. Implementation status matrix

| Item | Status |
|---|---|
| Tier-1 oversized `tool` role pruning | Implemented, but only with an ordinal placeholder and count. |
| Segment summary, retained tail, summary block, HTTP/MCP compaction | Implemented. |
| Summary memory and legacy `compaction_log` write | Implemented, best effort. |
| Durable checkpoint identity | Proposal only. |
| Canonical residual pack builder | Proposal only. |
| Structured tool-event preservation | Proposal only. |
| Additive checkpoint/pack schema | Proposal only. |
| Pack read/rehydration/search integration | Proposal only. |
| Evaluator integration | Proposal only; see [compaction-evaluator-spec.md](compaction-evaluator-spec.md). |

## 12. Reference

- Hanchen Xia et al., *REMORY: Learning Residual Memory for Context Compaction*, arXiv:2610.11287. REMORY's learned, bounded soft-memory supplement motivates the summary-plus-residual shape here; this design intentionally uses inspectable deterministic data instead of model-space memory.

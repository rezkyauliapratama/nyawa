# Compaction-Aware Evaluator Specification

**Status:** The pure offline scorer described here is implemented in `internal/compactioneval`, and a disabled-by-default passive Hermes event collector is implemented at `config/hermes/plugins/nyawa-compaction-trace`. The collector emits fail-open JSONL event records for the P0 baseline. Trace assembly, CLI scoring, residual-pack reader, metric endpoint, and runtime behavior remain proposal-only.

**Purpose:** Define reproducible metrics for whether a compaction strategy preserves useful state and reduces avoidable agent work. The target signals are inspired by REMORY's reported repeated-tool-output and tool-error counts, plus Nyawa-specific recall-after-compaction checks.

This specification complements [residual-pack-proposal.md](residual-pack-proposal.md). It deliberately separates a metric definition from a claim that Nyawa currently collects the data needed to calculate it.

## 1. Current capability boundary

| Capability | Current state | Evaluator consequence |
|---|---|---|
| `POST /v1/compact` | Implemented; returns compacted messages, summary-memory IDs, namespace, and `CompactionStats`. | It does not return a checkpoint ID, per-message prune manifest, tool event identity, tool status, or residual-pack digest. |
| `compact_context` MCP tool | Implemented; accepts the messages array encoded as a string. | It emits no tool telemetry beyond its compaction response. |
| `compact.CompactionStats` | Contains input/output message and token estimates, segment count, and pruned-tool-result count. | It is not a repeated-output, error, or recall-retention metric. |
| `compaction_log` | Implemented with session IDs, summary ID, and created timestamp only. | It cannot uniquely identify a checkpoint or join a later tool event to one. |
| `POST /v1/recall` | Implemented; request fields are `query`, `namespace`, `limit`, `min_score`, `exclude_types`; response results include `id`, `rank`, and score fields. | It is the current-compatible endpoint for ID-based recall probes. |
| Existing benchmark harness | `scripts/recall-bench/` measures general recall@1/@3/@5 and MRR by running the CLI against a read-only database. | It has no compaction trace, no tool telemetry, and maps CLI output back to IDs by content. It is useful precedent, not this evaluator. |
| Residual pack | Proposal only. | Direct pack lookup and residual-aware recall cannot be reported as implemented behavior. |

A pure scorer is implemented at `internal/compactioneval`. It validates and scores an externally captured trace now, but Nyawa cannot autonomously produce a complete trace with the required fields yet.

## 2. Evaluation questions

For each task run and each completed compaction boundary, answer three independent questions:

1. **Repeated tool outputs:** Did the agent receive identical outputs from the same tool more than necessary across the task, with polling and planning excluded from this metric?
2. **Tool errors:** How many tool attempts failed by an explicit tool error, timeout, or non-zero command outcome?
3. **Recall after compaction:** Can a post-compaction query retrieve pre-compaction target memories that were intentionally persisted and remain active?

The first two are trajectory metrics. The third is a retrieval metric. They must not be combined into one score: fewer calls can lower repeat counts while also reducing opportunities to observe failures, and high recall does not prove that an agent used the retrieved memory correctly.

## 3. Experimental unit and fair comparison

### 3.1 Unit of analysis

The primary unit is one **task run**, identified by `(suite_id, task_id, run_id, condition)`. A run may contain zero or more completed compactions. Tool-event counts are calculated over the full task run, not reset at a compaction boundary; post-compaction breakdowns are attributed to the most recent completed boundary.

A compaction is completed only when a durable checkpoint exists. For the current implementation, a successful non-dry-run response with a stored summary and the corresponding log row may be recorded as a legacy boundary by an external harness, but it is not a canonical checkpoint because the current log has no ID or atomicity guarantee. `noop: true` and `dry_run: true` are not completed compactions.

### 3.2 Conditions

Use explicit condition labels:

- `no_compaction` — no compaction is applied.
- `summary_only` — the existing summary block only.
- `summary_plus_residual_pack` — proposed future residual-pack rehydration is enabled.
- `retrieval_only` — a distinct retrieval intervention, if tested.

A comparison between `summary_only` and `summary_plus_residual_pack` must hold fixed the actor/model revision, system prompt, task input, tool implementation, tool permissions, compaction trigger, token budget, maximum turns, decoding settings, and evaluation environment. The summary bytes should be frozen and shared between paired conditions when the question is the effect of the pack rather than the effect of a different summary.

Current `SummarizeSegments` can call an LLM at temperature `0.3` when `NYAWA_LLM_API_KEY` is present. A reproducible evaluator must either run with that key absent to use the deterministic fallback, or capture and replay the exact summary text. It must record which choice was made.

### 3.3 Reporting paired evidence

For repeated runs of the same task, report task-level paired deltas in addition to aggregate counts. A future harness may compute a seeded paired bootstrap confidence interval over task IDs; it must record the seed, number of resamples, and whether the same task seeds were paired. A single run per condition may report descriptive counts only, not a reliability claim.

## 4. Required trace artifact

The evaluator consumes an immutable JSON trace. Raw output content is optional once a fingerprint is captured; this avoids making metric collection a new secret-retention mechanism.

```json
{
  "schema_version": "nyawa.compaction-eval/v1",
  "suite_id": "local-fixture",
  "task_id": "task-001",
  "run_id": "run-001",
  "condition": "summary_only",
  "environment": {
    "nyawa_revision": "git-or-build-identifier",
    "actor_revision": "model-or-agent-identifier",
    "summary_mode": "captured",
    "fingerprint_policy_id": "tool-output-bytes-v1"
  },
  "compactions": [],
  "tool_events": [],
  "recall_probes": []
}
```

All top-level IDs are opaque strings. The evaluator must reject a trace with duplicate `task_id`/`run_id`/`condition` records when aggregating a result set, rather than silently choosing one.

### 4.1 Compaction records

Each completed proposed checkpoint is recorded as:

```json
{
  "checkpoint_id": "ccp_v1_<digest>",
  "event_index": 20,
  "source_sha256": "sha256:<digest>",
  "summary_sha256": "sha256:<digest>",
  "residual_pack_sha256": "sha256:<digest>",
  "namespace": "context"
}
```

Rules:

- `event_index` is the first event after the completed boundary. It is strictly increasing within a run.
- `checkpoint_id` and `residual_pack_sha256` are required only for `summary_plus_residual_pack`; a legacy `summary_only` trace may use a harness-generated `legacy_boundary_id` and set residual fields to `null`.
- The evaluator verifies that a tool event attributed to a checkpoint has an `event_index` greater than or equal to that checkpoint's boundary and smaller than the next boundary's event index.
- A checkpoint field must never be populated for a dry run or a no-op result.

### 4.2 Tool events

```json
{
  "event_index": 21,
  "tool_name": "terminal",
  "tool_input_sha256": "sha256:<digest>",
  "output_sha256": "sha256:<digest>",
  "outcome": "failure",
  "error_class": "nonzero_exit",
  "is_planning": false,
  "is_polling": false,
  "checkpoint_id": "ccp_v1_<digest>"
}
```

Required fields are `event_index`, `tool_name`, `output_sha256`, `outcome`, `is_planning`, and `is_polling`. `tool_input_sha256`, `error_class`, and `checkpoint_id` may be `null` when unavailable or before the first compaction.

`outcome` is exactly one of:

```text
success | failure | tool_error | timeout | cancelled | unknown
```

The trace producer, not the evaluator, assigns the outcome from provider-native data. Examples:

- a command whose exit status is non-zero is `failure` with `error_class: "nonzero_exit"`;
- an explicit MCP/provider error response is `tool_error`;
- a completed timeout is `timeout`;
- an agent- or user-requested cancellation is `cancelled`, not an error by default;
- an adapter that cannot determine status emits `unknown`.

### 4.3 Output fingerprint policy

The default `tool-output-bytes-v1` fingerprint is SHA-256 over the exact UTF-8 bytes emitted by the tool adapter. It does not trim whitespace, remove timestamps, redact text, parse JSON, or normalize line endings. JSON outputs may instead use a declared canonical-JSON policy, but two compared conditions must use the same policy ID.

A trace may omit the raw output after computing `output_sha256`. If a producer redacts or normalizes output before hashing, it must use a different policy ID and cannot be pooled with strict-byte traces. This avoids false repeat reductions caused by an undocumented normalizer.

## 5. Metric definitions

### 5.1 Repeated tool outputs

The default definition follows the REMORY paper's aggregate description: identical outputs from the same tool within a task count as repeats; planning and polling are excluded.

Let `E` be all tool events in one task run. Let the eligible set be:

```text
A = { e in E | !e.is_planning && !e.is_polling && e.output_sha256 != null }
```

Group eligible events by `(tool_name, output_sha256)`. For each group `g`, its repeat contribution is:

```text
repeat_contribution(g) = max(0, count(g) - 1)
```

The first occurrence is therefore not a repeat; every later identical output from the same tool is. Events from different tool names never share a group even when their output digest matches. The input digest is reported diagnostically but does not alter the primary definition.

Report all of the following:

| Name | Definition |
|---|---|
| `repeated_tool_outputs` | Sum of group repeat contributions. |
| `eligible_tool_outputs` | Size of `A`. |
| `repeated_tool_output_rate` | `repeated_tool_outputs / eligible_tool_outputs`; `null` when the denominator is zero. |
| `post_compaction_repeated_tool_outputs` | Same count over eligible events attributed to a completed checkpoint. |
| `repeat_groups` | Optional audit rows with tool name, digest, count, first event index, and checkpoint attribution. |

A failed event may contribute to both the repeat and error metrics. This is intentional: the metrics describe different properties and must not be made mutually exclusive.

The evaluator also records a secondary, **not-primary** retry diagnostic grouped by `(tool_name, tool_input_sha256, output_sha256)`. It can distinguish repeated identical calls from repeated identical observations, but must not replace the primary metric without an explicit version bump.

### 5.2 Tool errors

The primary tool-error numerator is:

```text
error_tool_outputs = count(e in E where e.outcome in {failure, tool_error, timeout})
```

`cancelled` and `unknown` do not count as errors in the primary metric. They are reported separately because a cancellation may be intentional and an unknown adapter state is a data-quality issue, not evidence of success.

Unlike the repeat metric, the primary error metric includes planning and polling events when they fail. This matches the narrower paper wording: the planning/polling exclusion is stated for repeat counts, while errors are failed commands or explicit tool errors. For operational analysis, report an additional `nonplanning_nonpolling_tool_errors` count, but do not substitute it for the primary count.

| Name | Definition |
|---|---|
| `tool_errors` | `failure + tool_error + timeout` events. |
| `tool_attempts` | All tool events, including planning/polling and unknown outcomes. |
| `tool_error_rate` | `tool_errors / tool_attempts`; `null` when the denominator is zero. |
| `post_compaction_tool_errors` | Errors attributed to a completed checkpoint. |
| `cancelled_tool_outputs` | Count of `cancelled` events. |
| `unknown_tool_outcomes` | Count of `unknown` events; a data-completeness warning. |

### 5.3 Recall after compaction

This metric measures retrievability, not whether the actor consumed the retrieved result. A probe is valid only if it is issued after a completed boundary and its targets are observable in the tested retrieval surface.

A current-compatible Nyawa probe is:

```json
{
  "probe_id": "probe-001",
  "checkpoint_id": "legacy_or_canonical_boundary_id",
  "query": "fixed query text",
  "namespace": "context",
  "limit": 5,
  "min_score": 0,
  "exclude_types": [],
  "target_memory_ids": ["summary_or_other_active_memory_id"],
  "target_residual_entry_ids": [],
  "surface": "nyawa_recall"
}
```

For `surface: "nyawa_recall"`, the harness sends the exact documented request shape to `POST /v1/recall` and reads ordered `results[].id`. A hit occurs when at least one `target_memory_ids` value appears within the first `k` result IDs.

For a future `surface: "residual_pack_lookup"`, targets are `target_residual_entry_ids` and lookup is only valid after the proposed checkpoint/pack read API exists. It must be reported separately from Nyawa's general recall endpoint; a dedicated SQLite pack table is not automatically searched by `POST /v1/recall`.

For a valid probe `p`, with target set `T_p` and returned ordered IDs `R_p`:

```text
hit@k(p) = 1 if any target in T_p occurs in R_p[0:k], else 0
recall@k = mean(hit@k(p)) over valid probes
reciprocal_rank(p) = 1 / first target rank, or 0 when absent
MRR@k = mean(reciprocal_rank(p)) over valid probes, treating ranks above k as 0
```

Report `recall_after_compaction@1`, `@3`, `@5`, and `MRR@5` when the configured limits permit them. Also report:

| Name | Meaning |
|---|---|
| `valid_recall_probes` | Probes whose targets are active and observable on the stated surface. |
| `unobservable_recall_probes` | Probes for data that was never persisted to that surface, for example a current pruned raw tool payload. These are excluded, not counted as misses. |
| `superseded_target_probes` | Probes whose target is intentionally excluded by Nyawa's active-memory recall behavior. Report separately unless testing historical/time-travel behavior. |
| `pre_to_post_recall_delta` | Optional paired difference between a frozen pre-compaction probe and its post-compaction counterpart on the same retrieval snapshot. |

The current compactor stores a `TypeContext` summary but does not archive raw tool results. Consequently, a probe targeting an omitted raw tool payload is not a valid current `nyawa_recall` probe unless another explicit active memory holds that information. Reporting it as a miss would falsely measure an unimplemented archive feature.

### 5.4 Macro, micro, and deltas

For every rate, report both when applicable:

- **Micro:** sum numerators across runs divided by sum denominators across runs.
- **Macro:** arithmetic mean of non-null per-run rates, with the contributing-run count.

For a paired candidate/control result, report absolute delta as `candidate - control`. Report relative delta only when the control denominator is non-zero; otherwise use `null` rather than an invented percentage. Raw numerators and denominators remain the primary evidence.

## 6. Validation and attribution rules

Reject an invalid trace rather than silently repairing it:

1. Tool-event `event_index` values must be strictly increasing and unique within a run.
2. Compaction boundary event indexes must be strictly increasing and unique.
3. A non-null event `checkpoint_id` must be the most recent completed checkpoint at that event index.
4. `tool_name` and each `sha256:` digest must be non-empty and use the declared fingerprint policy.
5. A `nyawa_recall` probe must have at least one `target_memory_ids` member; a residual lookup probe must have at least one residual entry target.
6. A probe's `namespace`, `limit`, `min_score`, and `exclude_types` must be recorded exactly, because all can change recall results.
7. The evaluator must preserve failed and unknown calls in the trace even when their raw output body is unavailable.

For deterministic offline metric tests, scorer functions should be pure: input trace plus evaluator version yields result JSON with no database, clock, network, model, or random dependency.

## 7. Result artifact

A scorer implementation should emit a machine-readable result with both counts and policy metadata. `internal/compactioneval.MarshalResultJSON` emits this shape; the package does not perform file or network I/O:

```json
{
  "schema_version": "nyawa.compaction-eval-result/v1",
  "evaluator_version": "v1",
  "condition": "summary_only",
  "fingerprint_policy_id": "tool-output-bytes-v1",
  "runs": 1,
  "repeated_tool_outputs": 0,
  "eligible_tool_outputs": 0,
  "repeated_tool_output_rate_micro": null,
  "tool_errors": 0,
  "tool_attempts": 0,
  "tool_error_rate_micro": null,
  "recall_after_compaction": {
    "valid_recall_probes": 0,
    "unobservable_recall_probes": 0,
    "recall_at_1": null,
    "recall_at_3": null,
    "recall_at_5": null,
    "mrr_at_5": null
  },
  "warnings": []
}
```

A zero count is different from an unavailable rate. Empty denominators must yield `null` rates and a clear contributing-run count.

## 8. Normative conformance cases for future pure scorer tests

No scorer exists in this change, so these are test specifications rather than newly added Go tests. Once a pure scorer is introduced, table-driven tests should cover at least the following cases without starting a server or using a live database.

| Case | Input condition | Expected metric behavior |
|---|---|---|
| Same tool, same output three times | Three eligible `terminal` events share one output digest. | One group contributes two repeats; eligible output count is three. |
| Different tools, same output | Two eligible events share an output digest but have different tool names. | Zero repeats because tool name is part of the primary group key. |
| Polling duplicate | Two otherwise identical events are both `is_polling: true`. | Both are excluded from repeat numerator and denominator. |
| Repeated failure | Two eligible same-tool events share a digest and have `outcome: failure`. | The second contributes a repeat; both contribute tool errors. |
| Cancellation | One `cancelled` event. | It counts as an attempt and cancellation, but not a primary error. |
| Unknown outcome | One `unknown` event. | It counts as an attempt and unknown outcome; it is not treated as success or error. |
| Pre/post attribution | An event lies after checkpoint A and before checkpoint B. | It is attributed only to A's post-compaction bucket. |
| Current recall hit | An active expected memory ID is returned at rank at most `k`. | Hit@k is one and reciprocal rank uses that first rank. |
| Current unobservable source | Target is a pruned raw tool result not persisted as an active memory. | Probe is classified unobservable and excluded from recall denominator. |
| Superseded target | Target has been superseded and active recall omits it. | Probe is separately classified, not silently treated as a normal retrieval failure. |

The expected values in these cases should be generated by the test's pure scorer assertions, not copied into a benchmark report by hand.

## 9. Recommended future harness shape

A future additive harness can live beside `scripts/recall-bench/` as `scripts/compaction-eval/`, with two separable layers:

1. **Trace collector/adapter:** accepts provider-specific tool events and records a redaction-aware immutable trace. It is responsible for explicit outcomes and output fingerprints.
2. **Pure scorer:** validates JSON, groups events, computes metrics, and writes result JSON/Markdown. This layer should have focused unit tests only.
3. **Recall runner:** executes recorded `nyawa_recall` probes against an isolated temporary or snapshot database and appends ordered memory IDs to the trace/result. It must record Nyawa revision, recall environment variables, embedder availability, and query filters.

Do not use the production database for a benchmark that writes summaries or creates proposed checkpoint rows. `scripts/recall-bench/` is read-only by design; a compaction evaluator must instead use an isolated test database or replay pre-recorded responses. If testing current fallback compaction, run in an environment with `NYAWA_LLM_API_KEY` absent or replay a captured summary as described above.

## 10. Interpretation limits

- Matching output digests measure duplicate observations, not redundant intent. Dynamic timestamps or random IDs can hide semantically repeated work under the strict-byte default.
- Fewer repeats or errors can result from shorter or divergent trajectories. Always report task outcome, total tool attempts, compaction count, and token/cost context beside the metrics.
- Recall hit@k measures retrieval availability, not factual correctness, source attribution, or an agent's use of the result.
- Existing Nyawa recall excludes superseded memories and is affected by the recorded namespace, filters, ranking configuration, index availability, and concurrent writes. Isolate and record these conditions.
- A residual pack may improve a future direct rehydration surface without improving general `POST /v1/recall`; the surfaces must not be conflated.

## 11. Implementation status matrix

| Item | Status |
|---|---|
| General recall@k/MRR benchmark for CLI queries | Implemented in `scripts/recall-bench/`; not compaction-aware. |
| `/v1/compact`, MCP compaction, and coarse compaction stats | Implemented. |
| Tool-event trace schema | Proposal only. |
| Repeat/output fingerprint scorer | Implemented in `internal/compactioneval`; no collector integration. |
| Tool-error scorer | Implemented in `internal/compactioneval`; no collector integration. |
| Checkpoint attribution | Implemented in `internal/compactioneval` for supplied trace boundaries. |
| ID-based recall-after-compaction harness | Pure scoring is implemented; probe execution/DB harness remains proposal only. |
| Residual-pack lookup metric | Pure scoring is implemented; residual-pack lookup remains proposal only. |
| Focused pure metric tests | Implemented as table-driven tests in `internal/compactioneval/compactioneval_test.go`. |

## 12. Reference

- Hanchen Xia et al., *REMORY: Learning Residual Memory for Context Compaction*, arXiv:2610.11287. The paper describes repeated outputs as identical outputs from the same tool within a task while excluding planning and polling, and reports failed commands or explicit tool errors separately. This specification operationalizes those concepts for an inspectable Nyawa trace rather than reproducing REMORY's learned memory system.

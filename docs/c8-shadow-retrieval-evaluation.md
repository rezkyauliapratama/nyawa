# C8 - Shadow Retrieval Evaluation

## Purpose

C8 asks whether Nyawa returns the expected synthetic memory, in the right namespace, near the top of the result list. It measures retrieval candidates only. It does not measure answer quality from Hermes or any LLM.

## Execution boundary

- Manual `workflow_dispatch` on a GitHub-hosted runner; no VM and no production host load.
- Pinned Nyawa/BGE image digest `sha256:61da25caf922c0925f4fd8e3af1d811f6fc9d3a9df21beb44138c7300d47efb6`.
- Only synthetic records with namespaces prefixed `c8-synthetic-`.
- Disposable container: network disabled, read-only root, tmpfs-only scratch, no host mounts/ports, 4 GiB memory, 2 CPUs, 128 PIDs.
- No production database, real-user memory, Hermes request, provider call, context injection, collector, or Residual Pack.
- Cleanup checks that the container belongs to this run and reads back that it is absent.

## Corpus

`tests/fixtures/c8-shadow-retrieval-fixtures.json` has 5 query cases in each supported category, 25 queries total. The manual workflow defaults to `quick-smoke`, which runs these six deterministic cases after validating the full fixture:

- `Q01_SHA`: exact identifier;
- `Q06_DECISION`: semantic decision;
- `Q11_SCOPE_01` and `Q12_SCOPE_02`: namespace isolation in both directions;
- `Q16_ACTIVE_LEASE`: active vs retired filtering;
- `Q21_NO_HIT_LUNAR`: negative/no-answer query.

The quick scope seeds the same 31 fictional records and only reduces recall calls from 25 to 6; image/model startup and indexing still occur. Select `full` in *Run workflow* to run all 25 cases. A passing quick smoke proves only those six cases; it is not a full C8 gate or production-quality claim.

The remaining cases cover semantic/bilingual paraphrases, additional exact identifiers, active-vs-retired filtering, and negative queries.

All records and queries are fictional. The SHA-256 display is split into eight-character groups because Nyawa's secret filter intentionally rejects long contiguous key-like strings; it is a synthetic grouped display, not a canonical ungrouped digest. Retired records use type `superseded` so the API's `exclude_types` path can be exercised. This does **not** test the Nyawa supersession lifecycle or `superseded_at` mutation; the C4/C7-lite policy tests remain responsible for that. C8 also does not decide conflict policy.

## Measurement

For `k=5`, report:

- Recall@5: expected markers found in the top five;
- Precision@5: relevant returned candidates divided by candidates returned;
- MRR: reciprocal rank of the first expected candidate;
- exact-identifier Recall@5;
- cross-namespace and forbidden/stale marker leakage;
- false-positive rate for negative queries;
- per-query latency, median and p95;
- candidate character count and estimated token count (`ceil(chars / 4)`). The token count is a rough local estimate, not a provider tokenizer or billed call.

`B0` is an empty candidate set (no Nyawa retrieval); `B1` is Nyawa's returned candidates. The comparison measures candidate coverage only, not whether an LLM answer improved.

## Initial pass gates

- Overall positive Recall@5 >= 0.90.
- Exact-identifier Recall@5 = 1.00.
- Namespace leakage = 0.
- Forbidden-marker leakage, including excluded `superseded` type = 0.
- Every returned candidate has a numeric score.
- Each query requests at most 5 candidates; the report stores IDs, scores, counts, and timings, not full candidate text.

Precision@5, negative-query false positives, latency, and estimated tokens are reported for calibration, not hard-gated in this first corpus. Revise thresholds only after reviewing actual baseline results; do not claim production-quality improvement from this synthetic set.

## Fast run procedure

1. Open **Actions → C8 Shadow Retrieval Evaluation → Run workflow**; it is manual-only and does not run on pushes.
2. Keep `quick-smoke` for the six-case run. Choose `full` only when the full 25-case gate is wanted.
3. Review the job summary, then download the artifact named with scope, run ID, and attempt. The report/JSONL trace make the exact image, fixture, source revision, stages, timings, and selected cases auditable.

## Artifacts and next gate

The workflow writes a bounded JSON report to the job summary and uploads `c8-eval-report.json` plus `c8-eval-trace.jsonl` as a run-scoped artifact retained for 14 days. Trace entries include UTC time, run/attempt, stage, elapsed time/counts, and case IDs only—no query or candidate text. The report records the selected scope, image digest, fixture SHA-256, source SHA, runtime version, vector count, selected cases, metrics, and observation-only flags. Container/tmpfs cleanup is verified before artifact upload.

A `quick-smoke` pass proves only the six selected cases; run `full` for the complete 25-case synthetic C8 gate. Neither mode authorizes C9 activation, VM creation, production recall, or Hermes injection.

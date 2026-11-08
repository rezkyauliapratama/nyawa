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

`tests/fixtures/c8-shadow-retrieval-fixtures.json` has 5 query cases in each supported category, 25 queries total:

- exact identifiers: SHA-256, path, URL, and other literal anchors;
- semantic/bilingual paraphrases;
- namespace isolation;
- active-vs-retired type filtering (`exclude_types`);
- negative queries with no expected answer.

All records and queries are fictional. The retired records use type `superseded` so the API's `exclude_types` path can be exercised. This does **not** test the Nyawa supersession lifecycle or `superseded_at` mutation; the C4/C7-lite policy tests remain responsible for that. C8 also does not decide conflict policy.

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

## Artifacts and next gate

The workflow writes metrics to its job summary and logs, then deletes its disposable container and tmpfs database. A pass completes the synthetic C8 gate only. It does not authorize C9 activation, VM creation, production recall, or any Hermes injection.

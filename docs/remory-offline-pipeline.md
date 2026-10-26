# REMORY offline pipeline status

| Slice | Source status | Runtime status | Safety boundary |
|---|---|---|---|
| P0 passive collector | Implemented in the Hermes source tree; disabled by default | Not active by default | Hashes/redaction only; no raw tool payloads |
| Trace assembly | Implemented in `internal/compactioneval` and CLI `compaction-eval assemble` | Offline only | One run per stream; deterministic ordering; missing summary hash is rejected |
| P2 recall probes | Implemented in `internal/compactioneval` and CLI `compaction-eval recall-probe` | Offline only | Requires explicit `--snapshot`; SQLite `mode=ro` and `query_only`; no production default |
| P3 baseline report | Implemented in `internal/compactioneval` and CLI `compaction-eval baseline` | Offline only | Consumes immutable traces/results; empty input is not reported as collected |
| P4 residual pack | Pure builder implemented as shadow/dry-run code and CLI | Proposal-only at runtime | No DB schema, endpoint, MCP field, prompt injection, or active selection |
| P5 utility curator | Plan only | Not implemented | Requires separate utility labels and validation |
| P6 hierarchical/soft-token research | Plan only | Not implemented | No model, token, or learned-state dependency added |

The assembler intentionally does not convert a passive recall observation into a
recall probe: returned IDs are observations, not ground-truth targets. P2 probe
inputs must provide target IDs explicitly. The baseline report can aggregate
repeated outputs, tool errors, attempts, compaction presence, and optional token /
cost fields, but no observation data is synthesized.

The residual builder preserves explicit structured tool metadata only. It uses
canonical normalized source envelopes, stable SHA-256 identities, deterministic
selection and ordering, bounded UTF-8 excerpts/items/pack bytes, and redaction.
Its output is a sidecar preview and is never persisted or injected into a prompt.

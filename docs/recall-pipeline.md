# Recall pipeline

This document describes how `nyawa recall` (and the same pipeline behind
`nyawa_recall`, `POST /v1/recall`, and the dashboard) turns a free-text query
into a ranked list of memories. It is the long-form companion to the "Hybrid
Search & RRF" section of the [README](../README.md).

Everything here is implemented in `internal/search/` and `internal/store/`.
Which source file owns which number is called out inline. The tuning
coefficients are centralised in `internal/search/weights.go`; if the code and
this document ever disagree, the code wins.

## 1. Overview

Recall is hybrid: every query is run against two independent candidate
generators, the results are fused by weighted Reciprocal Rank Fusion (RRF), and
the fused ordering is then adjusted by small bounded relevance boosts.

```
query
  -> embed query
  -> vector leg:   HNSW nearest neighbours
  -> keyword leg:  SQLite FTS5 / BM25 over memories_fts
  -> weighted RRF fusion (k = 5)
  -> drop superseded
  -> post-fusion weighting (recency, importance, type, access, length, pinned, edge_count)
  -> graph merge (entity seeds; overlap boost; graph-only injection)
  -> time-travel filter (optional)
  -> min_score / exclude_types filter
  -> top-K cut
```

Steps run in `Pipeline.Search` (`internal/search/pipeline.go`). The two
candidate legs run concurrently in goroutines; the pipeline waits for both
before fusing.

## 2. Candidate generation

### 2.1 Query embedding

The query text is embedded with the configured embedder chain. If the embedder
is unavailable the pipeline logs it and continues with the keyword leg only
(`embedder unavailable, falling back to FTS5-only search`). Embedding failure
is not fatal.

### 2.2 Candidate count

Both legs are asked for the same number of candidates:

```
searchTopK = max(limit * 3, 50)
```

`limit` defaults to 20 (`types.DefaultQueryLimit`) when the caller passes 0 or
less; the CLI `recall` command passes 10 explicitly. The `max(..., 50)` floor
means the legs always return at least 50 candidates so RRF has enough to work
with even for a small final limit.

### 2.3 Vector leg (HNSW)

`store.VectorSearch` runs the nearest-neighbour search over the pure-Go HNSW
index (`<db>.hnsw`). The index holds a vector per embedded memory; a memory
stored while no embedder was available has no vector until `nyawa reindex`
fills it in (see [Reindexing](#7-reindexing-the-vector-index)). When a namespace
or time filter is set the vector IDs are post-filtered against `memories` with
`superseded_at IS NULL`.

### 2.4 Keyword leg (FTS5 / BM25)

The keyword leg queries the `memories_fts` FTS5 virtual table with
`ORDER BY rank`, which is FTS5's BM25 relevance. Documents that contain more,
and rarer, query terms rank first.

The free-text query is **not** passed to `MATCH` as-is. Raw text is fragile:
whitespace is an implicit `AND` (so a normal multi-word query matches almost
nothing), and punctuation is query syntax (`crypto-data` is parsed as a column
filter and errors with `no such column: data`; `SKILL.md` is invalid syntax).
Both symptoms caused the keyword leg to return **zero rows silently**, and
because fusion then saw only the vector list, recall quietly degenerated to
vector-only. That is the bug fixed by the FTS handle's `buildFTSMatchExpr`
(`internal/store/sqlite.go`).

The rewrite (`buildFTSMatchExpr` + `ftsTokens`):

1. Iterate runes; keep letters and digits, treat everything else as a
   separator. This strips `-`, `.`, `_` and similar punctuation.
2. Lowercase each token.
3. Drop tokens shorter than 2 runes (`ftsMinTokenLen`).
4. Drop duplicates.
5. Cap the list at 32 tokens (`ftsMaxTokens`) so a long query cannot blow up
   the statement.
6. Quote each remaining token and join them with ` OR `.

So `crypto-data` becomes `"crypto" OR "data"` and `SKILL.md` becomes
`"skill" OR "md"`. OR (not AND) is deliberate: a target memory does not have to
contain every query token to be a candidate; BM25 still ranks by how many and
how rare the matches are.

The rewrite is gated by `ftsRewriteEnabled()`, which mirrors
`search.RecallV2Enabled()` for the store layer (the store cannot import
`search` without an import cycle). Setting `NYAWA_RECALL_V2=0` disables it.

If the rewritten `MATCH` still returns an error, the pipeline logs
`fts5 search failed, falling back to vector-only` and drops the keyword list
rather than failing the whole query. Symmetrically, if the vector search
returns an error the whole query fails, because the vector leg is considered
required.

## 3. Fusion: weighted RRF

`RRF.FuseWeighted` (`internal/search/rrf.go`) merges the two ranked ID lists.
RRF uses only ranks, never raw scores, because HNSW cosine distances and BM25
scores are not comparable. For each memory:

```
raw(item) = w_v * 1/(k + rank_v(item)) + w_f * 1/(k + rank_f(item))
```

where a missing rank contributes nothing. The raw score is then normalised to
`[0,1]`:

```
norm      = (w_v + w_f) / (k + 1)
score(item) = raw(item) / norm
```

Counts and defaults:

| Symbol | Meaning | Default |
|--------|---------|---------|
| `w_v` | vector-leg weight | `1.0` (`DefaultVectorWeight`) |
| `w_f` | keyword-leg weight | `3.0` (`DefaultFTSWeight`) |
| `k` | RRF damping constant | `5` (`DefaultRRFK`) |

The keyword leg is trusted more (`w_f = 3`) because vector similarity is
computed over the whole memory text, so long documents dominate it and exact
keywords are missed, and because part of the corpus may have no vector at all.
The vector leg still contributes, so a purely semantic match with no keyword
overlap can still be retrieved. Values `<= 0` fall back to `1` so a
misconfiguration cannot silently drop a whole modality.

`k = 5`, not the textbook `60`, is intentional: at `k = 60` the gap between
adjacent ranks is tiny, so a memory that is mediocre in both lists can outrank
a memory that is the best hit in one list. A small `k` sharpens the fused
ranking toward the true top matches.

Normalisation matters because the boosts added next are on a similar scale:
without it the raw RRF score would be tiny next to every boost and a weakly
relevant but "important" memory would outrank the true nearest neighbour.

## 4. Post-fusion weighting

`PostProcessor.Process` (`internal/search/rrf.go`) turns each fused candidate
into a scored `MemoryResult`. A memory whose `superseded_at` is set is skipped
here as well as in SQL (defence in depth). The score is:

```
score = RRFScore
      + Recency    * exp(-ageHours / tau)
      + Importance * clamp01(importance)
      + Type       * RetrievalFactor(type)
      + Access     * min(access_count / 10, 1)
      + Length     * lengthFactor(len(content))
      + (Pinned if pinned)
      + Graph      * log1p(edge_count)
```

| Signal | Weight (default) | Behaviour |
|--------|------------------|-----------|
| recency | `0.05` | exponential decay `exp(-ageHours / tau)`, `tau` = per-type half-life (`DecayHours`, fallback 168) |
| importance | `0.10` | `clamp01(importance)` |
| mem_type | `0.12` | multiplied by `RetrievalFactor(type)` |
| access_count | `0.05` | saturates at 10 recalls (`min(count/10, 1)`) |
| content length | `0.05` | `lengthFactor`: content `<= 400` bytes scores `1.0`; longer decays as `1 / (1 + ln(n/400))` |
| pinned | `0.10` | flat bonus when `pinned` is true |
| edge_count | `0.03` | `log1p(edge_count)`, so the first edges matter most |

`RetrievalFactor` (`internal/types/memory.go`) lifts durable, high-signal
memories and never boosts raw chatter:

| Type | Factor |
|------|--------|
| rule, decision | 1.0 |
| preference | 0.9 |
| procedure | 0.8 |
| fact, insight | 0.7 |
| context, event | 0.5 |
| note, reference | 0.3 |
| conversation | 0.0 |

`conversation` is deliberately 0.0: transcripts are retrieval noise and must
not be boosted for existing. The content-length factor stops a long document
from winning purely because its embedding happens to look similar; length used
to be an unbounded advantage.

After scoring, results are sorted by score descending and re-ranked. A small
pool (`pool.ResultPool`, size 64) and a per-process result cache (256 entries,
5 minute TTL) sit in front of the store. The cache key is built from every
field that can change the result set: query text, namespace, limit, and, only
when non-zero, `min_score`, `exclude_types`, and time travel.

## 5. Graph merge

After weighting, `Pipeline.mergeGraphResults` runs entity-graph recall
(`internal/search/pipeline.go`):

1. Build seeds: entity names at least 3 characters long that appear in the
   query (case-insensitive substring match), capped at 3, listed from
   `ListEntityNames(10000)`. No seeds means the pipeline keeps the plain RRF
   result (`pure RRF fallback`).
2. Traverse from the seeds with depth 2 and `limit * 2`.
3. Memories already in the fused result that are also graph-reachable get a
   multiplicative boost `score *= 1 + OverlapWeight` (`OverlapWeight = 0.1`,
   so `x1.1`). The original hard-coded `1.5x` was reduced because it buried
   precise single-list matches.
4. Graph-only memories are injected, but their graph path score (an unbounded
   sum over paths) is first normalised to the best graph hit and then scaled by
   `GraphInjectWeight = 0.1`, and the same recency/importance/type/access/
   length/pinned/edge boosts are added so they are comparable to RRF-ranked
   results.
5. Results are re-sorted by score and re-ranked.

## 6. Filters, limit, and side effects

`applyFilters` (`internal/search/pipeline.go`) runs in this order:

- time travel, if `--at` / `time_travel` was given: keep memories created at or
  before the target time, and drop those superseded before it;
- `min_score`: drop results whose final `score` is strictly below the
  threshold. A value `<= 0` disables the filter rather than selecting
  everything. Because the score is the normalised RRF value in `[0,1]` plus
  small boosts, a useful threshold is well under `1` (roughly `0.1` to `0.3`);
- `exclude_types`: drop results whose memory `type` is in the list (for example
  `["note","conversation"]`).

Only then is the `limit` cut applied, so `limit` means "at most this many
results after filtering". The CLI `nyawa recall` does not expose these two
filters and always prints up to 10 results; `nyawa_recall` and
`POST /v1/recall` accept them (`query`, `namespace`, `limit`, `min_score`,
`exclude_types`).

Finally, `access_count` is incremented for every returned memory, in a
background goroutine that does not block the response.

## 7. Reindexing the vector index

`nyawa reindex <db>` (`cmd/nyawa/main.go`, `cmdReindex`) re-embeds active
memories that are missing a vector:

- It lists all active memories via `ListAllMemories`.
- For each, it checks HNSW membership first (`hnsw.Contains`), so already
  indexed memories are counted as `already` and never re-embedded.
- Missing ones are embedded and inserted. Embed failures (for example, no
  live embedder) are counted as `failed`.
- If anything was reindexed it persists through `HNSW.MergeAndSave`, which
  reloads the on-disk index under the cross-process lock, merges vectors a
  running gateway may have written during the run, and writes the union once,
  atomically. This makes reindex safe against a live database.

The command logs coverage, for example:

```
nyawa: hnsw: nothing to reindex, 8386 vectors in memory (coverage 187.5% of 4473 active memories)
Reindexed 0 memories (0 already indexed, 0 failed)
```

Coverage is `persisted / total active memories * 100`; values above 100% are
expected because the index can retain vectors for memories that are no longer
active. Without a live embedder the command logs `BGE unavailable` and the
memories that still need a vector are reported as failed.

## 8. Environment overrides

Every coefficient above is a named constant in `internal/search/weights.go` and
can be overridden at runtime, with no rebuild, through `NYAWA_RECALL_*`
variables. Unknown or unparseable values are ignored, so a typo never silently
zeroes a weight. The values are read per process, so set them in the environment
of `nyawa serve`, `nyawa mcp`, or whichever command is doing the recall.

| Env var | Default | Meaning |
|---------|---------|---------|
| `NYAWA_RECALL_V2` | on | `0`, `false`, `off`, `no` restore the pre-v2 ranking and disable the FTS query rewrite. Any other value (including unset) enables v2. |
| `NYAWA_RECALL_RRF_K` | `5` | RRF damping constant (must be `> 0`). |
| `NYAWA_RECALL_W_VECTOR` | `1` | vector-leg RRF weight. |
| `NYAWA_RECALL_W_FTS` | `3` | keyword-leg RRF weight. |
| `NYAWA_RECALL_W_RECENCY` | `0.05` | recency boost. |
| `NYAWA_RECALL_W_IMPORTANCE` | `0.10` | importance boost. |
| `NYAWA_RECALL_W_TYPE` | `0.12` | memory-type boost. |
| `NYAWA_RECALL_W_ACCESS` | `0.05` | access-count boost. |
| `NYAWA_RECALL_W_LENGTH` | `0.05` | content-length boost. |
| `NYAWA_RECALL_W_PINNED` | `0.10` | pinned boost. |
| `NYAWA_RECALL_W_GRAPH` | `0.03` | edge-count boost. |
| `NYAWA_RECALL_W_OVERLAP` | `0.10` | multiplicative boost for graph-reachable memories already in the fused list (`1 + weight`). May be `0` to disable. |
| `NYAWA_RECALL_W_GRAPH_INJECT` | `0.10` | injection weight for graph-only memories. May be `0` to disable. |

### Legacy mode (`NYAWA_RECALL_V2=0`)

This is the A/B escape hatch that reproduces the pre-v2 behaviour exactly:

- symmetric RRF, `w_v = w_f = 1`, at `k = 60`;
- the original `1.5x` overlap multiplier (`OverlapWeight` becomes `0.5`, since
  `1 + 0.5 = 1.5`);
- the old importance formula `Importance * type.Weight() * accessFactor`,
  without the type/access/length boosts;
- the FTS query rewrite is disabled, so the raw query goes to `MATCH` again.

## 9. Benchmark harness

`scripts/recall-bench/` measures recall against a live database. It runs a
fixed query set through a `nyawa` binary, parses the ranked output, and
computes recall@1, recall@3, recall@5 and MRR. It opens the database read-only
and never writes. It also probes the FTS5 leg directly, including a `ftsprobe`
Go helper that reproduces the runtime driver's raw `MATCH` behaviour.

```bash
# baseline against the runtime binary
python3 scripts/recall-bench/bench.py

# a candidate binary, under a separate label so results do not overwrite
python3 scripts/recall-bench/bench.py --binary ./nyawa-candidate --label candidate
```

See `scripts/recall-bench/README.md` for all arguments and the query file
format. Target matching is by content, not ID, because `nyawa recall` prints
content, not IDs; `recall` also returns at most 10 rows, so ranks above 10 are
reported as misses.

### Baseline numbers

Measured 2026-10-03 on the production database (12 queries, runtime binary
reporting `v1.2.0`), from `scripts/recall-bench/BASELINE-20261003.md`:

| Path | recall@1 | recall@3 | recall@5 | MRR |
|------|----------|----------|----------|-----|
| recall CLI, vector + RRF (before fix) | 0.333 | 0.417 | 0.500 | 0.392 |
| BM25 FTS token-OR (ceiling of the fix) | 0.833 | 1.000 | 1.000 | 0.903 |
| BM25 FTS token-AND | 0.500 | 0.667 | 0.667 | 0.569 |
| BM25 FTS raw (engine) | 0.333 | 0.500 | 0.500 | 0.403 |

After the FTS query rewrite shipped, the hybrid recall reaches the token-OR
ceiling on the same query set:

| Path | recall@1 | recall@3 | recall@5 | MRR |
|------|----------|----------|----------|-----|
| recall CLI, hybrid (after fix) | 0.833 | 1.000 | 1.000 | 0.903 |

The before numbers and the per-query breakdown, including the note that one
target was already superseded, live in `scripts/recall-bench/BASELINE-20261003.md`
and `scripts/recall-bench/BASELINE-NOTES.md`. The baseline is a dated artifact
measured against the pre-fix engine; its per-query "raw" columns describe the
bug, not the current rewrite.

## 10. Where the numbers live

| File | Contents |
|------|----------|
| `internal/search/weights.go` | every boost and RRF weight constant, env overrides, `NYAWA_RECALL_V2` toggle |
| `internal/search/rrf.go` | RRF fusion and the post-fusion post-processor |
| `internal/search/pipeline.go` | leg orchestration, graph merge, filters, cache |
| `internal/store/sqlite.go` | `FTS5Search`, `buildFTSMatchExpr`, `ftsTokens`, vector search |
| `internal/types/memory.go` | `RetrievalFactor`, `Weight`, `DecayHours`, `StoreQuery` |
| `cmd/nyawa/main.go` | `cmdRecall`, `cmdReindex` |
| `scripts/recall-bench/` | benchmark harness, query set, baseline |

## 11. Legacy disclaimers

- The legacy `MemoryType.Weight()` values still exist because the
  `NYAWA_RECALL_V2=0` path consumes them; the v2 path uses `RetrievalFactor`.
- `search.K = 60` remains exported in `pipeline.go` for compatibility but is
  not used by the v2 fusion path, which reads its `k` from the resolved
  weights.

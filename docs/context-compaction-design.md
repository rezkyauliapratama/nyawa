# Nyawa Context Compaction - Design Document

## Context Compaction Service + Hermes Integration

Document ID: NYAWA-ARCH-001
Version: 0.1
Classification: Internal - Architecture & Engineering
Author: Hermes Agent - Solution Architecture Support
Date: 2026-08-07
Status: Draft for Review

---

## Executive Summary

**EN:** Nyawa's Context Compaction feature provides a service that compresses long conversation transcripts into structured summaries while preserving retrievability. It is designed as a natural extension of the existing Nyawa memory engine: summaries are stored as first-class memories (`mem_type=summary`), linked into the entity graph, and original details are retained as archival (superseded) records that remain queryable via hybrid recall. The feature integrates with Hermes through four progressively deeper paths, from a zero-change memory-provider hook (Jalur A) to a full pluggable ContextEngine (Jalur C). Recommended delivery: Fase 1 = Jalur A + D (fast, safe, immediately usable), Fase 2 = Jalur C (full engine replacement, larger PR).

**ID:** Compression Hermes saat ini hanya memangkas 5-10% message dan rough_tokens justru naik setelah compress (bukti di agent.log). Context Compaction Nyawa menawarkan pendekatan industri: deterministic curation dulu (hapus tool output lama, pertahankan tail), summarization terstruktur sebagai last resort, dan semua jejak tersimpan di Nyawa sehingga bisa di-recall kapan pun. Design ini jadi dasar PR pengembangan Nyawa berikutnya (digabung dengan item SetMaxOpenConns yang sudah disepakati di-hold).

---

## Table of Contents

1. Background & Problem
2. Best Practices External
3. As-Is Architecture - Hermes
4. As-Is Architecture - Nyawa
5. Proposed Feature: Context Compaction
6. Hermes Integration Paths (4 Jalur)
7. Phased Recommendation
8. Technical Design Detail
9. Implementation Plan
10. Risk Assessment & Rollback
11. Monitoring & Observability
12. References

---

## 1. Background & Problem

### 1.1 Current Hermes Compression Behavior

Hermes (agent runtime, config `/opt/data/config.yaml`) has a built-in `ContextCompressor` engine (`context.engine: compressor`). Evidence from `/opt/data/logs/agent.log` shows the compression outcome is suboptimal:

| Session | Before | After | Delta |
|---------|--------|-------|-------|
| 20260727_222113_56a7d4fa | 473 msgs / ~164,933 tok | 448 msgs / ~170,659 tok | -25 msgs, tokens NAik +5,726 |
| 20260802_183307_e07d3eb1 (00:20) | 406 msgs / ~231,746 tok | 371 msgs / ~246,782 tok | -35 msgs, tokens naik +15,036 |
| 20260802_183307_e07d3eb1 (07:52) | 402 msgs / ~260,802 tok | 392 msgs / ~245,776 tok | -10 msgs, -15,026 tok |
| 20260802_183307_e07d3eb1 (08:47) | 406 msgs / ~220,524 tok | 402 msgs / ~247,641 tok | -4 msgs, tokens naik +27,117 |

All attempts logged `chunk_count: 0, chunking: false` - no hierarchical processing occurs.

### 1.2 Problem Statement

- Compression trims only 5-10% of messages; rough token estimates often INCREASE after compression (summary injection can exceed the tokens pruned)
- No structured segmentation or hierarchical summarization (`chunking: false`)
- Pruned content is discarded - no retrievable archive of what was removed
- No entity linking or long-term memory integration on the compacted content

### 1.3 Objective

- Reduce token count drastically (target: 50-70% reduction per compaction pass, aligning with industry compaction benchmarks [REF-01])
- Preserve critical information in a structured, continuation-ready summary
- Store every compaction result in Nyawa as retrievable memory (summary + archival)
- Enable Hermes to use Nyawa as its compaction backend via pluggable engine

---

## 2. Best Practices External

### 2.1 Claude Code 3-Tier Compaction

Claude Code does not use a single mechanism; it applies three tiers in sequence, each heavier than the last [REF-02] [REF-03]:

| Tier | Mechanism | Cost | Trigger |
|------|-----------|------|---------|
| Tier 1 | Microcompaction: clear old tool results, keep 5 most recent, replace rest with `[Old tool result content cleared]` | Free (deterministic, no LLM) | Before every API call |
| Tier 2 | API-level server-side strategies: thinking blocks, tool result clearing by token threshold | Free (server-side) | Token thresholds |
| Tier 3 | Full LLM summarization into structured 9-section summary (intent, technical concepts, files, errors+fixes, all user messages, pending tasks, current work) | Expensive + lossy | Context window pressure |

Key principle: **deterministic curation first, summarization as last resort**. Summary is the last-resort path because it is expensive and lossy.

### 2.2 Cache Economics

- Surgical `cache_edits` instead of message modification: when prompt cache is warm, old tool results are deleted by `tool_use_id` without invalidating the cached prefix [REF-03]
- The summarization call reuses the same system prompt, tools, model, and message prefix as the main conversation - same cache key, high hit rate
- Alternative (fresh summarization prompt) measured a 98% cache miss rate
- Design implication: compaction must avoid rebuilding the whole prompt from scratch

### 2.3 Post-Compaction Rehydration

After summarization, Claude Code rebuilds context methodically [REF-02]:

1. Boundary marker (marks compaction point with metadata)
2. Summary message (compressed working state)
3. Recent files - re-read 5 most recently accessed files (capped 50K tokens)
4. Todo list - preserve task state
5. Plan state (if in "plan required" workflow)
6. Hook outputs (startup hooks re-run)

Continuation message: "continue the conversation from where we left off without asking the user any further questions."

### 2.4 MemGPT Virtual Context Management

MemGPT borrows OS virtual memory paging: context window = RAM, external storage = disk, agent pages data in/out via function calls [REF-04]. Three-tier design:

- Working memory (current task state)
- Recall store (recent message history, searchable)
- Archival store (long-term searchable database)

### 2.5 Delta Summarization

Background/subagents use incremental updates instead of full reprocessing [REF-02]:

```text
You are given a few messages from a conversation, as well as a summary
of the conversation so far. Summarize the new messages based on the
summary so far. Aim for 1-2 sentences at most.
```

### 2.6 Key Takeaways for Nyawa

- Tiered approach: deterministic prune (free) before LLM summarization (expensive)
- Structured summary with continuation contract (not open-ended "summarize this")
- Everything discarded must remain retrievable (archival store pattern)
- Cache preservation matters (align with Hermes prompt-cache behavior)

---

## 3. As-Is Architecture - Hermes

### 3.1 Relevant Config

| Key | Value | Location |
|-----|-------|----------|
| `context.engine` | `compressor` (default built-in) | config.yaml:441 |
| `memory.provider` | `nyawa-local` (Nyawa already connected) | config.yaml:449 |
| `compression.enabled` | `true` | config.yaml:168 |
| `compression.threshold` | `0.75` | config.yaml:169 |
| `compression.target_ratio` | `0.2` | config.yaml:170 |
| `compression.protect_last_n` | `30` | config.yaml:171 |
| `compression.hygiene_hard_message_limit` | `400` | config.yaml:172 |
| `compression.protect_first_n` | `3` | config.yaml:173 |

### 3.2 ContextEngine ABC (`/opt/hermes/agent/context_engine.py:89`)

Pluggable abstract base class. Main contract (`compress`, line 163):

```python
def compress(
    self,
    messages: List[Dict[str, Any]],          # OpenAI-format message list
    current_tokens: Optional[int] = None,
    focus_topic: Optional[str] = None,       # from manual /compress <focus>
    force: bool = False,
    memory_context: str = "",                # insights from memory providers
) -> List[Dict[str, Any]]:
    """Return a (possibly shorter) valid OpenAI-format message sequence."""
```

Other hooks:

| Method | Purpose |
|--------|---------|
| `should_compress(prompt_tokens)` | Decide if compression is needed |
| `prune_tool_results_only(messages)` | Deterministic tool-result trim, no LLM (default no-op) |
| `select_context(messages)` | Per-turn context selection (orthogonal to compression) |
| `on_turn_complete(usage)` | Post-turn telemetry |
| `on_session_start(session_id, boundary_reason=..., old_session_id=...)` | Called after compression commit |
| `on_session_end`, `on_session_reset` | Session lifecycle |
| `get_tool_schemas`, `handle_tool_call` | Expose engine tools to agent |
| `get_status` | Status reporting |

### 3.3 Compression Flow (`/opt/hermes/agent/conversation_compression.py`)

1. `compress_context(agent, messages, system_message, ...)` (line 2128) is the entry point
2. Before summarization, memory providers are consulted: `memory_manager.on_pre_compress(messages)` (line 2745-2759) - provider returns a string of insights injected into the summary prompt (sanitized via `sanitize_memory_context`)
3. Engine's `compress()` produces the new message list
4. On durable commit, `_notify_context_engine_compression_complete` (line 2066) calls `on_session_start(new_session_id, boundary_reason="compression", old_session_id=..., platform=..., conversation_id=...)`
5. Session is rotated in SQLite (session split: old_session_id -> new_session_id)

### 3.4 Plugin Loading (`/opt/hermes/agent/agent_init.py:2360-2410`)

Selection order:

1. `config.yaml` `context.engine` setting
2. `plugins/context_engine/<name>/` directory (repo-shipped)
3. General plugin system (user-installed plugins, `get_plugin_context_engine`)
4. Fallback: built-in `ContextCompressor`

Switching engines = change one config value (`context.engine: nyawa`).

---

## 4. As-Is Architecture - Nyawa

### 4.1 Stack

- Go binary (live v1.1.6, dashboard container `ghcr.io/rezkyauliapratama/nyawa:1.1.6`)
- SQLite WAL + FTS5 (content table + triggers) + HNSW vector index (dim 384 local BGE embedder)
- Entity graph: 2,322 entity nodes, 9,816 edges (live query 2026-08-07)
- Dream cycle (1h interval): contradiction, dedup, link, priority, snapshot phases - runs in 8.5-11s (19 consecutive successes)
- MCP server + HTTP server (dashboard) + CLI

### 4.2 HTTP API (existing, `/opt/data/nyawa/internal/server/server.go:63-75`)

| Route | Method | Purpose |
|-------|--------|---------|
| `/v1/memories` | POST / GET | Store / list memories |
| `/v1/memories/` | GET / DELETE | By ID |
| `/v1/memories/batch` | POST | Batch store |
| `/v1/recall` | POST | Hybrid search |
| `/v1/stats` | GET | Stats |
| `/v1/health` | GET | Health check |
| `/v1/namespaces` | GET | List namespaces |
| `/v1/forget/` | DELETE | Soft-delete |
| `/v1/rag/collections` | GET / POST | List / create collections |
| `/v1/rag/collections/` | DELETE | By name |
| `/v1/rag/ingest` | POST | Ingest file |
| `/v1/rag/query` | POST | RAG query |
| `/v1/rag/stats` | GET | RAG stats |

### 4.3 Memory Schema (`/opt/data/nyawa/internal/store/sqlite.go:47-60`)

```sql
CREATE TABLE memories (
  id TEXT PRIMARY KEY,
  content TEXT NOT NULL,
  mem_type TEXT NOT NULL DEFAULT 'note',
  namespace TEXT NOT NULL DEFAULT 'default',
  importance REAL NOT NULL DEFAULT 0.4,
  access_count INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  superseded_at TEXT,
  edge_count INTEGER NOT NULL DEFAULT 0
);
```

Plus `memories_fts` (FTS5), triggers for insert/delete/update, `entity_nodes` + `entity_edges` tables.

### 4.4 Existing Dream Cycle Phases

| Phase | Function | Behavior |
|-------|----------|----------|
| Contradiction | `phaseContradiction` | Find contradictions |
| Dedup | `phaseDedup` (line 153) | Word-overlap dedup, supersede older |
| Link | `phaseLink` (line 199) | Co-occurrence entity linking (collect-then-update pattern, fixed in v1.1.6) |
| Priority | `phasePriority` (line 218) | Importance boost/decay |
| Snapshot | `phaseSnapshot` (line 229) | Consolidate old memories into `snap_*` snapshot memory |

`phaseSnapshot` is the existing precedent: consolidating old content into a summary-like memory with supersede. Context Compaction extends this pattern to conversation transcripts.

---

## 5. Proposed Feature: Context Compaction

### 5.1 Concept

Input: conversation transcript (array of OpenAI-format messages, same shape Hermes uses).

Pipeline:

1. **Segment**: split transcript into segments of N messages (default ~40), with overlap of ~5 for continuity
2. **Deterministic prune (Tier 1)**: drop old tool-result payloads above a size threshold, keep hot tail (last `preserve_recent_n`, default 30) verbatim; replace pruned tool results with `[tool result stored in nyawa:<memory_id>]`
3. **Summarize (Tier 3)**: each non-tail segment is summarized via LLM into a structured block (intent, decisions, files/actions, errors+fixes, pending) - prompt is continuation-contract based [REF-02]
4. **Store**: summary blocks saved as `mem_type=summary`, `namespace=context`, linked into entity graph (entities extracted via existing classifier v2)
5. **Archive**: original segment content stored as archival memories (superseded by the summary) - still recallable
6. **Output**: compact message list = boundary marker + summary block + recent tail verbatim + continuation instruction

### 5.2 API Design (proposed)

```http
POST /v1/compact
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "..."}, ...],
  "focus_topic": "optional focus string",
  "preserve_recent_n": 30,
  "segment_size": 40,
  "segment_overlap": 5,
  "namespace": "context",
  "model": "optional LLM override (default: configured)"
}
```

Response:

```json
{
  "summary_block": [
    {"role": "system", "content": "COMPACTION_BOUNDARY ..."},
    {"role": "system", "content": "<structured summary>"},
    {"role": "user", "content": "Continue without re-asking..."}
  ],
  "memory_ids": ["summary_...", "arch_..."],
  "stats": {
    "input_messages": 473,
    "output_messages": 8,
    "input_tokens": 164933,
    "output_tokens": 41000,
    "reduction_pct": 75.1,
    "segments": 11,
    "pruned_tool_results": 34
  }
}
```

### 5.3 Key Design Decisions

- **D-1**: Summary + archival both stored in Nyawa - nothing is lost, everything recallable (MemGPT archival-store pattern [REF-04])
- **D-2**: Deterministic Tier-1 prune runs BEFORE any LLM call - free wins first [REF-03]
- **D-3**: Output MUST be a valid OpenAI-format message sequence (Hermes contract, `context_engine.py:compress`)
- **D-4**: Session-split compatibility: Nyawa stores `old_session_id`/`new_session_id` mapping so history is traceable across Hermes rotation
- **D-5**: LLM summarization delegated to configured provider (Hermes DeepSeek via aux model, or Nyawa's own provider config) - Nyawa stays offline-first for storage/search

---

## 6. Hermes Integration Paths (4 Jalur)

### 6.1 Jalur A: Memory Provider `on_pre_compress` (zero Hermes change)

Nyawa is already `memory.provider: nyawa-local`. The provider implements `on_pre_compress(messages)` (memory_provider.py:258) which returns a string of insights injected into the compression summary.

- Nyawa recalls relevant long-term context (from entity graph / RAG) matching the current transcript
- Returns insights string; Hermes injects into summary prompt (conversation_compression.py:2745-2759)
- Effect: Hermes summaries become "smarter" (grounded in Nyawa long-term memory)

| Pros | Cons |
|------|------|
| 0 Hermes code change | Does not replace compression itself |
| Ships immediately | Insight injection only |

### 6.2 Jalur B: MCP Tool `compact_context` (agent-driven)

Nyawa exposes `compact_context` as an MCP tool (add to `internal/mcp/`). Agent Hermes calls it manually when context is long (or via skill/cron).

- Nyawa executes the full compaction pipeline
- Returns compact block + memory IDs; agent replaces its message list

| Pros | Cons |
|------|------|
| No Hermes core change | Requires agent discipline |
| Full pipeline available now | Manual trigger, not automatic |

### 6.3 Jalur C: ContextEngine Plugin (full integration)

Implement `plugins/context_engine/nyawa/` in the personal-ai-stack repo, subclassing `ContextEngine` ABC. Override `compress()` to call Nyawa `/v1/compact`. Config: `context.engine: nyawa`.

Flow: threshold hit -> Hermes calls `nyawa.compress(messages)` -> Nyawa compacts -> returns valid message list -> Hermes splits session in SQLite as usual. Hermes does not care who compresses; contract is identical.

| Pros | Cons |
|------|------|
| Full replacement of summarizer | Larger PR (plugin + Nyawa API + config) |
| Automatic at threshold 0.75 | Needs Hermes restart to activate |
| Native session-split integration | Plugin lives in personal-ai-stack repo |

### 6.4 Jalur D: Post-Compression Hook (required complement, all options)

Use `on_session_start(boundary_reason="compression", old_session_id, new_session_id)` (conversation_compression.py:2066): every time Hermes finishes a compression, Nyawa stores `(old_session_id, summary, new_session_id, pointer)` as `mem_type=summary` in its DB.

- Gives Nyawa a complete trail of every Hermes compression
- Enables recall: "summary <session_id>" returns the compaction history
- Works regardless of which Jalur (A/B/C) is active

---

## 7. Phased Recommendation

| Phase | Scope | Effort | Value |
|-------|-------|--------|-------|
| Fase 1 | Jalur A + D (on_pre_compress insights + post-compression archiving) | Small | Immediate: smarter summaries + full compression trail |
| Fase 2 | Nyawa `/v1/compact` API + MCP tool (Jalur B) | Medium | Full compaction pipeline, agent-driven |
| Fase 3 | ContextEngine plugin (Jalur C) | Large | Fully automatic compaction via Nyawa |

Fase 1 can ship without any Nyawa binary change (only memory provider plugin in personal-ai-stack). Fase 2/3 need Nyawa Go code + PR (can be combined with the agreed SetMaxOpenConns change in `sqlite.go:35`).

---

## 8. Technical Design Detail

### 8.1 Summary Block Structure (OpenAI-format)

```json
[
  {"role": "system", "content": "=== CONTEXT COMPACTION BOUNDARY ===\nsource_session: 20260727_222113_56a7d4fa\ncompaction_time: 2026-08-07T11:00:00Z\nfull_archive: nyawa://recall?q=summary+20260727_222113_56a7d4fa"},
  {"role": "system", "content": "## COMPACTED WORKING STATE\n### User Intent\n...\n### Key Decisions\n...\n### Files / Actions\n...\n### Errors & Fixes\n...\n### Pending Tasks\n..."},
  {"role": "user", "content": "This session continues from a compacted conversation. Continue the last task without re-asking questions."}
]
```

### 8.2 Segmentation

- Default `segment_size: 40` messages, `segment_overlap: 5` (boundary context preserved)
- Tail (last `preserve_recent_n: 30`) kept verbatim - never summarized
- Tool-result payloads > 8KB pruned in Tier 1, replaced with pointer

### 8.3 Storage

- Summary: `mem_type=summary`, `namespace=context`, `importance=0.7`, `pinned=0`
- Archival segments: `mem_type=archive`, superseded by summary ID (existing `superseded_at` mechanism)
- Session mapping: new table `compaction_log(session_key, old_session_id, new_session_id, summary_id, created_at)`
- Entity linking: existing `InsertMemoryEntities` (classifier v2) extracts entities from summary + archive -> graph grows naturally

### 8.4 Recall Back

- `nyawa_recall("summary <session_id>")` returns compaction history
- `nyawa_recall("<focus_topic>")` returns relevant archived detail via hybrid search
- Graph query: `nyawa_graph_query` from summary entities to related memories

---

## 9. Implementation Plan

| Step | Task | Repo | Depends on |
|------|------|------|------------|
| 1 | `compaction_log` table + migration | nyawa | - |
| 2 | Tier-1 deterministic prune utility | nyawa | - |
| 3 | Segmenter (size/overlap/tail) | nyawa | - |
| 4 | Summarization call (provider-agnostic) | nyawa | 3 |
| 5 | `/v1/compact` handler + store integration | nyawa | 1,2,3,4 |
| 6 | MCP tool `compact_context` | nyawa | 5 |
| 7 | Memory provider `on_pre_compress` + `on_session_start` (Jalur A+D) | personal-ai-stack | 5 (optional) |
| 8 | ContextEngine plugin `plugins/context_engine/nyawa/` (Jalur C) | personal-ai-stack | 5 |
| 9 | Config `context.engine: nyawa` + restart | personal-ai-stack | 8 |
| 10 | Combine SetMaxOpenConns (sqlite.go:35) in same dev PR | nyawa | - |

Release chain follows existing pattern: PR -> merge -> tag -> GHCR image -> compose bump -> swap binary.

---

## 10. Risk Assessment & Rollback

| Risk | Severity | Mitigation |
|------|----------|------------|
| Summary loses critical detail | Medium | Focus topic + 9-section checklist contract + archive retains full detail |
| Compaction output breaks Hermes message format | High | Strict OpenAI-format validation; dry-run mode returns stats without applying |
| LLM summarization cost/latency | Medium | Tier-1 deterministic prune first; summarize per segment not whole transcript; rehydrate tail verbatim |
| Cache invalidation on Hermes side | Medium | Boundary marker keeps prefix stable; test cache-hit ratio before/after [REF-03] |
| Entity graph pollution from archives | Low | `mem_type=archive` low importance; dream cycle dedup handles overlap |
| Deadlock regression in new DB ops | High | Follow v1.1.6 lesson: collect-then-update pattern, never Exec inside `rows.Next()` [REF-05] |

Rollback: Fase 1 is config-only (revert `context.engine` to `compressor`). Fase 2/3 rollback = revert binary to `.bak-v115` (kept in `/opt/data/.nyawa/`) + compose image pin.

---

## 11. Monitoring & Observability

- `stats.reduction_pct` per compaction (target 50-70% [REF-01])
- `compaction_log` row count growth (expected: 1 per Hermes compression)
- Dream cycle duration (must stay < 15s with new compaction memories)
- WAL size trend (compaction writes increase write volume; watch checkpoint)
- Hermes log: `rough_tokens` before/after must trend DOWN (fixes the current inversion)
- Dashboard: add `/v1/stats` section for compaction counters

---

## 12. References

| ID | Source |
|----|--------|
| [REF-01] | Morph - Context Compaction: 50-70% compression with 98% verbatim accuracy (morphllm.com/context-compaction) |
| [REF-02] | Decode Claude - Inside Claude Code's Compaction System (decodeclaude.com/compaction-deep-dive) |
| [REF-03] | Barazany - Claude Code's Compaction Engine: source code analysis (barazany.dev/blog/claude-codes-compaction-engine) |
| [REF-04] | MemGPT: Towards LLMs as Operating Systems, Packer et al., UC Berkeley, arXiv:2310.08560 |
| [REF-05] | Nyawa v1.1.6 session resume - phaseLink deadlock lesson (session_resume_nyawa_v116.md) |
| [REF-06] | Anthropic - Compaction (platform.claude.com/docs/en/build-with-claude/compaction) |

---

*End of Document - NYAWA-ARCH-001 v0.1*

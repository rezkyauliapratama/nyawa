# Memory Conventions — Panduan Nulis Memory Biar Graph Rakin

Nyawa v1.1.1+ punya **entity extraction v2**: tech dictionary case-insensitive
+ alias normalization + auto re-extract tiap Dream Cycle. Tapi extraction
tetap butuh input yang bagus. Ini konvensi yang bikin graph makin kaya.

## Kenapa penting

- Graph node dibuat dari entity yang **terekstrak dari content memory**
- Extraction pakai regex + dictionary (bukan LLM) — deterministic, offline
- Entity yang disebut konsisten → masuk graph → bisa di-traverse via `nyawa_graph_query`
- Entity yang ditulis asal → terlewat atau jadi node duplikat

## Aturan 1: Sebut nama persis (canonical)

Pakai nama resmi biar alias-nya ter-normalisasi ke 1 node:

```text
✅ "Rezky deploy ke GCP pakai Terraform"
✅ "model: DeepSeek via AWS Bedrock di ap-southeast-3"
❌ "deps pake google cloud ama tf nya"
```

Alias yang SUDAH otomatis dinormalisasi (v1.1.1):
- `gcp` / `google cloud` / `google cloud platform` → `GCP`
- `aws` / `amazon web services` → `AWS`
- `deepseek` → `DeepSeek` (case-insensitive)
- `bedrock` / `amazon bedrock` → `AWS Bedrock`
- `k8s` → `Kubernetes`
- `golang` → `Go`
- `postgres` → `PostgreSQL`
- `mcp` / `model context protocol` → `MCP`

## Aturan 2: Satu memori, satu topik

Memory yang fokus → entity yang muncul = konteks yang relevan:

```text
✅ "Arsitektur V3: crypto-engine (Go) pakai Kafka untuk streaming"
❌ "hari ini rapat, terus benerin bug, oh iya Kafka dipakai juga, besok deploy GCP"
```

Memory campur baur bikin co-occurrence edge jadi noise (entity gak terkait
jadi dianggap "related").

## Aturan 3: Sebut relasi eksplisit untuk typed edges

Typed edges (`works_at`, `uses`, `located_in`, `part_of`) dibuat dari pola
tertentu — sebut relasinya dengan kata kunci:

```text
✅ works_at:   "Rezky bekerja di Bank Sinarmas"   / "Rezky works at Bank Sinarmas"
✅ uses:       "tim pakai Kafka untuk streaming"  / "team uses Kafka for streaming"
✅ located_in: "server berlokasi di Jakarta"      / "server located in Jakarta"
✅ part_of:    "Kafka bagian dari MCP"            / "Kafka part of MCP"
```

## Aturan 4: Jangan ragu repeat — dedup otomatis

Tulis ulang fakta yang sama di memori berbeda itu OK:
- Dream Cycle dedup (threshold 0.92) bakal merge yang duplikat literal
- Entity yang muncul di banyak memory → weight edge naik → makin relevan di recall

## Flow otomatis (v1.1.1)

```text
store memory → extract entities (dict v2) → insert nodes/edges
     ↓
Dream Cycle tiap 1 jam:
  1. ReextractEntities()  → backfill entity yang terlewat di memories lama
  2. RebuildGraph()       → rebuild co-occurrence edges + prune stale
  3. InferTypedEdges()    → typed edges baru dari relasi eksplisit
```

Artinya: **memories lama pun ikut diperkaya** tiap cycle — gak perlu
re-store manual. Update dictionary di release berikutnya langsung
backfill otomatis juga.

## Checklist sebelum store memory

- [ ] Nama entity pakai canonical (cek Aturan 1)
- [ ] Satu memori = satu topik
- [ ] Kalau ada relasi, sebut kata kunci (Aturan 3)
- [ ] Bahasa bebas (ID/EN) — dictionary & typed edges bilingual

# Scale Projection — Nyawa Phase 1a
## Estimasi Resource di Berbagai Scale

### Asumsi Dasar
- Rata-rata memory: ~500 bytes content + ~200 bytes metadata + 384×4 bytes vector = ~2.2KB/memory
- HNSW index overhead: ~1.2× data size (typical for HNSW with efConstruction=200, M=16)
- SQLite overhead: ~10% (pages, indexes, WAL)
- Ollama/API embedder: RAM diluar process (external HTTP)
- BGE-small GGUF: ~30MB model + ~25MB runtime (CGO, in-process)

### Proyeksi per Scale

| Scale | Memories | Data Size | HNSW Index | SQLite Overhead | Total Disk | RAM (BGE) | RAM (Ollama) | RAM (OpenAI) |
|-------|----------|-----------|-------------|-----------------|------------|-----------|--------------|--------------|
| Dev | 1K | ~2MB | ~2.4MB | ~0.2MB | ~4.6MB | ~55MB | ~20MB | ~20MB |
| MVP | 10K | ~22MB | ~26MB | ~2MB | ~50MB | ~55MB | ~20MB | ~20MB |
| Production | 100K | ~220MB | ~264MB | ~22MB | ~506MB | ~55MB* | ~20MB | ~20MB |
| Enterprise | 1M | ~2.2GB | ~2.6GB | ~220MB | ~5GB | ~55MB* | ~20MB | ~20MB |

*\*RAM BGE-small tetap ~55MB karena model size fixed — bukan fungsi dari jumlah memories.*

### Critical Constraint: HNSW Memory

HNSW harus fully in-memory untuk performa optimal.

| Scale | HNSW Size | In-Memory? | Impact |
|-------|-----------|------------|--------|
| ≤10K | ≤26MB | ✅ Yes | Latency 1-3ms |
| 100K | ~264MB | ✅ Possible (total ~320MB) | Latency 5-15ms |
| 1M | ~2.6GB | ⚠️ Large (but feasible on 4GB+ VM) | Latency 20-50ms |
| 10M | ~26GB | 🔴 Not feasible | Must use disk-based index (ANN) |

### Rekomendasi Arsitektur per Scale

**Scale ≤10K (MVP — Phase 1a/1b)**
- Single binary, single process
- BGE-small GGUF in-process
- HNSW fully in-memory
- Total RAM: ~55-80MB
- ✅ No changes needed

**Scale 100K (Production — Phase 2b/3)**
- Single process, but Dream Cycle process isolation recommended
- HNSW ~264MB — still feasible in-memory (total ~320MB + model)
- SQLite WAL mode with checkpoint tuning
- Total RAM: ~200-350MB
- ⚠️ Monitor GC pressure from HNSW allocations

**Scale 1M+ (Enterprise — Phase 4)**
- HNSW ~2.6GB — needs dedicated allocation
- Consider disk-based ANN (e.g., SQLite VSS extension instead of HNSW)
- Or: tiered storage (hot: HNSW in-memory, warm: SQLite VSS disk-based)
- Dream Cycle must be process-isolated (Layer 2 mitigation)
- Run on 4GB+ VM with GOMEMLIMIT=3GB
- 🔴 Significant re-architecture may be needed

### GC Pressure Estimation

| Scale | Allocation Rate | GC Frequency | GC Pause (Go 1.23) | Impact on p99 |
|-------|----------------|-------------|-------------------|---------------|
| 10K | ~50KB/recall | Every 5-10min | ~1ms | Negligible |
| 100K | ~200KB/recall | Every 2-5min | ~2-3ms | Low (with memory pool) |
| 1M | ~1MB/recall | Every 1-2min | ~5-10ms | ⚠️ Medium (needs process isolation) |

### Kesimpulan

1. **Phase 1a MVP (≤10K):** Tidak ada masalah skalabilitas. Go GC tidak signifikan.
2. **Phase 3 Production (100K):** Monitor. Memory pool + adaptive scheduling cukup.
3. **Phase 4 Enterprise (1M+):** Process isolation required. HNSW memory-bound jadi bottleneck utama.
4. **Kapan harus action:** Saat total memories > 500K atau recall QPS > 100.

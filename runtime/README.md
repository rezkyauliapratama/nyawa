# Durable Nyawa BGE runtime (Phase C code-only)

Runtime ini memakai satu proses Nyawa dan satu child `bge_server.py`. Child
memuat model BGE ONNX sekali, membaca JSONL dari `stdin`, menulis JSONL ke
`stdout`, dan hanya mengirim marker `READY dim=...` ke `stderr`. Tidak ada Unix
socket, pencarian virtualenv host secara implisit, probe Python palsu, atau model
yang diambil saat startup.

## Kontrak provenance dan artefak

- Build ini adalah **source build** dari checkout ini dengan builder Go; bukan
  turunan dari image Nyawa yang sudah dipublikasikan.
- Digest Nyawa v1.2.4 yang disetujui,
  `sha256:6a2830f831b8aa8bcc6c82ed07b5ccbfd82882aa98c0bfb176e7a97a498b1ec0`,
  hanya baseline reference dan direkam sebagai OCI label. Ia bukan stage
  Docker yang tidak terpakai dan bukan parent image patched binary.
- Python lock hanya memuat import runtime langsung dari `bge_server.py`:
  `numpy`, `onnxruntime`, dan `tokenizers`, semuanya hash-pinned. Docker
  menginstal wheel tersebut dengan `--no-deps` karena client Hub/transitive
  packages tidak dipakai oleh server embedded.
- Model asset SHA-256: `model.onnx` = `66fc00f5f29afcaff34092e1bdd20008ca3918265a82fb9695a551e510cc4ebc`;
  `tokenizer.json` = `b60b6b43406a48bf3638526314f3d232d97058bc93472ff2de930d43686fa441`.
- `model.onnx` berukuran lebih dari batas file GitHub biasa, sehingga disimpan
  sebagai asset release `bge-model-v1.2.4`; workflow publish mengunduh dan
  memverifikasi checksum sebelum build. `tokenizer.json` tetap berada di repo.
- `NYAWA_EMBED_MODEL_DIR` menunjuk ke `/opt/nyawa/model` di dalam image.
- Secara default, Nyawa tetap mencoba interpreter image dalam urutan
  `/usr/local/bin/python3`, `/usr/bin/python3`, lalu `python3` dari `PATH`.
  `NYAWA_EMBED_PYTHON` adalah override opt-in untuk kebutuhan runtime khusus:
  nilainya harus berupa path absolut ke executable Python yang dapat mengimpor
  `numpy` dan `onnxruntime`. Nyawa menjalankan probe import tanpa shell sebelum
  memakai interpreter tersebut. Nilai override yang relatif, kosong, tidak
  executable, atau gagal mengimpor dependensi membuat startup gagal; tidak ada
  fallback ke kandidat default bila override sudah disetel.

## Local cross-encoder confidence (opt-in)

The optional `Horizon-Labs/multilingual-reranker-small` ONNX model is packaged only when the publish workflow is dispatched with `include_reranker=true`. The build pins revision `23139fec6f1f448cdf89a5ac2d2c6bf5e8c7b398` and verifies the SHA-256 of `onnx/model_quantized.onnx` (`f913154682468bfe92d6a7e02ed4e994e548c112fddc74452b573b17c949046e`) and `tokenizer.json` (`7925f638b5444897efe3b4608eed56c1652f0ed39c2662c8302de1bc264c6088`). The upstream model is Apache-2.0 and supports 40+ languages.

`NYAWA_ENABLE_CROSS_ENCODER` defaults off. When explicitly enabled, the local ONNX scorer returns a candidate confidence logit as additive recall telemetry; it does not reorder results, modify RRF, or call a provider. Inference is offline and limited to the final recall candidates. Runtime requirements remain the existing hash-pinned `numpy`, `onnxruntime`, and `tokenizers` packages.

## Validasi code-only

```bash
python3 -m py_compile internal/embedder/bge_server.py scripts/validate-bge-runtime.py scripts/test-bge-handshake.py
python3 scripts/validate-bge-runtime.py
python3 scripts/test-bge-handshake.py
gofmt -d internal/embedder/py_embedder.go internal/embedder/py_embedder_socket_test.go internal/embedder/modeldir_test.go
git diff --check
```

`test-bge-handshake.py` menjalankan child Python nyata dengan model dan modul
palsu di direktori temporary; ia mengirim satu request JSONL `embed` melalui
stdin dan memvalidasi response JSONL melalui stdout. Test ini tidak menyentuh
DB, Docker, socket, virtualenv host, atau resource production.

## Slow apply terpisah (perlu approval baru)

1. Review diff, static validation, dan hasil handshake.
2. Di non-production, jalankan tepat satu `docker build` dengan context
   repository ini. `docker run` hanya boleh dilakukan bila smoke terpisah telah
   disetujui, memakai DB salinan tanpa mount socket atau virtualenv host.
3. Untuk production, backup DB/HNSW, freeze write window, deploy image digest
   hasil build, start hanya service Nyawa, lalu jalankan smoke read/write yang
   dibatasi. Gateway tidak direstart sebagai bagian dari build gate.
4. Rollback: hentikan image baru pada service Nyawa, kembalikan image digest
   sebelumnya, restore DB/HNSW hanya bila smoke menunjukkan write corruption,
   lalu verifikasi stats dan recall. Jangan menghapus volume.

Semua langkah apply production sengaja tidak termasuk dalam Phase C code-only
ini.

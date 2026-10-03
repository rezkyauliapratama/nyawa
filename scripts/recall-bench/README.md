# recall-bench

Harness benchmark **recall** Nyawa. Menjalankan daftar kueri tetap ke sebuah
binary `nyawa recall`, mem-parsing keluarannya, lalu menghitung **recall@1 /
recall@3 / recall@5** dan **MRR**. Sebagai pembanding leksikal, harness juga
menjalankan pencarian **BM25 FTS5** langsung ke DB (read-only).

Harness **tidak pernah menulis** ke DB (koneksi `mode=ro`) dan **tidak
menyentuh** binary/kode mesin recall. Ini murni alat ukur.

## Berkas

| Berkas | Isi |
|---|---|
| `bench.py` | runner + penghitung metrik; menulis JSON + ringkasan markdown |
| `queries.tsv` | daftar kueri dan id memori target (1 kueri per baris) |
| `ftsprobe/` | helper Go: mereproduksi leg BM25 FTS5 persis seperti mesin (driver runtime) |
| `BASELINE-*.md` / `baseline-*.json` | hasil yang dihasilkan (lihat juga riwayat git) |
| `BASELINE-NOTES.md` | analisis/temuan baseline (disisipkan ke markdown via `--notes-file`) |
| `README.md` | dokumen ini |

## Cara pakai

```bash
# baseline terhadap binary runtime (default)
python3 scripts/recall-bench/bench.py

# binary lain, dengan label agar berkas hasil tidak saling menimpa
python3 scripts/recall-bench/bench.py \
    --binary /opt/data/.nyawa/nyawa-kandidat \
    --label kandidat

# DB/kueri lain
python3 scripts/recall-bench/bench.py \
    --db /opt/data/.nyawa/memory.db \
    --queries scripts/recall-bench/queries.tsv \
    --out-dir /tmp/hasil-bench
```

Argumen penting:

| Argumen | Default | Keterangan |
|---|---|---|
| `--binary` | `/opt/data/.nyawa/nyawa` | binary `nyawa` yang diuji |
| `--db` | `/opt/data/.nyawa/memory.db` | DB SQLite (dibuka read-only) |
| `--queries` | `queries.tsv` di folder skrip | daftar kueri |
| `--out-dir` | folder skrip | lokasi output |
| `--label` | `runtime` | sufiks nama berkas & penanda di JSON |
| `--ns` | kosong | namespace default bila kolom ke-3 tsv kosong |
| `--timeout` | 120 | timeout per panggilan recall (detik) |
| `--fts-limit` | 100 | top-K pencarian BM25 FTS (mode python) |
| `--engine-fts-json` | — | keluaran `ftsprobe` untuk menggabungkan leg FTS mentah **mesin** |
| `--notes-file` | — | markdown tambahan untuk disisipkan ke bagian Catatan |

### Leg FTS mentah mesin (ftsprobe)

Mesin mengoper teks kueri mentah ke `memories_fts MATCH ?` (tanpa escape).
Karena parsing FTS5 bergantung pada build SQLite, harness menyediakan helper Go
`ftsprobe` yang memakai driver dan SQL yang **sama** dengan runtime:

```bash
# dari root repo (butuh toolchain Go + tag sqlite_fts5)
go run -tags sqlite_fts5 ./scripts/recall-bench/ftsprobe \
    scripts/recall-bench/queries.tsv > /tmp/fts-engine.json

python3 scripts/recall-bench/bench.py \
    --engine-fts-json /tmp/fts-engine.json \
    --notes-file scripts/recall-bench/BASELINE-NOTES.md
```

Bila `--engine-fts-json` diberikan, kolom & agregat "FTS mesin (raw)" muncul di
laporan. Perlu dicatat: build SQLite Python dan runtime bisa berbeda — build
Python (3.53.4) melempar error untuk query bertanda hubung/titik, sedangkan
build runtime (3.53.3) mengembalikan **nol hasil tanpa error**. Karena itu
keduanya dilaporkan terpisah.

### Membandingkan dua binary

Jalankan dua kali dengan label berbeda lalu bandingkan berkas JSON/markdown-nya:

```bash
python3 scripts/recall-bench/bench.py --label runtime            # -> baseline-<tgl>-runtime.md
python3 scripts/recall-bench/bench.py --binary ./nyawa-baru --label kandidat
diff <(jq '.aggregate' baseline-*.json) ...
```

Urutan `--label` juga menentukan nama berkas: `--label runtime` menulis
`BASELINE-<tanggal>.md`, label lain menulis `BASELINE-<tanggal>-<label>.md`.

## Format `queries.tsv`

```
<query>\t<target_id_1>,<target_id_2>\t<namespace opsional>
```

- Kolom 1: teks kueri apa adanya yang dikirim ke `nyawa recall`.
- Kolom 2: satu atau lebih id memori yang diharapkan relevan (dipisah koma).
  Sebuah kueri dianggap "hit@k" bila **minimal satu** target muncul di top-k;
  MRR memakai peringkat target teratas yang ditemukan.
- Kolom 3 (opsional): namespace; kosong berarti tanpa filter namespace.

Baris diawali `#` adalah komentar.

## Metrik

- **recall@k** — proporsi kueri yang punya minimal satu target di peringkat ≤ k.
- **MRR** — rata-rata `1/peringkat` target teratas (0 bila tak ditemukan).
- Pembanding BM25 dijalankan tiga mode:
  - **raw** — persis seperti mesin: query mentah dioper ke `memories_fts MATCH ?`
    `ORDER BY rank`. Bisa **gagal** bila query memuat sintaks FTS5 (mis. tanda
    hubung, yang dibaca sebagai operator).
  - **or** — token query digabung dengan `OR` (paling longgar).
  - **and** — token query digabung dengan `AND` (paling ketat).

  Ketiganya menyaring `superseded_at IS NULL`, sama seperti mesin.

## Cara kerja & batasan

- **Pencocokan target berbasis isi.** Keluaran `nyawa recall` hanya memuat
  peringkat, skor, dan isi — **tanpa id memori**. Harness memetakan hasil ke id
  dengan mencocokkan isi hasil terhadap isi memori di DB. Bila dua memori punya
  isi identik (duplikat), pencocokan bisa ambigu; harness mencatat id pertama.
  Saat ini DB memuat banyak isi duplikat, jadi peringkat per-target bisa
  ambigu untuk memori nondescript — namun target benchmark yang dipakai unik.
- `nyawa recall` selalu mengembalikan maksimal **10** hasil, sehingga peringkat
  > 10 selalu dilaporkan "TIDAK KETEMU".
- Target yang sudah **superseded** tidak akan pernah muncul di recall aktif
  (filter `superseded_at IS NULL`); harness tetap menandainya di catatan.
- Byte non-UTF8 pada sebagian memori didekode dengan `errors="replace"` di kedua
  sisi (output CLI dan isi DB) agar pencocokan konsisten.

## Menjalankan ulang baseline

Baseline terakhir ada di `BASELINE-<tanggal>.md` + `baseline-<tanggal>.json`.
Untuk mereproduksi: pastikan binary runtime dan DB pada kondisi yang sama,
lalu `python3 scripts/recall-bench/bench.py`. Hash binary dan mtime DB
tercantum di bagian meta tiap hasil.

### Ringkasan temuan

**Angka utama (baseline, binary runtime v1.2.0):**

| Jalur | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| recall CLI (vector + RRF) | 0.333 | 0.417 | 0.500 | 0.392 |
| BM25 FTS token-OR | 0.833 | 1.000 | 1.000 | 0.903 |
| BM25 FTS token-AND | 0.500 | 0.667 | 0.667 | 0.569 |
| BM25 FTS mentah (mesin) | 0.333 | 0.500 | 0.500 | 0.403 |

**Temuan 1 — recall CLI meleset tepat pada kueri yang leg FTS mentahnya kosong.**
Enam kueri yang gagal di recall (1, 5, 7, 9, 10, 12) persis sama dengan enam
kueri yang leg FTS mentah mesin mengembalikan **nol baris**. Enam kueri lain
(2, 3, 4, 6, 8, 11) berhasil di recall, dan semuanya leg FTS-nya berisi hasil.
Korelasi 12/12 ini menunjuk ke satu arah: pada kondisi sekarang sinyal recall
praktis hanya datang dari leg FTS (mentah), dan begitu leg itu kosong, vector
tidak menyelamatkan.

**Temuan 2 — leg FTS mentah tidak di-escape, jadi diam-diam kosong.**
Mesin mengoper teks kueri mentah ke `memories_fts MATCH ?` (`internal/store/sqlite.go`
`FTS5Search`, tanpa sanitasi). Kueri yang memuat `-` atau `.` memecah sintaks
FTS5: `crypto-data` dan `bersih-bersih` dibaca sebagai operator, `SKILL.md`
sebagai sintaks tak valid. Hasilnya di runtime (driver SQLite 3.53.3 via
mattn/go-sqlite3) adalah **nol hasil tanpa error**, sehingga RRF hanya menerima
urutan vector. Justru kueri bertanda hubung/titik inilah yang paling butuh
bantuan leksikal.

**Temuan 3 — beda build SQLite mengubah gejalanya.**
Build SQLite Python (3.53.4) melempar `OperationalError` untuk query yang sama,
sedangkan build runtime (3.53.3, mattn/go-sqlite3) mengembalikannya kosong tanpa
error. Karena itu leg mentah tidak boleh dinilai sebagai "error" saja —
di runtime ia gagal secara senyap. Harness karena itu menyimpan dua ukuran:
`FTS(raw, python)` dan `FTS mesin (raw)` (dari `ftsprobe`, driver runtime).

**Temuan 4 — secara leksikal semua target mudah dijangkau.**
Dengan tokenisasi sederhana digabung `OR`, target muncul di **peringkat 1 untuk
seluruh 12 kueri** (MRR 0.903). Artinya masalahnya bukan "memori target tidak
ada / tidak terindeks teks", melainkan cara query recall dibentuk dan
digabungkan. Perbaikan pada pembentukan query leksikal (escape/tokenize) punya
ruang perbaikan besar dan terukur.

**Temuan 5 — satu target sudah superseded.**
`mem_1790421531821504368` (aturan docker larangan `docker volume prune`) memiliki
`superseded_at = 2026-10-03T09:35:01Z`, jadi mustahil muncul di recall aktif
(filter `superseded_at IS NULL`). Kueri docker (1) menyisakan satu target aktif
`mem_1790421515511059827`; leg FTS token-OR menaruhnya di peringkat 1, tapi
recall CLI tidak menemukannya sama sekali.

### Keraguan dan batasan

- **Pencocokan target berbasis isi.** Keluaran `nyawa recall` tidak memuat id
  memori, jadi hasil dicocokkan ke id lewat kesamaan isi. DB memuat 1.531 isi
  duplikat; untuk target benchmark ini isinya unik sehingga aman, tetapi
  pencocokan by-id langsung tidak tersedia dari CLI.
- **`recall` hanya 10 hasil**, jadi peringkat > 10 selalu "TIDAK KETEMU".
- **Perbandingan FTS token-OR/AND memakai tokenisasi sederhana** (regex alnum,
  lowercase) lalu mengandalkan tokenizer FTS5 `porter unicode61`. Ini meniru
  semangat pencarian leksikal, bukan persis pre-processing yang akan dipakai
  perbaikan nanti.
- **ftsprobe memakai limit 50** (sama dengan `searchTopK` mesin untuk limit 10);
  peringkat di luar 50 dilaporkan "NOT".
- Baseline diukur pada DB yang hidup (WAL) dan terus berubah; hash binary dan
  mtime DB dicantumkan di bagian meta JSON/MD agar perbandingan antar-run adil.
- Satu kueri tambahan (12) targetnya justru memuat kalimat yang mengeluh hasil
  `nyawa_recall` meleset — kebetulan yang terdokumentasi, bukan disengaja.

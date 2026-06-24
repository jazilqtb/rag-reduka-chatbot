# Engineering Journal

Catatan proses development — bukan dokumentasi formal. Format storytelling
per milestone, fokus ke *kenapa* dan *trade-off apa yang diterima*.

> **Format ADR (Architecture Decision Record) per keputusan lengkap ada di**
> `docs/decisions/`. Journal ini cerita yang lebih naratif tentang perjalanan
> development.

---

## Konteks asal: project untuk Reduka

Project ini awalnya dirancang sebagai backend service untuk frontend startup
edtech bernama Reduka. Akan dipanggil dari BE Golang dengan volume request
besar di musim tryout. Dua hal jadi prioritas dari awal:

1. **Biaya per query**, karena tiap call Gemini berbayar dan banyak siswa
   bertanya satu soal yang sama dengan kalimat berbeda.
2. **Latency request pertama**. Cold start LLM client bisa 3-5 detik kalau
   diinstansiasi per request.

Sebagian besar keputusan teknis di bawah berakar dari dua kendala itu.

---

## Milestone 1: Strategi retrieval bertingkat

Iterasi pertama saya pakai pendekatan textbook: similarity search ke ChromaDB
untuk tiap query. Cepat selesai, tapi waste ke kasus simple seperti
*"jelaskan soal nomor 3"* — pattern jelas yang bisa di-detect tanpa embedding.

Setelah ngecek log mock-query, ~60% query siswa punya pola regex yang clean:
ada nomor soal, kadang ada subject, kadang ada keyword "tryout". Buat apa
saya bayar 1 embedding call kalau bisa langsung filter metadata
`{id_soal: "3", subject: "Penalaran Umum"}` ke ChromaDB?

Lalu masalah follow-up muncul: query *"kenapa jawabannya B?"* gagal di regex
karena tidak ada nomor soal eksplisit. Solusi: simpan entitas yang
sebelumnya berhasil di-extract ke Redis per-user (TTL 30 menit). Saat regex
gagal, cek cache dulu — kalau ada entity terakhir, asumsi follow-up dan
re-use.

Ini akhirnya jadi **4 layer retrieval** yang sengaja diurutkan dari termurah:
regex → similarity → cache → LLM extractor. Komentar di kode bahkan
menjelaskan filosofinya:

> *"Strategi retrieval diurutkan dari biaya paling rendah ke tertinggi"*

Hasilnya estimasi <5% query benar-benar sampai ke Layer 4 (LLM extractor).
Detail di [ADR 0002](decisions/0002-layered-retrieval.md).

---

## Milestone 2: Hybrid history dengan rolling summary

Konteks history yang dikirim ke LLM punya implikasi cost yang nyata. Naive
approach: kirim semua pesan setiap kali. Untuk sesi 20-30 pesan, ini bisa
ratusan-ribuan token tiap request. Cost balloon di sesi panjang.

Tapi ringkasan murni juga problem: LLM jadi lupa konteks soal yang baru
dibahas 2 turn yang lalu. Siswa bilang *"oh berarti yang itu"* — model
tidak tahu *"itu"* yang mana.

Solusi: **hybrid**. N=10 pesan terakhir dikirim full (untuk konteks
immediate), pesan lama di-summarize ke rolling summary (1-3 kalimat fokus ke
soal apa saja yang sudah dibahas). Summary di-update incremental setiap kali
ambang trigger (SUMMARY_TRIGGER=20) tercapai, dengan tracking
`summarized_upto` supaya tidak re-summarize pesan yang sama.

Biaya summary: 1 LLM call ringan (~200 token) saat sesi panjang. Worth it
karena bound cost untuk sesi marathon.

Detail di [ADR 0003](decisions/0003-hybrid-history.md).

---

## Milestone 3: Re-packaging dari Reduka jadi project pribadi

Reduka tidak lanjut menggunakan service ini secara reguler, jadi saya
putuskan repackage jadi project pribadi untuk portfolio. Tantangannya:

1. **Branding "Reduka" tersebar di kode** — title FastAPI, collection name
   ChromaDB (`RAG_REDUKA_DOC_KNOWLEDGE`), schema description, prefix Redis.
2. **Tidak ada UI** — service ini awalnya hanya dipanggil dari BE Golang.
   Orang luar tidak bisa "merasakan" kualitas chatbot tanpa nulis client.
3. **Storage metadata di Redis** — semua document metadata di-store sebagai
   Redis HASH dengan SET index. Cepat tapi tidak queryable, dan ilang kalau
   Redis restart tanpa AOF.
4. **Ingestion non-incremental** — `IngestionService` di-design awal untuk
   one-shot init: `__init__` me-reset ChromaDB dengan `shutil.rmtree`. Saat
   user upload file baru, semua data lama hilang.
5. **Tidak ada test infrastructure** — no `tests/`, no pytest, no fixtures.

Saya bagi pengerjaan ke 6 stage supaya tiap stage punya output yang
bisa di-commit dan tidak meninggalkan project broken di tengah jalan.

---

## Milestone 4: Polyglot persistence (Stage 4)

Saat memikirkan storage metadata document, opsi awal: pakai Postgres untuk
semuanya termasuk history chat. Konsisten, satu tool.

Tapi Redis punya use case yang Postgres tidak cocok:
- Atomic counter untuk rate limit (`INCR` + `EXPIRE` adalah 2 line)
- Sorted set sliding window
- Mutex via `SET NX EX`
- TTL native — chat history 24 jam tanpa cron job cleanup

Postgres unggul untuk:
- Persistent metadata yang harus survive restart
- Queryable filter & pagination
- Audit trail (kapan file di-upload, siapa, status)
- Schema constraints (CHECK, partial unique index)

Saya pilih **polyglot persistence**: Redis untuk semua yang TTL/atomic,
Postgres untuk persistent metadata. Detail mapping di
[ADR 0001](decisions/0001-polyglot-persistence.md).

Trade-off yang diterima: 2 service yang harus running, 2 driver yang
harus di-maintain. Tapi ini bukan over-engineered — masing-masing data
benar-benar fit ke tool-nya.

---

## Milestone 5: Refactor service yang sudah membesar (Stage 5)

`ChatService` sudah membengkak ke ~400 baris dengan dua tanggung jawab:
orchestrate retrieve→LLM (the interesting part) dan Redis history management
(plumbing). Sulit di-test history secara independen karena harus init LLM
client dan RetrieveService dulu.

Saya pecah jadi `HistoryService` (Redis I/O + rolling summary) yang
di-inject ke `ChatService`. Hasilnya: ChatService lebih fokus, history bisa
di-test dengan fakeredis + stub LLM, dan kalau nanti mau pindah history dari
Redis ke Postgres untuk archive, hanya satu file yang berubah.

Pattern serupa untuk `IngestionService` — pecah jadi `PDFParser`
(stateless parsing logic) yang di-compose oleh `IngestionService`
(orchestrator + ChromaDB I/O).

---

## Milestone 6: Fix incremental ingestion (Stage 5)

Bug yang udah di-flag di komentar kode existing tapi tidak pernah di-fix:

> *"IngestionService.run() saat ini memproses SEMUA file di raw_docs dan
> me-reset ChromaDB secara penuh (shutil.rmtree di __init__). Implikasinya:
> selama job ingestion berlangsung (~1-3 menit), query chatbot mungkin
> mendapat hasil kosong dari ChromaDB."*

Untuk service edtech yang punya siswa aktif, downtime ChromaDB 1-3 menit per
upload baru adalah unacceptable. Saya refactor ke incremental: hapus
chunk dengan metadata `source==filename` dulu, baru insert chunk baru. File
lain tidak terpengaruh.

API juga berubah: `run(filenames=None)` — backward compat kalau dipanggil
tanpa argumen, incremental kalau ada filter. Detail di
[ADR 0004](decisions/0004-incremental-ingestion.md).

---

## Milestone 7: Test infrastructure (Stage 5)

Sengaja saya tidak tulis test sebelum refactor selesai — refactor besar
biasanya butuh ubah test berkali-kali. Setelah service stabil, baru
mulai tulis test.

Fixtures yang saya bangun:
- `fake_redis` — fakeredis instance per test
- `db_session` — SQLite in-memory dengan rollback per test
- `test_client` — FastAPI TestClient dengan dependency override

Satu hack yang menarik: SQLAlchemy model punya CHECK constraint dengan
regex operator `~` yang PostgreSQL-only. SQLite tidak mendukung. Saya
strip constraint regex di conftest sebelum `create_all()` — test-only,
production tetap pakai constraint asli dari `init.sql`.

Hasilnya 69 test pass dalam <2 detik. Bukan exhaustive coverage, tapi cover
critical path: schemas validation, repository CRUD, endpoint auth + validation.

---

## Milestone 8: Frontend React + Vite (Stage 6)

Service ini awalnya tidak punya UI. Untuk portfolio, perlu ada cara
visitor merasakan kualitas chatbot tanpa nulis client.

Pilihan stack: Streamlit (cepat) vs Next.js (heavy) vs React+Vite (middle).
Saya pilih **React + Vite** — cukup profesional untuk portfolio AI Engineer
full-stack tanpa overkill framework features yang tidak dipakai.

Design philosophy yang saya pegang: **satu signature element**, sisanya
quiet. Untuk Tutor UTBK saya pakai aksen *highlighter amber* untuk source
citations — evokes the study/notebook vibe. Sisanya off-white background,
slate text, indigo aksen AI. Hindari default cream-serif look yang
AI-generated banget.

Settings (base URL, API key) disimpan di localStorage. Tidak ada backend
session di frontend — pure SPA yang call backend langsung. CORS sudah
di-allow di backend untuk demo.

---

## Milestone 9: Admin UI — document management (Stage 7)

Setelah chat UI ada, gap yang paling terasa: untuk upload file baru harus
pakai curl atau Postman. Untuk visitor portfolio yang ingin mencoba, ini
terlalu tinggi barrier-nya. Juga tidak ada cara melihat status ingestion
atau menghapus dokumen tanpa CLI.

Stage 7 menambahkan dua halaman baru tanpa menyentuh satu baris pun di
service layer:

**Tab Dokumen** — UI lengkap untuk siklus hidup dokumen:
- Upload drag & drop dengan validasi nama file di client side (`soal_*.pdf` / `jawaban_*.pdf`)
- Status card semua komponen backend dengan auto-fetch saat mount
- Trigger ingestion + progress bar yang polling setiap 3 detik
- Timeline riwayat job ingestion
- Tabel dokumen dengan status badge dan hapus via confirmation modal

**Tab Panduan** — guide 6 langkah dengan contoh format PDF soal/jawaban,
aturan label jenis ujian, dan FAQ. Tombol navigasi antar halaman langsung
dari konten panduan.

Satu keputusan teknis menarik: backend `DocumentItem` schema awalnya
mengembalikan `ingested: bool` bukan `status: string`. Admin UI butuh status
yang lebih kaya (`uploaded` / `ingested` / `failed`) untuk badge warna.
Solusi: tambah field `status` di schema (additive, backward compat) dan
petakan ORM `doc.status` langsung. Field `ingested: bool` tetap ada untuk
compat klien lama.

Satu lagi: field `IngestJobSummary` di Pydantic tidak bisa langsung
`model_validate()` dari ORM karena nama field berbeda (`total_files` vs
`files_queued`, `started_at` vs `created_at`). Solusi pakai
`validation_alias` di Pydantic v2 — alias untuk input, field name Python
untuk serialisasi. Ini lebih bersih dari manual mapper karena tipe tetap
type-safe.

Juga di sesi ini: markdown rendering untuk chat bubble. Sebelumnya jawaban
AI yang mengandung `**bold**` muncul sebagai plain text. Tambah
`react-markdown` + `remark-gfm` dengan custom Tailwind components per
elemen HTML. User pesan tetap plain text karena tidak ada nilai tambah
markdown di sana.

---

## Apa yang saya pelajari

1. **Refactor bertahap > big-bang rewrite.** Stage 1-6 jadi possible karena
   tiap stage punya kontrak yang jelas dan output yang bisa di-merge tanpa
   menunggu stage berikutnya.

2. **Backward compat matters even in personal projects.** Stage 2 split
   `schemas.py` ke folder tetap pakai `__init__.py` re-export supaya import
   lama (`from src.domain.schemas import X`) tidak break. Ini menghemat
   ratusan baris edit di endpoint files.

3. **Test infrastructure layak diinvestasi sekali.** Setup conftest yang
   solid bikin nambah test baru hampir effortless. Yang berat adalah hari
   pertama, sisanya gampang.

4. **Cost optimization adalah engineering work yang underrated.** 4-layer
   retrieval menghemat ribuan API call per hari untuk Reduka. Bukan
   trick, tapi keputusan arsitektur yang sengaja.

5. **AI sebagai pair programmer paling efektif kalau saya punya plan dulu.**
   AI bagus untuk eksekusi cepat dan brainstorm trade-off. Tapi
   keputusan apa yang mau dibangun — itu tetap saya. Plan stage 1-6 ini
   saya susun sendiri sebelum implement; AI bantu generate code per file
   sesuai blueprint yang sudah jelas.
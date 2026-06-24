# 0004 · Incremental ingestion (per-source delete-then-insert)

**Status:** Accepted
**Date:** 2026
**Stage:** 5

## Context

Versi awal `IngestionService` punya design flaw yang sudah di-flag di
komentar kode tapi tidak pernah di-fix:

```python
# Di __init__:
if os.path.exists(self.db_dir):
    shutil.rmtree(self.db_dir)
    self.logger.info("The old database has been successfully deleted (Reset).")
```

Setiap kali ingest dipanggil:
1. ChromaDB persistence dir dihapus total
2. Semua chunk lama hilang
3. `run()` iterate ALL `soal_*.pdf` di disk
4. Re-create vector store dari nol

Implikasinya untuk Reduka (production):
- Saat admin upload 1 soal baru, semua 50+ soal yang sudah di-ingest harus
  di-process ulang. Wasted ~5-10 menit + ~50× Gemini API call.
- Selama re-ingestion berjalan, query chatbot return hasil kosong
  (ChromaDB dalam state "kosong"). Siswa yang chat selama window itu
  dapat respon "tidak ada konteks ditemukan".

Untuk demo publik dengan file yang mungkin di-upload satu per satu,
behavior ini benar-benar tidak acceptable.

## Options considered

**A. Status quo — full rebuild setiap ingest**
- **Pro:** simple. Tidak perlu kompleksitas tracking.
- **Con:** see above. Disqualified.

**B. Append-only — tidak pernah hapus chunk**
- INSERT chunk baru, jangan touch yang lama
- **Pro:** sangat simple
- **Con:** duplicate kalau file sama di-ingest ulang. Vector store membengkak.
  Re-ingest setelah edit PDF tidak update chunk lama.

**C. Per-source delete-then-insert (incremental)**
- Untuk tiap file yang di-ingest, hapus chunk lama dengan metadata
  `source==filename` dulu
- Lalu insert chunk baru
- File lain tidak terpengaruh
- **Pro:** clean state per file. Re-ingest aman. File lain stay intact.
- **Con:** delete query ke ChromaDB tambah 1 operation per file.

**D. Hash-based diff**
- Hash konten file, skip jika sudah pernah di-ingest dengan hash yang sama
- **Pro:** smartest — re-ingest no-op kalau file tidak berubah
- **Con:** kompleks. Butuh column `content_hash` di documents table.
  Edge case kalau prompt structuring berubah tapi PDF sama → cached result jadi stale.

## Decision

**Option C — incremental per-source delete-then-insert.**

Implementasi di Stage 5 refactor:

```python
# IngestionService.run() — refactored
def run(self, filenames: Optional[List[str]] = None) -> int:
    # ── Tentukan target ──
    if filenames is None:
        # Backward compat: process semua soal_*.pdf di dir
        target_soal = [...]
    else:
        # Incremental: hanya files yang disebutkan
        target_soal = [f for f in filenames if f.startswith("soal_") and ...]

    # ── Process per-file ──
    for soal_filename in target_soal:
        self._delete_existing_chunks(soal_filename)  # ← key change
        docs = self._process_single_file(soal_filename)
        if docs:
            self.save_to_chroma(docs)
    
def _delete_existing_chunks(self, filename: str) -> int:
    result = self.vector_store._collection.get(
        where={"source": {"$eq": filename}},
        include=[],
    )
    doc_ids = result.get("ids", [])
    if doc_ids:
        self.vector_store._collection.delete(ids=doc_ids)
    return len(doc_ids)
```

Critical: `__init__` **tidak lagi** memanggil `shutil.rmtree(db_dir)`.
ChromaDB persistence terjaga antar restart service dan antar ingest call.

## Trade-offs accepted

1. **Per-file overhead** — `_delete_existing_chunks()` jalan untuk setiap
   file. Untuk delete pertama kali (tidak ada chunk lama), overhead ~10-50ms
   per file. Acceptable.

2. **Tidak detect "no-op re-ingest"** — kalau user re-ingest file yang sama
   tanpa perubahan, delete-then-insert tetap jalan (cost: re-embed chunks
   yang sebenarnya identik). Option D akan hindari ini, tapi kompleksitas
   tracking hash tidak worth untuk Phase 1.

3. **Eventual consistency window** — antara delete dan insert ada window
   ~100-500ms di mana query untuk file itu return kosong. Untuk file
   tertentu yang sedang di-update. Acceptable karena hanya kasus self-update.

4. **API contract change** — `run()` sekarang punya parameter optional
   `filenames`. Backward compat kalau dipanggil tanpa argumen, tapi caller
   harus update untuk dapat benefit incremental.

## Consequences

- `IngestionService.__init__` lebih simple (tidak lagi cleanup ChromaDB).
- `document.py` endpoint `_run_ingestion_job` perlu update untuk pass
  `filenames` ke `run()` — convert `file_ids` ke `original_filename` via
  `DocumentRepository.get_by_id()`.
- Re-ingest file yang sama jadi safe operation: chunk lama dihapus, baru
  dibuat. Cost: 1× re-process (image caption + LLM structuring + embedding).
- Test `ingestion_service` perlu mock ChromaDB delete + add, tidak cukup
  hanya check side effect di disk.

## Pattern consistency

Pattern delete-then-insert ini juga dipakai di endpoint
`DELETE /v1/documents/{file_id}`:
- Hapus file dari disk
- Hapus chunk dari ChromaDB dengan `source` filter
- Soft-delete row di Postgres

Konsisten dengan pattern di ingestion → easier mental model untuk maintainers.

## See also

- Implementation: `backend/src/services/ingestion_service.py` (Stage 5 version)
- Pattern reference: `backend/src/api/endpoints/document.py::delete_document`
- Engineering journal: Milestone 6 di [`journal.md`](../journal.md)

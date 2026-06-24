# 0002 · 4-layer cost-optimized retrieval strategy

**Status:** Accepted
**Date:** 2026
**Stage:** Pre-stage (kept from original Reduka design)

## Context

Setiap call ke LLM Generation API berbayar (Gemini 2.5 Flash: $0.075/1M input
tokens, $0.30/1M output). Embedding API jauh lebih murah (~10-50× lebih murah
dari generation per token). Regex bebas biaya.

Pattern query siswa yang saya observe dari log mock:
- ~60% punya pola jelas: *"jelaskan soal nomor 3 Penalaran Umum"* — ada
  nomor, kadang ada subject.
- ~25% pertanyaan follow-up: *"kenapa jawabannya B?"* — tidak ada nomor,
  tapi konteks dari turn sebelumnya jelas.
- ~10% phrasing tak terduga: *"yang tentang gradien itu lho"* — perlu
  semantic search.
- ~5% benar-benar ambigu / topik berbeda: butuh entity extraction yang
  proper.

## Options considered

**A. LLM extractor untuk semua query**
- Setiap query → LLM extract `{id_soal, subject}` → query ChromaDB
- **Pro:** consistent, satu path code
- **Con:** 1 LLM call untuk 100% query. Mahal & latency tinggi (~1-3s).

**B. Similarity search murni untuk semua**
- Setiap query → embed → ChromaDB similarity → top-k
- **Pro:** simple
- **Con:** ~25% follow-up queries ("kenapa B?") yield irrelevant results
  karena query terlalu generic. Tidak handle context dari turn sebelumnya.

**C. Hybrid bertingkat sesuai biaya**
- Try regex first (0 API call)
- Fallback similarity (1 embedding call)
- Fallback Redis cache + re-fetch (0 API call)
- Last resort LLM extractor (1 LLM call)

## Decision

**Option C — 4-layer retrieval bertingkat.**

```
Query masuk
    │
    ▼
┌─────────────────────────────────────────────┐
│ Layer 1: Regex + ChromaDB metadata filter   │ ~60% queries land here
│ [0 API call]                                 │ Latency: ~10ms
└─────────────────────┬───────────────────────┘
                      │ regex gagal / hasil kosong
                      ▼
┌─────────────────────────────────────────────┐
│ Layer 2: Similarity search                   │ ~30% land here
│ [1 embedding call]                           │ Latency: ~300ms
└─────────────────────┬───────────────────────┘
                      │ similarity score rendah / 0 hasil
                      ▼
┌─────────────────────────────────────────────┐
│ Layer 3: Redis entity cache + re-fetch       │ ~5% land here
│ [0 API call]                                 │ Latency: ~50ms
└─────────────────────┬───────────────────────┘
                      │ no cache available
                      ▼
┌─────────────────────────────────────────────┐
│ Layer 4: LLM entity extractor                │ <5% land here
│ [1 LLM call]                                 │ Latency: ~1-3s
└──────────────────────────────────────────────┘
```

Estimated cost saving vs Option A: **~95% lebih murah per query average**.

## Trade-offs accepted

1. **Code complexity** — `RetrieveService.search()` lebih panjang dengan
   4 fallback branches. Tapi tiap layer kohesif dan terdokumentasi.
2. **Debugging lebih sulit** — kalau query gagal, perlu cek layer mana yang
   trigger. Mitigated dengan info logging di tiap layer transition.
3. **Layer 3 (cache) butuh entity di Redis** — kalau cache expired (TTL 30m),
   query follow-up bisa fallback ke Layer 4. Acceptable cost untuk safety net.
4. **Layer 1 metadata filter exact match** — kalau siswa salah ketik subject
   ("Penalaran Umun" instead of "Penalaran Umum"), regex tetap match tapi
   filter ChromaDB gagal. Acceptable — fallback ke Layer 2 handle ini.

## Consequences

- `RetrieveService` jadi service kompleks (~300 baris), tapi tiap layer
  punya method terpisah yang testable independen.
- Redis schema bertambah: `entity:{user_id}` dan `context:{user_id}`.
- Saat siswa mulai sesi baru, cache di-clear via endpoint `DELETE /v1/session`.
- LLM entity extractor prompt di-tune untuk extract `{id_soal, subject}`
  saja (output JSON), bukan untuk RAG-style response.

## Validation

Cara verify cost saving: hitung distribusi layer hits di production log.
Layer 1 + 3 = 0 API call. Layer 2 = 1 embedding call. Layer 4 = 1 generation
call (paling mahal).

Untuk benchmark, jalankan suite test queries representatif dan log
`layer_hit_distribution` di RetrieveService.

## See also

- Implementation: `backend/src/services/retrieve_service.py`
- Entity extractor: `backend/src/services/regex_entities_extractor.py`
- Engineering journal: Milestone 1 di [`journal.md`](../journal.md)

# 0003 · Hybrid history with rolling summary

**Status:** Accepted
**Date:** 2026
**Stage:** Pre-stage

## Context

LLM butuh konteks percakapan sebelumnya untuk respon koheren ("oh, lanjut
soal yang tadi", "kenapa B?", dst). Tapi mengirim full history setiap kali
ada masalah cost dan latency:

- Tiap pesan rata-rata ~150 token
- Setelah 30 pesan: ~4500 token input per request
- Tiap request berbayar; siswa pakai berhari-hari → cost balloons

Tapi truncation murni juga bermasalah: kalau pesan ke-5 cuma dilihat, model
lupa soal apa yang dibahas di pesan ke-1. Siswa bilang *"tadi yang nomor 3
itu"* — model tidak tahu konteks tadi-nya apa.

## Options considered

**A. Full history setiap request**
- Kirim semua pesan ke LLM tiap call
- **Pro:** maksimum context, model tidak pernah "lupa"
- **Con:** cost & latency tumbuh linear dengan panjang sesi. 30 pesan = 4500
  token input, cost ~$0.0003/request, sesi 100 pesan = $0.001/request.
  Bukan mahal di test, tapi balloon di production.

**B. Sliding window N pesan terakhir**
- Hanya kirim N pesan terakhir, drop yang lebih lama
- **Pro:** bounded cost
- **Con:** konteks awal sesi hilang total. Siswa reference soal di awal,
  model bingung.

**C. Pure summarization**
- Compress semua history ke 1-2 paragraf ringkasan, kirim itu saja
- **Pro:** sangat compact
- **Con:** Lose immediate context. Pesan terakhir di-summarize sebelum di-respond,
  detail-detail spesifik hilang.

**D. Hybrid — full recent + rolling summary**
- N=10 pesan terakhir kirim full (immediate context preserved)
- Pesan lama di-summarize ke rolling summary (background updater)
- Kirim summary + recent 10 ke LLM

## Decision

**Option D — hybrid history.**

Konstanta:
- `MAX_RECENT_MESSAGES = 10` — N pesan terakhir dikirim full
- `SUMMARY_TRIGGER = 20` — picu summarize saat total pesan >= 20

Flow:
1. User kirim pesan ke-N
2. `HistoryService.try_summarize()` cek apakah `len(messages) >= 20`
3. Kalau ya, ambil pesan yang ada di luar window 10 terakhir
4. Gabungkan ke "old summary + new convo", panggil LLM untuk perbarui summary
5. Simpan summary baru + `summarized_upto = cutoff_index` ke Redis
6. `HistoryService.get_llm_context()` return `(summary, last 10 messages)`
7. ChatService inject summary + recent ke prompt

Counter `summarized_upto` mencegah re-summarize pesan yang sama. Incremental
update — summary lama jadi konteks untuk summary baru.

## Trade-offs accepted

1. **Summary kadang miss detail halus** — kalau siswa diskusi soal nomor 3
   dengan banyak nuance di pesan 5-15, summary mungkin generalize ke "siswa
   bertanya tentang Penalaran Umum nomor 3". Mitigated dengan window 10
   yang relatif besar.

2. **1 LLM call extra setiap kali trigger** — biaya summarize ~$0.0001
   per trigger. Trade vs saving ratusan token input per request: net positive
   untuk sesi panjang.

3. **Summary quality bergantung LLM prompt** — prompt summarize hard-coded
   di `HistoryService.try_summarize()` untuk fokus ke "soal nomor berapa dan
   materi apa yang sudah dibahas". Fokus narrow ini sengaja supaya summary
   bermanfaat untuk RAG context, bukan general chit-chat summary.

4. **Race condition theoretical** — dua request bersamaan untuk sesi yang
   sama bisa keduanya trigger summarize. Implementasi naive: dua-duanya
   tulis Redis. Last-write-wins, idempotent enough untuk use case kita.
   Tidak pakai lock karena summary content tetap valid; cuma waste 1 LLM call.

## Consequences

- Redis schema: 3 key per session:
  - `chat:messages:{user_id}:{session_id}` — LIST of JSON messages
  - `chat:summary:{user_id}:{session_id}` — STRING summary
  - `chat:summarized_upto:{user_id}:{session_id}` — STRING int counter
- TTL semua 24 jam, di-refresh tiap append message baru.
- `clear_session` endpoint hapus ketiga-tiganya.
- ChatService.generate_response panggil `history_service.try_summarize()`
  sebelum build prompt — overhead bound karena trigger dicek dulu sebelum
  panggil LLM.

## Tuning

Kalau `MAX_RECENT_MESSAGES` di-naikkan dari 10 ke 15:
- Token input naik ~50%
- Summary jarang trigger (butuh 30 pesan minimum)
- Konteks immediate lebih kaya

Kalau `SUMMARY_TRIGGER` di-turunkan dari 20 ke 14:
- Summary trigger lebih sering (cost summary call naik)
- Konteks awal sesi terjaga lebih cepat

Kombinasi optimal tergantung distribusi panjang sesi. Untuk Reduka (siswa
average sesi 8-15 pesan), nilai default (10/20) hampir tidak pernah trigger
summary — sehingga overhead minimal. Untuk demo publik dengan power user
yang chat 30+ pesan, summary jadi krusial.

## See also

- Implementation: `backend/src/services/history_service.py`
- Engineering journal: Milestone 2 di [`journal.md`](../journal.md)

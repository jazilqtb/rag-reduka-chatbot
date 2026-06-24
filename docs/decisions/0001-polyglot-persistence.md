# 0001 · Polyglot persistence: Redis + PostgreSQL

**Status:** Accepted
**Date:** 2026
**Stage:** 4

## Context

Project butuh menyimpan beberapa kategori data dengan karakteristik berbeda:

1. **Chat history** — N pesan terakhir per session, perlu append + range read.
   Boleh hilang setelah 24 jam tanpa aktivitas (TTL natural).
2. **Rolling summary** — string ringkasan sesi, TTL refresh tiap pesan baru.
3. **Entity & context cache** — JSON kecil per user, TTL 30 menit, optimasi
   retrieval cost.
4. **Rate limit counter** — atomic INCR per user per menit untuk sliding window.
5. **Ingest mutex** — boolean "ada job ingestion lagi jalan?", TTL safety 10 menit.
6. **Document metadata** — persistent, queryable, audit trail (`file_id`,
   `filename`, `status`, `chunk_count`, timestamps).
7. **Ingest job history** — audit trail per job (id, status, files, errors,
   started_at, completed_at).

Single-storage solution akan force trade-off: kalau pakai Postgres untuk semua,
TTL data perlu cron cleanup; rate limit perlu lock + transaction (slow);
mutex perlu advisory lock (complex). Kalau pakai Redis untuk semua, kehilangan
queryability dan audit trail untuk #6 dan #7.

## Options considered

**A. Redis-only** (pendekatan original sebelum Stage 4)
- All data di Redis (HASH + SET + LIST + STRING)
- Document metadata stored as HASH per file_id + SET index
- **Pro:** satu service, satu driver
- **Con:** no queryable filter (`WHERE jenis_ujian = ?`), no audit trail
  setelah Redis restart tanpa AOF, no constraint enforcement

**B. Postgres-only**
- Pakai Postgres untuk chat history, cache, rate limit, dst
- **Pro:** ACID, queryable, satu sumber kebenaran
- **Con:** TTL harus pakai cron job atau `pg_cron`. Sliding window rate limit
  butuh row lock. Mutex pakai advisory lock yang lebih kompleks.
  Latency lebih tinggi untuk hot path (chat).

**C. Polyglot — Redis untuk TTL, Postgres untuk persistent**
- Redis: chat history, summary, entity/context cache, rate limit, mutex
- Postgres: document metadata, ingest job history
- **Pro:** tiap data dapat tool yang fit. Hot path (chat) tetap di-Redis-cepat.
  Audit trail aman.
- **Con:** 2 service running. 2 driver di backend. Lebih banyak yang bisa salah.

## Decision

**Option C — polyglot persistence.**

Pembagian:

| Kategori | Storage | Alasan |
|---|---|---|
| Chat history | Redis LIST | append-only, range read, TTL 24h native |
| Rolling summary | Redis STRING | overwrite, TTL refresh |
| Entity & context cache | Redis STRING (JSON) | TTL 30m, no query needed |
| Rate limit counter | Redis sorted set | atomic ZADD + ZREMRANGEBYSCORE |
| Ingest mutex | Redis STRING (SET NX EX) | atomic, fast |
| Document metadata | Postgres `documents` | queryable, audit, constraints |
| Ingest job history | Postgres `ingest_jobs` | audit trail, status query |

## Trade-offs accepted

1. **Operational complexity** — `docker-compose.yml` punya 2 service infrastruktur
   bukan 1. `make up-infra` perlu start keduanya.
2. **Dual driver di backend** — backend butuh `redis` + `sqlalchemy + psycopg2`.
   Total ~10 MB tambahan di Docker image.
3. **Eventual consistency lintas store** — kalau insert document ke Postgres
   sukses tapi update Chroma gagal, ada inconsistency. Kami mitigate dengan
   `try/except` + rollback delete file dari disk kalau DB INSERT gagal.
4. **No cross-store transaction** — tidak ada 2-phase commit antar Redis dan
   Postgres. Untuk use case kami ini OK karena tidak ada operation yang
   benar-benar butuh atomicity lintas store.

## Consequences

- `infra/postgres/init.sql` jadi source of truth untuk schema.
- `src/db/repositories/` abstract semua DB call. Service tidak panggil SQLAlchemy
  langsung.
- `check_db_connection()` ditambahkan ke `/v1/health/detailed` supaya Postgres
  jadi visible health concern.
- Background thread untuk ingestion butuh manual session management
  (`SessionLocal()` + `try/finally close()`) karena tidak punya FastAPI
  Depends context.

## See also

- Schema: [`docs/schema.sql`](../schema.sql)
- Implementation: `backend/src/db/repositories/`
- Migration notes: Stage 4 di [`journal.md`](../journal.md)

# API Reference

Reference manual untuk endpoint UTBK Tutor RAG service. Untuk eksplorasi
interaktif, gunakan **Swagger UI** di `http://localhost:8000/docs` setelah
backend running.

## Conventions

- Base URL: `http://localhost:8000` (default) atau sesuai deployment
- Versi API: `/v1`
- Authentication: header `X-API-Key: <key>` di SEMUA request
- Content-Type: `application/json` (kecuali upload file → `multipart/form-data`)
- Error format: konsisten `{error, message, detail?}` JSON

## Authentication

Setiap request wajib menyertakan header:

```
X-API-Key: <your_api_key>
```

API key di-set saat deploy via env var `API_KEY` di `.env`.

Response error:
- `401 Unauthorized` — header `X-API-Key` tidak ada → `{"error": "missing_api_key"}`
- `403 Forbidden` — key salah → `{"error": "invalid_api_key"}`

---

## Chat

### `POST /v1/chat`

Kirim query siswa, dapat respon dari Tutor AI.

**Request body:**

```json
{
  "user_id":    "usr_student001",
  "session_id": "sess_abc123xyz",
  "query":      "Jelaskan soal nomor 3 Penalaran Umum dong"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `user_id` | string | yes | Format: `usr_<alphanum_underscore>{3,49}` |
| `session_id` | string | no | Format: `sess_<alphanum_underscore>{4,49}`. Kalau tidak diisi, server generate dan kembalikan |
| `query` | string | yes | 2-2000 char |

**Response 200:**

```json
{
  "session_id": "sess_abc123xyz",
  "answer": "Soal nomor 3 Penalaran Umum ini menanyakan tentang ...",
  "sources": [
    {
      "subject":     "Penalaran Umum",
      "jenis_ujian": "Tryout 1",
      "id_soal":     "3",
      "source":      "soal_tryout1.pdf"
    }
  ],
  "meta": {
    "latency_ms": 2150
  }
}
```

**Error codes:**

| Code | Cause |
|---|---|
| `422` | Request body invalid (validation error) |
| `429` | Rate limit terlampaui (default 30 req/menit per user_id) |
| `500` | LLM gagal setelah retry |

**Example (curl):**

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "usr_demo",
    "query": "Jelaskan soal nomor 3 PU"
  }'
```

**Example (Python):**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat",
    headers={"X-API-Key": "your_key"},
    json={"user_id": "usr_demo", "query": "Jelaskan soal nomor 3 PU"},
)
data = resp.json()
print(data["answer"])
print(f"Sources: {data['sources']}")
```

**Example (TypeScript/fetch):**

```typescript
const resp = await fetch("http://localhost:8000/v1/chat", {
  method: "POST",
  headers: {
    "X-API-Key":    apiKey,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    user_id: "usr_demo",
    query:   "Jelaskan soal nomor 3 PU",
  }),
});
const data = await resp.json();
```

---

## Documents

### `POST /v1/documents/upload`

Upload satu file PDF (soal atau jawaban). File belum masuk ChromaDB —
panggil `/v1/documents/ingest` setelah semua pasangan ter-upload.

**Request:** `multipart/form-data`

| Field | Type | Notes |
|---|---|---|
| `file` | file | PDF. Nama wajib: `soal_X.pdf` atau `jawaban_X.pdf` (X = alphanum_underscore, max 50 char) |
| `doc_type` | string | `"soal"` atau `"jawaban"` |
| `jenis_ujian` | string | Label ujian, max 100 char. Contoh: `"Tryout 1"` |

**Response 201:**

```json
{
  "file_id":     "file_2026_abc123",
  "filename":    "soal_tryout1.pdf",
  "doc_type":    "soal",
  "jenis_ujian": "Tryout 1",
  "size_bytes":  524288
}
```

**Error codes:**

| Code | Cause |
|---|---|
| `400` | Validasi gagal (nama file, MIME type, size, doc_type-filename mismatch) |
| `409` | File dengan nama sama sudah ada |

---

### `POST /v1/documents/ingest`

Mulai job ingestion async ke ChromaDB. Return `job_id` untuk polling.

**Request body:**

```json
{
  "file_ids": ["file_abc123", "file_def456"],
  "ingest_all_pending": false
}
```

Minimal salah satu harus aktif:
- `file_ids`: list `file_id` yang ingin di-ingest, ATAU
- `ingest_all_pending: true`: proses semua file `soal` dengan status `uploaded`

**Response 202:**

```json
{
  "job_id":       "job_2026_xyz789",
  "files_queued": 2
}
```

**Error codes:**

| Code | Cause |
|---|---|
| `400` | `ingest_all_pending=true` tapi tidak ada file pending |
| `404` | Salah satu `file_id` tidak ditemukan |
| `409` | Ada job ingestion yang sedang berjalan (Redis mutex) |

---

### `GET /v1/documents/ingest/{job_id}`

Polling status job ingestion.

**Response 200:**

```json
{
  "job_id":          "job_2026_xyz789",
  "status":          "done",
  "files_queued":    2,
  "files_processed": 2,
  "files_failed":    0,
  "errors":          [],
  "created_at":      "2026-01-15T10:30:00.000Z",
  "completed_at":    "2026-01-15T10:32:15.000Z"
}
```

`status`: `"processing"` | `"done"` | `"failed"`

---

### `GET /v1/documents`

List dokumen terdaftar. Pagination support.

**Query params:** `doc_type`, `jenis_ujian`, `page`, `limit` (1-100, default 20)

**Response 200:**

```json
{
  "total": 4,
  "page":  1,
  "limit": 20,
  "items": [
    {
      "file_id":     "file_abc123",
      "filename":    "soal_tryout1.pdf",
      "doc_type":    "soal",
      "jenis_ujian": "Tryout 1",
      "size_bytes":  524288,
      "ingested":    true,
      "uploaded_at": "2026-01-15T10:00:00.000Z",
      "ingested_at": "2026-01-15T10:05:00.000Z",
      "chunk_count": 25
    }
  ]
}
```

---

### `DELETE /v1/documents/{file_id}`

Hapus dokumen: dari disk + dari ChromaDB + soft-delete row di Postgres.

**Response 200:**

```json
{
  "file_id":               "file_abc123",
  "deleted_from_storage":  true,
  "deleted_from_vectordb": true,
  "chunks_removed":        25,
  "message":               "File 'soal_tryout1.pdf' berhasil dihapus. 25 chunk dihapus dari ChromaDB."
}
```

**Error codes:**

| Code | Cause |
|---|---|
| `404` | `file_id` tidak ditemukan |
| `409` | Ada job ingestion sedang berjalan |
| `422` | Format `file_id` invalid |

---

## Session

### `GET /v1/session/{user_id}/history`

Ambil history sesi spesifik.

**Query params:** `session_id` (required), `limit` (1-200, default 50)

**Response 200:**

```json
{
  "user_id":       "usr_demo",
  "session_id":    "sess_abc",
  "message_count": 6,
  "summary":       "Siswa bertanya tentang soal nomor 3 Penalaran Umum...",
  "messages": [
    {
      "role":      "human",
      "content":   "Jelaskan soal nomor 3 PU",
      "timestamp": "2026-01-15T10:00:00.000Z"
    },
    {
      "role":      "ai",
      "content":   "Soal nomor 3 ini menanyakan...",
      "timestamp": "2026-01-15T10:00:02.150Z"
    }
  ]
}
```

**Error codes:**

| Code | Cause |
|---|---|
| `404` | Sesi tidak ditemukan / sudah expired (>24 jam tanpa aktivitas) |
| `422` | Format `user_id` / `session_id` invalid |

---

### `DELETE /v1/session/{user_id}`

Hapus data sesi user. Pakai sebelum siswa mulai sesi belajar baru.

**Query params:** `session_id` (optional). Kalau ada, hanya sesi itu yang dihapus.
Kalau tidak, SEMUA key Redis milik user dihapus (chat history, summary, cache, dst).

**Response 200:**

```json
{
  "user_id": "usr_demo",
  "cleared": ["history", "summary", "entity_cache", "context_cache"]
}
```

---

## Health

### `GET /v1/health`

Fast liveness check. Tidak ping ke Redis/Postgres/ChromaDB. Aman dipanggil
sering oleh load balancer.

**Response 200:**

```json
{
  "status":    "ok",
  "timestamp": "2026-01-15T10:00:00.000Z"
}
```

---

### `GET /v1/health/detailed`

Readiness check. Ping ke semua dependency: Postgres, Redis, ChromaDB,
storage dir. Validasi config Gemini (tanpa actual API call).

**Response 200:**

```json
{
  "status":    "ok",
  "timestamp": "2026-01-15T10:00:00.000Z",
  "components": {
    "postgres": {"status": "ok", "latency_ms": 12,  "detail": "Connected to localhost:5432/tutor_utbk"},
    "redis":    {"status": "ok", "latency_ms": 2,   "detail": "Connected to localhost:6379"},
    "chromadb": {"status": "ok", "latency_ms": 45,  "detail": "Collection 'UTBK_TUTOR_KNOWLEDGE' — 25 dokumen."},
    "gemini":   {"status": "ok",                    "detail": "model=models/gemini-2.5-flash, embedding=models/gemini-embedding-001. API key configured (actual connectivity not tested)."},
    "storage":  {"status": "ok",                    "detail": "raw_docs dir accessible. 4 file PDF ditemukan."}
  }
}
```

Overall status logic:
- `"ok"` — semua komponen ok
- `"degraded"` — postgres + redis ok, tapi ada komponen lain error (masih bisa serve)
- `"down"` — postgres atau redis error (tidak bisa serve traffic)

---

## Rate Limiting

Endpoint `/v1/chat` di-rate-limit **30 request per menit per `user_id`**
(konfigurable via env `RATE_LIMIT_CHAT_MAX`).

Limit terlampaui → `429 Too Many Requests`:

```json
{
  "error":   "rate_limit_exceeded",
  "message": "Maksimum 30 request/menit per user_id terlampaui. Coba lagi sebentar."
}
```

Implementasi: Redis sorted set sliding window. Reset setiap detik.

---

## Error Format (global)

Semua error response konsisten:

```json
{
  "error":   "validation_error",
  "message": "user_id harus format 'usr_<alphanum_underscore>'...",
  "detail":  null
}
```

| Field | Notes |
|---|---|
| `error` | Code untuk programmatic handling |
| `message` | Human-readable, bisa di-show ke end user |
| `detail` | Optional, bisa berisi field validation errors atau context tambahan |
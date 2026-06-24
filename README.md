<div align="center">

# UTBK Tutor AI

**RAG-based chatbot untuk membantu siswa SMA memahami pembahasan soal tryout
UTBK SNBT (ujian masuk PTN Indonesia).**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-FF6B6B)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Apa ini?

Project ini awalnya dirancang sebagai backend service untuk frontend startup
edtech bernama Reduka, lalu di-repackage sebagai project pribadi dengan UI
mandiri supaya siapa pun bisa mencobanya.

> *Lihat branch "Main" untuk mengetahui raw project sebelum di-repackage ke format Portfolio*

Inti masalahnya: siswa yang baru selesai tryout UTBK biasanya bingung dengan
soal yang dijawab salah — mereka tahu kuncinya tapi tidak tahu **kenapa**
jawabannya itu. Tutor AI ini mengambil PDF soal + kunci jawaban, mengingest
ke vector store, lalu menjawab pertanyaan siswa dengan
mengacu ke soal spesifik menggunakan RAG.

> 🎯 **Demo singkat:** start docker compose → buka frontend → tanya
> *"Jelaskan soal nomor 3 Penalaran Umum"* → dapat penjelasan plus sitasi
> dokumen sumber.

## Tech Stack

**Backend** · Python 3.10 · FastAPI · Pydantic v2 · SQLAlchemy 2.0
**LLM/RAG** · LangChain · Google Gemini (chat 2.5-flash + embedding 001) · ChromaDB
**Storage** · PostgreSQL 16 (metadata persisten) · Redis 7 (cache + session + mutex)
**Frontend** · React 18 · TypeScript · Vite 5 · Tailwind CSS
**Infrastructure** · Docker Compose · nginx (frontend serve)

## Features

- **4-layer cost-optimized retrieval** — regex → similarity → Redis cache → LLM extractor. Terurut dari biaya termurah ke termahal supaya >95% query tidak menyentuh LLM extractor.
- **Hybrid history dengan rolling summary** — N pesan terakhir dikirim full ke LLM, sisanya diringkas otomatis ke 3 kalimat oleh background summary updater.
- **Polyglot persistence** — Redis untuk TTL-based data (chat history, cache, rate limit), PostgreSQL untuk persistent metadata (documents, ingest jobs).
- **Incremental ingestion** — re-ingest 1 file tidak hapus chunk file lain di ChromaDB. Per-source delete-then-insert.
- **Multimodal PDF parsing** — gambar di dalam PDF di-caption oleh LLM dan di-inject ke teks sebelum chunking, supaya soal yang punya grafik tetap bisa di-retrieve.
- **Admin UI bawaan** — tab Dokumen untuk upload/ingest/delete PDF, tab Panduan dengan 6-langkah guide, status sistem real-time per komponen (Postgres/Redis/ChromaDB/Gemini/Storage).
- **Microservice-friendly API** — RESTful dengan OpenAPI docs di `/docs`. Bisa dipakai sebagai backend untuk integrator lain.
- **Comprehensive testing** — 69 test pass (pytest + fakeredis + SQLite in-memory).

## Quickstart

**Prasyarat:** Docker + Docker Compose. Google Gemini API key (gratis tier OK
untuk demo).

```bash
# 1. Clone & setup env
git clone https://github.com/Jazil-CS25/tutor-utbk-rag-system.git
cd tutor-utbk-rag-system
cp .env.example .env
# Edit .env: isi GOOGLE_API_KEY (dari aistudio.google.com/apikey)

# 2. Start full stack
docker compose --profile ui up -d --build

# 3. Cek health
curl http://localhost:8000/v1/health
# {"status":"ok","timestamp":"..."}

# 4. Buka di browser
open http://localhost:3000   # macOS
xdg-open http://localhost:3000   # Linux
```

### Untuk environment dengan npm registry lambat dari Docker

Beberapa network (terutama di balik proxy/firewall korporat atau dengan
konfigurasi Docker tertentu) memiliki koneksi lambat ke npm registry
dari dalam Docker container. Kalau `docker compose build frontend`
hang lebih dari 5 menit, gunakan mode local-build:

```bash
# Prasyarat tambahan: Node.js 20+ di host
cd frontend && npm install && npm run build && cd ..

# Build & start dengan override file
docker compose -f docker-compose.yml -f docker-compose.local.yml \
               --profile ui up -d --build
```

Atau pakai shortcut:

```bash
make up-ui-local
```

Kedua mode menghasilkan image runtime yang identik (nginx serving static
files). Bedanya cuma di mana `npm run build` dijalankan — di Docker builder
stage vs di host.

## Architecture

Lihat **[docs/architecture.md](docs/architecture.md)** untuk diagram lengkap
dan penjelasan tiap komponen.

Singkatnya:

```
            ┌──────────────┐
            │  React UI    │  http://localhost:3000
            └──────┬───────┘
                   │ X-API-Key
                   ▼
            ┌──────────────┐         ┌────────────┐
            │ FastAPI      │────────▶│ ChromaDB   │  (vector store)
            │  /v1/chat    │         └────────────┘
            │  /v1/docs    │
            └──┬───────┬───┘         ┌────────────┐
               │       │             │ PostgreSQL │  (doc metadata + ingest jobs)
               │       └────────────▶└────────────┘
               │                     ┌────────────┐
               └────────────────────▶│   Redis    │  (chat history + cache + mutex)
                                     └────────────┘
                   │
                   │ via Google Gemini API
                   ▼
            ┌──────────────┐
            │  Gemini 2.5  │  (chat + embedding + image caption)
            └──────────────┘
```

## Design Decisions

Empat keputusan teknis utama yang punya implikasi besar di arsitektur.
Detail per-decision ada di `docs/decisions/`.

| # | Decision | Trade-off yang diterima |
|---|---|---|
| 1 | **[Polyglot persistence](docs/decisions/0001-polyglot-persistence.md)** — Redis + PostgreSQL bukan satu storage saja | Operational overhead 2 service. Tapi tiap data dapat tool yang tepat: TTL/atomic ops di Redis, queryable persistent di Postgres. |
| 2 | **[4-layer retrieval](docs/decisions/0002-layered-retrieval.md)** — regex → similarity → cache → LLM | Kode lebih panjang & 4 fallback path. Tapi cost <5% query sampai ke LLM extractor; sisanya 0-1 API call. |
| 3 | **[Hybrid history](docs/decisions/0003-hybrid-history.md)** — full recent + rolling summary | Summary kadang miss konteks halus. Tapi token cost bounded meski sesi panjang berhari-hari. |
| 4 | **[Incremental ingestion](docs/decisions/0004-incremental-ingestion.md)** — per-source delete-then-insert | Lebih kompleks dari rebuild total. Tapi file lain tidak hilang saat upload baru. |

## Engineering Journal

Catatan proses development, problem yang ditemui, dan reasoning di balik
refactor besar. Lihat **[docs/journal.md](docs/journal.md)**.

> AI dipakai sebagai pair programmer untuk eksekusi kode dan brainstorm
> trade-off. Arsitektur, pilihan stack, dan keputusan desain ada di section
> Design Decisions di atas dan di `docs/decisions/`.

## API Reference

OpenAPI Swagger UI: `http://localhost:8000/docs` setelah backend running.

Endpoint utama:

| Method | Path | Deskripsi |
|---|---|---|
| `POST` | `/v1/chat` | Kirim query siswa, dapat jawaban + source citation |
| `POST` | `/v1/documents/upload` | Upload PDF soal/jawaban |
| `POST` | `/v1/documents/ingest` | Mulai job ingestion async |
| `GET`  | `/v1/documents/ingest` | List riwayat job ingestion (newest-first) |
| `GET`  | `/v1/documents/ingest/{job_id}` | Polling status job |
| `GET`  | `/v1/documents` | List dokumen terdaftar |
| `DELETE` | `/v1/documents/{file_id}` | Hapus dokumen (storage + vector store) |
| `GET`  | `/v1/session/{user_id}/history` | Ambil history sesi |
| `DELETE` | `/v1/session/{user_id}` | Hapus data sesi user |
| `GET`  | `/v1/health` | Liveness check (fast) |
| `GET`  | `/v1/health/detailed` | Readiness check (semua dependency) |

Detail request/response di **[docs/api.md](docs/api.md)**.

## Project Structure

```
tutor-utbk-rag-system/
├── backend/                # FastAPI + RAG service
│   ├── main.py
│   ├── src/
│   │   ├── api/            # router + endpoints + deps
│   │   ├── core/           # config + logger + security
│   │   ├── db/             # SQLAlchemy + repositories
│   │   ├── domain/         # Pydantic schemas + ORM models
│   │   └── services/       # chat / history / retrieve / ingestion / pdf_parser
│   ├── tests/              # 69 tests (unit + integration)
│   ├── config/prompts.yaml
│   ├── data/               # raw_docs + vector_store + debug (gitignored)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               # React + Vite + TS + Tailwind
│   ├── src/
│   │   ├── api/            # client.ts + chat.ts + documents.ts
│   │   ├── components/     # Header, Navigation, ChatWindow, MessageBubble…
│   │   │   └── documents/  # SystemStatusCard, UploadDropzone, IngestPanel…
│   │   ├── hooks/          # useChat, useSettings, useDocuments, useIngestJob
│   │   ├── pages/          # DocumentsPage, GettingStartedPage
│   │   ├── types/api.ts
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── infra/postgres/init.sql # database schema
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── journal.md
│   ├── schema.sql
│   └── decisions/          # ADRs
├── notebooks/exploration.ipynb
├── docker-compose.yml
├── Makefile
└── .env.example
```

## Roadmap & Known Limitations

- [ ] **Streaming response** — saat ini chat reply menunggu LLM selesai. Pakai SSE atau WebSocket untuk word-by-word streaming.
- [ ] **Observability** — tambah structured JSON logging + Prometheus metrics.
- [ ] **Conversation archive** — pindah history dari Redis ke Postgres setelah TTL untuk analytics.
- [ ] **CI/CD** — GitHub Actions untuk lint + test + build image.
- [ ] **Auth yang lebih kuat** — saat ini single shared API key. Untuk multi-user perlu JWT/OAuth.

## Contact

- **Author:** Ach. Jazilul Qutbi
- **Email:** jazilulq@gmail.com
- **LinkedIn:** [linkedin.com/in/achjazilulqutbi](https://www.linkedin.com/in/achjazilulqutbi/)
- **GitHub:** [github.com/jazilqtb](https://github.com/jazilqtb/)

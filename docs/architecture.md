# Architecture

Penjelasan komponen, alur request, dan reasoning di balik struktur sistem.

---

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Browser (User)                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  React + Vite SPA                                                  │ │
│  │   - Chat UI (App.tsx + components/)                                │ │
│  │   - localStorage untuk settings                                    │ │
│  │   - fetch dengan X-API-Key header                                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS / X-API-Key
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend (uvicorn)                          │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │   API Layer (src/api/)                                            │  │
│  │   - router.py: aggregator /v1                                     │  │
│  │   - endpoints/chat.py, document.py, session.py, health.py         │  │
│  │   - deps.py: singletons + Depends factories                       │  │
│  │   - middleware: CORS, auth, rate limit                            │  │
│  └────────────────────────┬─────────────────────────────────────────┘  │
│                           │                                              │
│  ┌────────────────────────┴─────────────────────────────────────────┐  │
│  │   Services Layer (src/services/)                                  │  │
│  │   ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐│  │
│  │   │ ChatService     │  │ RetrieveService  │  │IngestionService ││  │
│  │   │ (orchestrator)  │  │ (4-layer)        │  │ (orchestrator)  ││  │
│  │   └────┬────────────┘  └─────┬────────────┘  └────┬────────────┘│  │
│  │        │                     │                    │              │  │
│  │   ┌────▼──────────┐    ┌─────▼──────────┐  ┌─────▼──────────┐  │  │
│  │   │HistoryService │    │RegexEntity     │  │  PDFParser     │  │  │
│  │   │(Redis + sum.) │    │Extractor       │  │ (text+caption) │  │  │
│  │   └───────────────┘    └────────────────┘  └────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │   Data Layer (src/db/, src/domain/)                              │  │
│  │   - SQLAlchemy ORM models                                         │  │
│  │   - Repository pattern: DocumentRepo + IngestJobRepo              │  │
│  │   - Session factory + FastAPI dependency                          │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐
   │ChromaDB │   │  Redis  │   │Postgres │   │ Gemini API  │
   │ (vector)│   │ (cache) │   │(persist)│   │(LLM+embed)  │
   └─────────┘   └─────────┘   └─────────┘   └─────────────┘
```

## Per-component breakdown

### API Layer

Tipis sengaja. Tugas: parse request, panggil service, format response.
Tidak ada business logic di endpoint.

- **`router.py`** — aggregator. Semua resource di-mount di `/v1`.
- **`endpoints/*.py`** — satu file per resource. Validasi format ID via
  `Depends(valid_user_id)` dll. Auth dan rate limit juga via dependency.
- **`deps.py`** — singleton service via `lru_cache(maxsize=1)`. Repository
  factory yang inject DB session per-request.

### Services Layer

Single Responsibility per service. Dependency injection lewat constructor.

| Service | Job | Dependency |
|---|---|---|
| `ChatService` | Orchestrate retrieve → format → LLM → persist | RetrieveService, HistoryService, LLM |
| `HistoryService` | Redis I/O untuk chat history + rolling summary | Redis, LLM (untuk summary) |
| `RetrieveService` | 4-layer document retrieval | Redis, ChromaDB, LLM, RegexExtractor |
| `IngestionService` | Orchestrate PDF parse → structure → save | PDFParser, ChromaDB |
| `PDFParser` | Pure PDF + image caption + LLM JSON structure | LLM |
| `RegexEntityExtractor` | Regex-based entity extraction (0 API call) | — |

### Data Layer

- **PostgreSQL** untuk persistent metadata (lihat `docs/schema.sql`):
  - `documents` table: file_id, filename, type, jenis_ujian, status, chunk_count, timestamps
  - `ingest_jobs` table: job_id, status, file_ids, counts, errors
- **Redis** untuk TTL data:
  - `chat:messages:{uid}:{sid}` — LIST of JSON messages, TTL 24h
  - `chat:summary:{uid}:{sid}` — STRING rolling summary
  - `chat:summarized_upto:{uid}:{sid}` — STRING int counter
  - `entity:{uid}` — JSON entity cache, TTL 30m
  - `context:{uid}` — JSON last context docs, TTL 30m
  - `ingest:lock` — STRING mutex, TTL 10m
  - `ratelimit:chat:{uid}` — sorted set sliding window
- **ChromaDB** untuk embeddings — collection `UTBK_TUTOR_KNOWLEDGE`.

## Request flow: `POST /v1/chat`

```
1. Client kirim {user_id, session_id?, query} + X-API-Key
                  │
                  ▼
2. FastAPI validate via Pydantic + Depends(require_api_key)
                  │
                  ▼
3. apply_chat_rate_limit() — Redis sliding window
                  │
                  ▼
4. ChatService.generate_response(query, user_id, session_id)
   │
   ├── RetrieveService.search(user_id, query)
   │   ├── Layer 1: Regex + ChromaDB metadata filter  [0 API call]
   │   ├── Layer 2: Similarity search                  [1 embedding call]
   │   ├── Layer 3: Redis entity cache + ChromaDB     [0 API call]
   │   └── Layer 4: LLM entity extractor              [1 LLM call]  ← last resort
   │
   ├── HistoryService.try_summarize()
   │   └── (if msg count >= 20) LLM summarize old msgs → Redis
   │
   ├── HistoryService.get_llm_context()
   │   └── Return (summary, last 10 msgs)
   │
   ├── Build prompt: system + context docs + history + query
   │
   ├── _invoke_with_retry(chain, …) [up to 3 retries on 429]
   │   └── LLM generate answer
   │
   └── HistoryService.append_exchange() → Redis LIST
                  │
                  ▼
5. Return {session_id, answer, sources, meta}
```

## Request flow: `POST /v1/documents/ingest`

```
1. Client kirim {file_ids: [...] OR ingest_all_pending: true}
                  │
                  ▼
2. Endpoint check Redis SET NX `ingest:lock` (mutex)
                  │
                  ▼
3. Resolve target_ids (via DocumentRepo if ingest_all_pending)
                  │
                  ▼
4. INSERT row ke `ingest_jobs` (status=pending) via IngestJobRepo
                  │
                  ▼
5. Spawn daemon thread `_run_ingestion_job`
                  │              │
                  ▼              ▼
   Return 202 {job_id}    Thread:
                          ├── UPDATE jobs SET status='running'
                          ├── IngestionService.run(filenames=[...])
                          │   └── PDFParser per file:
                          │       ├── parse_answer_key()  (regex, 0 LLM)
                          │       ├── parse_pdf_multimodal() (N image-caption LLM calls)
                          │       └── structure_text_to_documents() (1 LLM call)
                          │   └── ChromaDB delete_existing → add_documents
                          ├── UPDATE documents SET status='ingested', chunk_count=N
                          ├── UPDATE jobs SET status='completed', completed_at=NOW()
                          └── Redis DEL `ingest:lock`
                  │
                  ▼
6. Client polling GET /v1/documents/ingest/{job_id} → IngestJobRepo
```

## Why these boundaries?

**API ↔ Service:** isolasi protocol concerns dari business logic. Bisa swap
FastAPI ke Flask/Litestar tanpa sentuh service.

**Service ↔ Repository:** isolasi storage detail. Test pakai sqlite + fakeredis,
production pakai postgres + redis, code yang sama jalan di keduanya.

**Service composition (ChatService → HistoryService, IngestionService →
PDFParser):** dipecah saat kompleksitas tumbuh, bukan over-engineered dari
awal. Stage 5 refactor split — sebelumnya semuanya di satu file
800-baris.

## Where things live

| Concern | File |
|---|---|
| Configuration | `src/core/config.py` (pydantic-settings) |
| Logging | `src/core/logger.py` |
| Auth + rate limit | `src/core/security.py` |
| LLM client init | `src/services/chat_service.py` + `ingestion_service.py` |
| Embedding model init | `src/services/retrieve_service.py` + `ingestion_service.py` |
| Prompt templates | `config/prompts.yaml` |
| ChromaDB persist dir | `data/vector_store/` |
| Uploaded PDFs | `data/raw_docs/` |
| Debug ingestion JSON | `data/debug/` |
# UTBK Tutor Frontend

React + Vite + TypeScript + Tailwind. SPA dengan tiga tab: **Chat**, **Dokumen** (admin UI), dan **Panduan**.

## Stack

- **React 18** dengan TypeScript strict mode
- **Vite 5** untuk dev server dan production build
- **Tailwind CSS** dengan custom design tokens (lihat `tailwind.config.ts`)
- **react-markdown** + **remark-gfm** untuk render markdown di bubble chat AI
- Tanpa state library tambahan — pure React hooks (useState, useEffect, useCallback, useMemo)
- Fonts via Google Fonts: Inter (body) + JetBrains Mono (citations, kode)

## Design philosophy

Satu signature element: **highlighter amber** untuk source citations dan key
accent. Sisanya quiet — off-white background, slate text, indigo untuk aksen
AI. Hindari default AI-style cream/serif look.

## Quickstart

```bash
# Install deps
npm install

# Dev mode (http://localhost:5173)
npm run dev

# Production build → dist/
npm run build

# Preview production build
npm run preview
```

Setelah `npm run dev`, buka aplikasi, klik tombol kuning di kanan atas, lalu
masukkan:

- **Base URL** — endpoint backend FastAPI (default `http://localhost:8000`)
- **API Key** — nilai dari env `API_KEY` di backend `.env`

Settings disimpan di `localStorage`.

## Halaman & Fitur

### Tab Chat
Interface tanya-jawab dengan Tutor AI. Jawaban chatbot merender markdown (bold, italic, heading, code block, tabel, list) secara native. Footer tiap bubble AI menampilkan latency dan estimasi token.

### Tab Dokumen (Admin UI)
Manajemen dokumen tanpa perlu curl/Postman:
- **Status Sistem** — health card tiap komponen backend (Postgres, Redis, ChromaDB, Gemini, Storage) dengan refresh manual
- **Upload** — drag & drop atau klik, validasi nama file `soal_*.pdf`/`jawaban_*.pdf` di client side, progress per file
- **Ingestion** — trigger ingest semua dokumen pending, progress bar dengan polling setiap 3 detik
- **Riwayat Ingestion** — timeline job sebelumnya dengan durasi dan error detail
- **Tabel Dokumen** — daftar semua file dengan status badge, ukuran, chunk count, dan tombol hapus (dengan konfirmasi modal)

### Tab Panduan
6-langkah guide lengkap: format nama file, struktur isi PDF soal/jawaban, label jenis ujian, cara upload & ingest, cara chat. Plus FAQ.

## Struktur folder

```
src/
├── api/
│   ├── client.ts           # ApiClient class dengan X-API-Key header
│   ├── chat.ts             # sendChat, clearSession
│   └── documents.ts        # listDocuments, uploadDocument, deleteDocument,
│                           # triggerIngest, getIngestJobStatus, listIngestJobs,
│                           # getHealthDetail
├── components/
│   ├── ChatWindow.tsx
│   ├── Header.tsx          # Sticky header dengan tombol Settings + Sesi baru (conditional)
│   ├── InputBox.tsx
│   ├── MessageBubble.tsx   # Markdown rendering untuk pesan AI
│   ├── Navigation.tsx      # Tab Chat / Dokumen / Panduan
│   ├── SettingsPanel.tsx
│   ├── SourceCitation.tsx
│   └── documents/
│       ├── SystemStatusCard.tsx    # Health detail per komponen
│       ├── UploadDropzone.tsx      # Drag & drop dengan validasi filename
│       ├── IngestPanel.tsx         # Trigger ingest + progress bar
│       ├── IngestJobLog.tsx        # Timeline riwayat job
│       ├── DeleteConfirmModal.tsx  # Modal konfirmasi hapus
│       └── DocumentTable.tsx       # Tabel dokumen dengan status badge
├── hooks/
│   ├── useChat.ts          # Kirim pesan + state messages
│   ├── useSettings.ts      # localStorage settings
│   ├── useDocuments.ts     # Fetch/upload/delete dokumen
│   └── useIngestJob.ts     # Trigger ingest + polling 3s + job history
├── lib/utils.ts            # cn(), generateUserId(), generateSessionId(), formatLatency()
├── pages/
│   ├── DocumentsPage.tsx   # Compose semua komponen dokumen
│   └── GettingStartedPage.tsx  # 6-langkah panduan + FAQ
├── styles/globals.css
├── types/api.ts            # TypeScript types matching backend Pydantic schemas
├── App.tsx                 # State-based routing 3 halaman
└── main.tsx
```

## Docker

```bash
# Build image (dari folder frontend/)
docker build -t utbk-tutor-frontend .

# Run
docker run -p 3000:80 utbk-tutor-frontend
```

Atau lewat `docker compose --profile ui up` dari repo root.

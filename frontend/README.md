# UTBK Tutor Frontend

React + Vite + TypeScript + Tailwind. Single-page chat UI untuk RAG backend.

## Stack

- **React 18** dengan TypeScript strict mode
- **Vite 5** untuk dev server dan production build
- **Tailwind CSS** dengan custom design tokens (lihat `tailwind.config.ts`)
- Tanpa state library tambahan — pure React hooks (useState, useEffect)
- Fonts via Google Fonts: Inter (body) + JetBrains Mono (citations)

## Design philosophy

Satu signature element: **highlighter amber** untuk source citations dan key
accent. Sisanya quiet — off-white background, slate text, indigo untuk aksen
AI. Hindari default AI-style cream/serif look.

## Quickstart

```bash
# Install deps
npm install

# Dev mode (http://localhost:3000)
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

## Struktur folder

```
src/
├── api/             # HTTP client + endpoint wrappers (Batch 6.3)
├── components/      # React komponen (Batch 6.3)
├── hooks/           # Custom hooks (Batch 6.3)
├── lib/             # Util helpers (Batch 6.3)
├── styles/
│   └── globals.css  # Tailwind base
├── types/           # TypeScript types (Batch 6.3)
├── App.tsx
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

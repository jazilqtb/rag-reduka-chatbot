"""
Entry point aplikasi FastAPI — RAG Reduka Tutor AI UTBK SNBT.

Jalankan dengan:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.router import api_router
from src.core.logger import get_logger

logger = get_logger("main")


# ── Lifespan: warm-up singleton services saat startup ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm-up dilakukan sekali saat startup agar request pertama tidak lambat.
    lru_cache di deps.py memastikan service tidak re-init setiap request.
    """
    logger.info("=" * 60)
    logger.info("  RAG Reduka API — Starting up...")
    logger.info("=" * 60)

    from src.api.deps import _get_redis_singleton, _get_chat_service_singleton

    # Inisialisasi Redis singleton
    try:
        redis = _get_redis_singleton()
        redis.ping()
        logger.info("  [OK] Redis connected.")
    except Exception as e:
        logger.error(f"  [WARN] Redis connection failed: {e}. Lanjut tanpa Redis.")

    # Inisialisasi ChatService singleton (sekaligus RetrieveService + embedding model)
    try:
        _get_chat_service_singleton()
        logger.info("  [OK] ChatService initialized (LLM + Embeddings + ChromaDB).")
    except Exception as e:
        logger.error(f"  [ERROR] ChatService init failed: {e}")

    logger.info("=" * 60)
    logger.info("  RAG Reduka API — Ready to serve.")
    logger.info("=" * 60)

    yield  # Aplikasi berjalan di sini

    logger.info("RAG Reduka API — Shutting down.")


# ── Inisialisasi FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Reduka — Tutor AI UTBK SNBT",
    description=(
        "API untuk Tutor AI dari Bimbel Reduka yang membantu siswa SMA/Gapyear "
        "memahami soal tryout dan materi UTBK SNBT.\n\n"
        "**Autentikasi:** Semua endpoint membutuhkan header `X-API-Key`.\n\n"
        "**Rate Limit:** Endpoint `/v1/chat` dibatasi 30 request/menit per user_id."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# Sesuaikan allow_origins dengan domain BE Golang di production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Ganti ke domain spesifik di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount semua router ────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/v1")


# ── Root endpoint (tanpa auth, untuk quick liveness check) ───────────────────
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return JSONResponse(
        content={
            "service": "RAG Reduka Tutor AI",
            "status":  "running",
            "docs":    "/docs",
            "health":  "/v1/health",
        }
    )
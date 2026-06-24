"""
Entry point aplikasi FastAPI — UTBK Tutor RAG.

Jalankan dengan:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    logger.info("  UTBK Tutor RAG API — Starting up...")
    logger.info("=" * 60)

    from src.api.deps import _get_chat_service_singleton, _get_redis_singleton
    from src.db.session import check_db_connection

    # 1. PostgreSQL — fail-fast kalau DB tidak reachable
    try:
        if check_db_connection():
            logger.info("  [OK] PostgreSQL connected.")
        else:
            logger.error(
                "  [WARN] PostgreSQL connection failed. "
                "Document/ingest endpoints akan error."
            )
    except Exception as e:
        logger.error(f"  [WARN] PostgreSQL check failed: {e}")

    # 2. Redis singleton
    try:
        redis = _get_redis_singleton()
        redis.ping()
        logger.info("  [OK] Redis connected.")
    except Exception as e:
        logger.error(f"  [WARN] Redis connection failed: {e}. Lanjut tanpa Redis.")

    # 3. ChatService singleton (sekaligus RetrieveService + embedding model)
    try:
        _get_chat_service_singleton()
        logger.info("  [OK] ChatService initialized (LLM + Embeddings + ChromaDB).")
    except Exception as e:
        logger.error(f"  [ERROR] ChatService init failed: {e}")

    logger.info("=" * 60)
    logger.info("  UTBK Tutor RAG API — Ready to serve.")
    logger.info("=" * 60)

    yield  # Aplikasi berjalan di sini

    logger.info("UTBK Tutor RAG API — Shutting down.")


# ── Inisialisasi FastAPI ──────────────────────────────────────────────────────
app = FastAPI(
    title="UTBK Tutor AI — RAG Chatbot",
    description=(
        "API untuk chatbot RAG yang membantu siswa SMA / gap-year memahami "
        "soal-soal tryout UTBK SNBT (ujian masuk PTN di Indonesia).\n\n"
        "**Autentikasi:** Semua endpoint membutuhkan header `X-API-Key`.\n\n"
        "**Rate Limit:** Endpoint `/v1/chat` dibatasi 30 request/menit per user_id."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
# TODO: lock down ke origin frontend di production (Stage 6 saat deploy).
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Mount routers ────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/v1")


# ── Root endpoint (info) ─────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "UTBK Tutor AI — RAG Chatbot",
        "version": app.version,
        "docs":    "/docs",
        "health":  "/v1/health",
    }
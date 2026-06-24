"""
Endpoints: Health Check
  GET /v1/health           — Fast liveness check (no external calls)
  GET /v1/health/detailed  — Readiness check (ping Redis + PostgreSQL + ChromaDB)

Stage 4: Tambah komponen PostgreSQL di detailed check.
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from redis import Redis

from src.api.deps import get_redis
from src.core.config import settings
from src.core.logger import get_logger
from src.db.session import check_db_connection
from src.domain.schemas import (
    ComponentStatus,
    HealthDetailedResponse,
    HealthResponse,
)

router = APIRouter(tags=["Health"])
logger = get_logger("endpoint.health")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# LIVENESS — Fast check, no external calls
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check (cepat, tanpa external call)",
    description="Cek apakah server hidup. Tidak ping ke Redis/Postgres/ChromaDB.",
)
async def health_check() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=_utcnow_iso())


# ══════════════════════════════════════════════════════════════════════════════
# READINESS — Detailed check, ping all dependencies
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/health/detailed",
    response_model=HealthDetailedResponse,
    summary="Readiness check (lengkap, ping semua dependency)",
    description=(
        "Ping semua dependency: PostgreSQL, Redis, ChromaDB, dan validasi config Gemini. "
        "Tidak memanggil Gemini API untuk menghemat cost & latency. "
        "Jika ada yang `error`, status menjadi `degraded` atau `down`."
    ),
)
async def health_check_detailed(
    redis: Redis = Depends(get_redis),
) -> HealthDetailedResponse:

    components: dict[str, ComponentStatus] = {}
    any_error = False

    # ── 1. PostgreSQL ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        if check_db_connection():
            db_latency = int((time.perf_counter() - t0) * 1000)
            components["postgres"] = ComponentStatus(
                status     = "ok",
                latency_ms = db_latency,
                detail     = (
                    f"Connected to {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
                    f"{settings.POSTGRES_DB}"
                ),
            )
        else:
            any_error = True
            components["postgres"] = ComponentStatus(
                status = "error",
                detail = (
                    f"Cannot connect to {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
                    f"{settings.POSTGRES_DB}"
                ),
            )
    except Exception as e:
        any_error = True
        components["postgres"] = ComponentStatus(status="error", detail=str(e))

    # ── 2. Redis ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        redis.ping()
        redis_latency = int((time.perf_counter() - t0) * 1000)
        components["redis"] = ComponentStatus(
            status     = "ok",
            latency_ms = redis_latency,
            detail     = f"Connected to {settings.REDIS_HOST}:{settings.REDIS_PORT}",
        )
    except Exception as e:
        any_error = True
        components["redis"] = ComponentStatus(status="error", detail=str(e))

    # ── 3. ChromaDB ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        emb = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_document",
            google_api_key=settings.GOOGLE_API_KEY,
        )
        vector_store = Chroma(
            collection_name="UTBK_TUTOR_KNOWLEDGE",
            embedding_function=emb,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
        )
        doc_count = vector_store._collection.count()
        chroma_latency = int((time.perf_counter() - t0) * 1000)
        components["chromadb"] = ComponentStatus(
            status     = "ok",
            latency_ms = chroma_latency,
            detail     = f"Collection 'UTBK_TUTOR_KNOWLEDGE' — {doc_count} dokumen.",
        )
    except Exception as e:
        any_error = True
        components["chromadb"] = ComponentStatus(status="error", detail=str(e))

    # ── 4. Gemini Config (tanpa API call) ────────────────────────────────────
    # Kita hanya cek apakah API key dan model name terkonfigurasi.
    # Memanggil Gemini API di health check akan membuang token dan menambah latency.
    try:
        api_key_set = bool(getattr(settings, "GOOGLE_API_KEY", ""))
        model_name  = getattr(settings, "GENAI_MODEL", "")
        embed_name  = getattr(settings, "EMBEDDING_MODEL", "")

        if api_key_set and model_name and embed_name:
            components["gemini"] = ComponentStatus(
                status = "ok",
                detail = (
                    f"model={model_name}, embedding={embed_name}. "
                    f"API key configured (actual connectivity not tested)."
                ),
            )
        else:
            any_error = True
            components["gemini"] = ComponentStatus(
                status = "error",
                detail = "GOOGLE_API_KEY atau GENAI_MODEL tidak terkonfigurasi di settings.",
            )
    except Exception as e:
        any_error = True
        components["gemini"] = ComponentStatus(status="error", detail=str(e))

    # ── 5. Storage (raw_docs dir) ─────────────────────────────────────────────
    try:
        raw_docs_dir = settings.DATA_DIR / "raw_docs"
        accessible   = raw_docs_dir.exists() and raw_docs_dir.is_dir()
        pdf_count    = len(list(raw_docs_dir.glob("*.pdf"))) if accessible else 0

        if accessible:
            components["storage"] = ComponentStatus(
                status = "ok",
                detail = f"raw_docs dir accessible. {pdf_count} file PDF ditemukan.",
            )
        else:
            any_error = True
            components["storage"] = ComponentStatus(
                status = "error",
                detail = f"Direktori raw_docs tidak ditemukan: {raw_docs_dir}",
            )
    except Exception as e:
        any_error = True
        components["storage"] = ComponentStatus(status="error", detail=str(e))

    # ── Overall status ────────────────────────────────────────────────────────
    # "ok"       → semua komponen ok
    # "degraded" → core (postgres+redis) ok tapi ada komponen lain error
    # "down"     → core (postgres atau redis) error — tidak bisa serve traffic
    postgres_ok = components.get("postgres", ComponentStatus(status="error")).status == "ok"
    redis_ok    = components.get("redis",    ComponentStatus(status="error")).status == "ok"
    core_ok     = postgres_ok and redis_ok

    if not any_error:
        overall = "ok"
    elif core_ok:
        overall = "degraded"
    else:
        overall = "down"

    logger.info(
        f"[Health/Detailed] status={overall}, "
        f"components={[f'{k}:{v.status}' for k, v in components.items()]}"
    )

    return HealthDetailedResponse(
        status     = overall,
        timestamp  = _utcnow_iso(),
        components = components,
    )
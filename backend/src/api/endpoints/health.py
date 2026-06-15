"""
Endpoints: Health Check
  GET /v1/health          — Fast liveness check (<50ms, tanpa API call eksternal)
  GET /v1/health/detailed — Detailed readiness check (Redis + ChromaDB + Storage)

Catatan desain:
  - Gemini API TIDAK di-ping di health check untuk menghindari biaya dan latency.
    Ketersediaan model diindikasikan hanya dari konfigurasi (model name + API key tersedia).
  - /v1/health bisa dipakai sebagai liveness probe (k8s/docker).
  - /v1/health/detailed bisa dipakai sebagai readiness probe.
"""

import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from redis import Redis

from src.api.deps import get_redis
from src.core.config import settings
from src.core.logger import get_logger
from src.domain.schemas import (
    ComponentStatus,
    HealthDetailedResponse,
    HealthResponse,
)

router = APIRouter(prefix="/health", tags=["Health"])
logger = get_logger("endpoint.health")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════════════════════
# GET /v1/health — Fast liveness check
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "",
    response_model=HealthResponse,
    summary="Liveness check",
    description=(
        "Cek cepat apakah service sedang berjalan. "
        "Tidak melakukan pengecekan ke dependensi eksternal (Redis, ChromaDB). "
        "Cocok dipakai sebagai **liveness probe**."
    ),
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status    = "ok",
        timestamp = _utcnow_iso(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /v1/health/detailed — Readiness check
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/detailed",
    response_model=HealthDetailedResponse,
    summary="Readiness check (detail semua komponen)",
    description=(
        "Cek status semua komponen yang dibutuhkan service:\n\n"
        "- **redis**: Koneksi Redis (PING)\n"
        "- **chromadb**: ChromaDB collection & jumlah dokumen\n"
        "- **gemini**: Validasi konfigurasi model (tanpa API call agar tidak berbiaya)\n"
        "- **storage**: Aksesibilitas direktori raw_docs\n\n"
        "Overall `status` = `ok` hanya jika semua komponen `ok`. "
        "Jika ada yang `error`, status menjadi `degraded` atau `down`."
    ),
)
async def health_check_detailed(
    redis: Redis = Depends(get_redis),
) -> HealthDetailedResponse:

    components: dict[str, ComponentStatus] = {}
    any_error = False

    # ── 1. Redis ──────────────────────────────────────────────────────────────
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
        components["redis"] = ComponentStatus(
            status = "error",
            detail = str(e),
        )

    # ── 2. ChromaDB ───────────────────────────────────────────────────────────
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
            collection_name="RAG_REDUKA_DOC_KNOWLEDGE",
            embedding_function=emb,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
        )
        doc_count = vector_store._collection.count()
        chroma_latency = int((time.perf_counter() - t0) * 1000)
        components["chromadb"] = ComponentStatus(
            status     = "ok",
            latency_ms = chroma_latency,
            detail     = f"Collection 'RAG_REDUKA_DOC_KNOWLEDGE' — {doc_count} dokumen.",
        )
    except Exception as e:
        any_error = True
        components["chromadb"] = ComponentStatus(
            status = "error",
            detail = str(e),
        )

    # ── 3. Gemini Config (tanpa API call) ────────────────────────────────────
    # Kita hanya cek apakah API key dan model name terkonfigurasi.
    # Memanggil Gemini API di health check akan membuang token dan menambah latency.
    try:
        api_key_set  = bool(getattr(settings, "GOOGLE_API_KEY", ""))
        model_name   = getattr(settings, "GENAI_MODEL", "")
        embed_name   = getattr(settings, "EMBEDDING_MODEL", "")

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
        components["gemini"] = ComponentStatus(
            status = "error",
            detail = str(e),
        )

    # ── 4. Storage (raw_docs dir) ─────────────────────────────────────────────
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
        components["storage"] = ComponentStatus(
            status = "error",
            detail = str(e),
        )

    # ── Overall status ────────────────────────────────────────────────────────
    # "ok"       → semua komponen ok
    # "degraded" → redis ok tapi ada komponen lain error (masih bisa serve)
    # "down"     → redis error (tidak bisa menyimpan history / rate limit)
    redis_ok = components.get("redis", ComponentStatus(status="error")).status == "ok"

    if not any_error:
        overall = "ok"
    elif redis_ok:
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
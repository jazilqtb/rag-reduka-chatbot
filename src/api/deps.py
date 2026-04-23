"""
FastAPI Shared Dependencies.

Semua dependency diinject via FastAPI Depends() — tidak ada yang diinstansiasi
langsung di endpoint. Ini memastikan:
  - Singleton service (LLM client tidak re-init setiap request)
  - Validasi terpusat (API key, rate limit, format ID)
  - Mudah di-mock saat unit testing
"""

import re
from functools import lru_cache
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from redis import Redis

from src.core.config import settings
from src.core.logger import get_logger
from src.core.security import verify_api_key, check_rate_limit
from src.services.chat_service import ChatService

logger = get_logger("deps")

# ── Regex validators ──────────────────────────────────────────────────────────
_RE_USER_ID    = re.compile(r"^usr_[a-zA-Z0-9_]{3,49}$")
_RE_SESSION_ID = re.compile(r"^sess_[a-zA-Z0-9_]{4,49}$")
_RE_FILE_ID    = re.compile(r"^file_[a-zA-Z0-9_]{1,80}$")
_RE_JOB_ID     = re.compile(r"^job_[a-zA-Z0-9_]{1,80}$")


# ══════════════════════════════════════════════════════════════════════════════
# SINGLETON SERVICES
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def _get_redis_singleton() -> Redis:
    """
    Singleton Redis connection — dibuat sekali, dipakai semua request.
    lru_cache(maxsize=1) memastikan hanya ada satu instance.
    """
    r = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        decode_responses=True,
    )
    try:
        r.ping()
        logger.info("Redis connection established.")
    except Exception as e:
        logger.error(f"Redis connection FAILED: {e}")
    return r


@lru_cache(maxsize=1)
def _get_chat_service_singleton() -> ChatService:
    """
    Singleton ChatService — LLM client dan embedding model diinit sekali.
    Re-init setiap request akan sangat lambat dan boros.
    """
    logger.info("Initializing ChatService singleton...")
    return ChatService()


def get_redis() -> Redis:
    """Dependency: kembalikan Redis singleton."""
    return _get_redis_singleton()


def get_chat_service() -> ChatService:
    """Dependency: kembalikan ChatService singleton."""
    return _get_chat_service_singleton()


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")
) -> None:
    """
    Dependency: validasi X-API-Key header.
    Gunakan di semua endpoint dengan Depends(require_api_key).
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error":   "missing_api_key",
                "message": "Header 'X-API-Key' wajib disertakan.",
            },
        )
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error":   "invalid_api_key",
                "message": "API key tidak valid.",
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# PATH PARAMETER VALIDATORS
# ══════════════════════════════════════════════════════════════════════════════

def valid_user_id(user_id: str) -> str:
    """
    Dependency untuk path parameter user_id.
    Contoh: GET /v1/session/{user_id}/history
    """
    if not _RE_USER_ID.match(user_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error":   "invalid_user_id",
                "message": (
                    f"user_id tidak valid: '{user_id}'. "
                    "Format: usr_<alphanum_underscore>, panjang 4-53 char."
                ),
            },
        )
    return user_id


def valid_file_id(file_id: str) -> str:
    """
    Dependency untuk path parameter file_id.
    Contoh: DELETE /v1/documents/{file_id}
    """
    if not _RE_FILE_ID.match(file_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error":   "invalid_file_id",
                "message": (
                    f"file_id tidak valid: '{file_id}'. "
                    "Format: file_<alphanum_underscore>."
                ),
            },
        )
    return file_id


def valid_job_id(job_id: str) -> str:
    """
    Dependency untuk path parameter job_id.
    Contoh: GET /v1/documents/ingest/{job_id}
    """
    if not _RE_JOB_ID.match(job_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error":   "invalid_job_id",
                "message": (
                    f"job_id tidak valid: '{job_id}'. "
                    "Format: job_<alphanum_underscore>."
                ),
            },
        )
    return job_id


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════════

def build_rate_limit_dep(user_id_source: str = "body"):
    """
    Factory untuk membuat dependency rate limiter.
    Dipakai di endpoint chat yang butuh rate limiting per user_id.

    Penggunaan di endpoint:
        async def chat(req: ChatRequest, _: None = Depends(chat_rate_limit)):
    """
    async def _rate_limit_dep(
        # user_id didapat dari request body yang sudah divalidasi sebelumnya
        # Dependency ini dipanggil setelah body parsed, tapi kita perlu user_id
        # Solusi: caller inject user_id secara manual (lihat chat.py)
    ):
        pass
    return _rate_limit_dep


async def apply_chat_rate_limit(
    user_id: str,
    redis:   Redis = Depends(get_redis),
) -> None:
    """
    Helper rate limiter untuk endpoint chat.
    Dipanggil manual dari dalam endpoint handler karena butuh user_id dari body.
    """
    check_rate_limit(user_id=user_id, redis=redis)
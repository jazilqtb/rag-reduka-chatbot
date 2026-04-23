"""
Endpoints: Session Management
  GET    /v1/session/{user_id}/history   — Ambil history percakapan per sesi
  DELETE /v1/session/{user_id}           — Hapus semua data sesi user (reset)
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from redis import Redis

from src.api.deps import (
    get_chat_service,
    get_redis,
    require_api_key,
    valid_user_id,
)
from src.core.logger import get_logger
from src.core.security import _RE_SESSION_ID  # reuse validator dari security
from src.domain.schemas import (
    ErrorResponse,
    MessageItem,
    SessionClearResponse,
    SessionHistoryResponse,
)
from src.services.chat_service import ChatService

router = APIRouter(prefix="/session", tags=["Session"])
logger = get_logger("endpoint.session")


# ── Helper: validasi session_id query param ───────────────────────────────────

def _require_session_id(session_id: Optional[str]) -> str:
    """Pastikan session_id valid jika disertakan sebagai query param."""
    if not session_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error":   "missing_session_id",
                "message": "Query parameter 'session_id' wajib disertakan.",
            },
        )
    if not _RE_SESSION_ID.match(session_id.strip()):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error":   "invalid_session_id",
                "message": (
                    f"session_id tidak valid: '{session_id}'. "
                    "Format: sess_<alphanum_underscore>, panjang 5-54 char."
                ),
            },
        )
    return session_id.strip()


@router.get(
    "/{user_id}/history",
    response_model=SessionHistoryResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "Sesi tidak ditemukan atau sudah expired"},
        422: {"model": ErrorResponse},
    },
    summary="Ambil history percakapan",
    description=(
        "Kembalikan seluruh pesan dalam satu sesi percakapan beserta summary "
        "jika ada. Query param `session_id` wajib disertakan. "
        "History di Redis memiliki TTL 24 jam — jika expired, response akan kosong."
    ),
)
async def get_session_history(
    user_id:      str           = Depends(valid_user_id),
    session_id:   Optional[str] = Query(default=None, description="ID sesi yang ingin diambil historynya."),
    limit:        int           = Query(default=50, ge=1, le=200, description="Jumlah pesan terakhir yang dikembalikan."),
    chat_service: ChatService   = Depends(get_chat_service),
    _:            None          = Depends(require_api_key),
) -> SessionHistoryResponse:

    session_id = _require_session_id(session_id)

    raw_messages = chat_service.get_session_messages(user_id, session_id)
    summary      = chat_service.get_session_summary(user_id, session_id)

    if not raw_messages and not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "session_not_found",
                "message": (
                    f"Sesi '{session_id}' untuk user '{user_id}' tidak ditemukan "
                    "atau sudah expired (>24 jam tanpa aktivitas)."
                ),
            },
        )

    # Ambil hanya 'limit' pesan terakhir untuk response
    recent_raw = raw_messages[-limit:] if len(raw_messages) > limit else raw_messages

    messages = [
        MessageItem(
            role      = msg.get("role", ""),
            content   = msg.get("content", ""),
            timestamp = msg.get("timestamp"),
        )
        for msg in recent_raw
    ]

    logger.info(
        f"[GET /session/{user_id}/history] session='{session_id}', "
        f"total_msg={len(raw_messages)}, returned={len(messages)}"
    )

    return SessionHistoryResponse(
        user_id       = user_id,
        session_id    = session_id,
        message_count = len(raw_messages),
        summary       = summary,
        messages      = messages,
    )


@router.delete(
    "/{user_id}",
    response_model=SessionClearResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    summary="Hapus semua data sesi user",
    description=(
        "Hapus history chat, summary, dan Redis cache (entity + context) untuk user ini. "
        "Jika `session_id` disertakan, hanya sesi tersebut yang dihapus. "
        "Jika tidak, seluruh Redis key untuk user ini dihapus (semua sesi). "
        "Gunakan sebelum siswa memulai sesi belajar baru agar AI tidak terkontaminasi "
        "konteks soal yang sudah lama."
    ),
)
async def clear_session(
    user_id:      str           = Depends(valid_user_id),
    session_id:   Optional[str] = Query(
        default=None,
        description="Hapus sesi spesifik. Jika kosong, semua data user dihapus."
    ),
    chat_service: ChatService   = Depends(get_chat_service),
    redis:        Redis         = Depends(get_redis),
    _:            None          = Depends(require_api_key),
) -> SessionClearResponse:

    cleared_keys = []

    if session_id:
        # ── Hapus sesi spesifik ───────────────────────────────────────────────
        session_id = _require_session_id(session_id)
        cleared_keys = chat_service.clear_session(user_id, session_id)
        logger.info(
            f"[DELETE /session/{user_id}] session='{session_id}' dihapus. "
            f"Keys cleared: {cleared_keys}"
        )
    else:
        # ── Hapus semua key milik user_id (scan Redis) ────────────────────────
        # Pattern key milik user ini:
        #   chat:messages:{user_id}:*
        #   chat:summary:{user_id}:*
        #   chat:summarized_upto:{user_id}:*
        #   entity:{user_id}
        #   context:{user_id}
        #   ratelimit:chat:{user_id}
        patterns = [
            f"chat:messages:{user_id}:*",
            f"chat:summary:{user_id}:*",
            f"chat:summarized_upto:{user_id}:*",
        ]
        exact_keys = [
            f"entity:{user_id}",
            f"context:{user_id}",
            f"ratelimit:chat:{user_id}",
        ]

        try:
            # Scan semua pattern key
            all_keys_to_delete = list(exact_keys)
            for pattern in patterns:
                cursor = 0
                while True:
                    cursor, keys = redis.scan(cursor=cursor, match=pattern, count=100)
                    all_keys_to_delete.extend(keys)
                    if cursor == 0:
                        break

            # Hapus semua key
            if all_keys_to_delete:
                deleted_count = redis.delete(*all_keys_to_delete)
                cleared_keys  = all_keys_to_delete[:deleted_count] if deleted_count else all_keys_to_delete
            else:
                cleared_keys = []

            logger.info(
                f"[DELETE /session/{user_id}] Semua sesi dihapus. "
                f"{len(cleared_keys)} key dihapus dari Redis."
            )

        except Exception as e:
            logger.error(f"[DELETE /session/{user_id}] Error: {e}")
            cleared_keys = []

    return SessionClearResponse(
        user_id = user_id,
        cleared = cleared_keys,
    )
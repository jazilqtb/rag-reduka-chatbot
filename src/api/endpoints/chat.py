"""
Endpoint: Chat
  POST /v1/chat — Generate respon AI berdasarkan query siswa
"""

from fastapi import APIRouter, Depends, HTTPException, status
from redis import Redis

from src.api.deps import (
    get_redis,
    get_chat_service,
    require_api_key,
    apply_chat_rate_limit,
)
from src.core.logger import get_logger
from src.core.security import generate_session_id
from src.domain.schemas import ChatRequest, ChatResponse, ErrorResponse
from src.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger("endpoint.chat")


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        401: {"model": ErrorResponse, "description": "API key tidak ada"},
        403: {"model": ErrorResponse, "description": "API key tidak valid"},
        422: {"model": ErrorResponse, "description": "Payload tidak valid"},
        429: {"model": ErrorResponse, "description": "Rate limit terlampaui"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Generate respon chatbot",
    description=(
        "Kirim query siswa dan dapatkan respon dari Tutor AI. "
        "Jika session_id tidak disertakan, sistem akan generate dan mengembalikannya — "
        "**simpan session_id ini** dan sertakan di request berikutnya agar history terjaga."
    ),
)
async def chat(
    req:          ChatRequest,
    redis:        Redis       = Depends(get_redis),
    chat_service: ChatService = Depends(get_chat_service),
    _:            None        = Depends(require_api_key),
) -> ChatResponse:
    # ── Rate limiting per user ────────────────────────────────────────────────
    await apply_chat_rate_limit(user_id=req.user_id, redis=redis)

    # ── Auto-generate session_id jika tidak dikirim ───────────────────────────
    session_id = req.session_id or generate_session_id()
    logger.info(
        f"[POST /chat] user_id='{req.user_id}', session_id='{session_id}', "
        f"query_len={len(req.query)}"
    )

    # ── Generate respon ───────────────────────────────────────────────────────
    response = chat_service.generate_response(
        query      = req.query,
        user_id    = req.user_id,
        session_id = session_id,
    )

    return response
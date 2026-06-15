"""
Schemas untuk endpoint /v1/session/*.

  History : MessageItem, SessionHistoryResponse
  Clear   : SessionClearResponse
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════

class MessageItem(BaseModel):
    """Satu pesan dalam history percakapan."""

    role:      str           = Field(..., description="'human' atau 'ai'")
    content:   str
    timestamp: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    """Respon GET /v1/session/{user_id}/history."""

    user_id:       str
    session_id:    str
    message_count: int
    summary:       Optional[str] = Field(
        default=None,
        description="Ringkasan percakapan lama jika ada.",
    )
    messages:      List[MessageItem]


# ══════════════════════════════════════════════════════════════════════════════
# CLEAR
# ══════════════════════════════════════════════════════════════════════════════

class SessionClearResponse(BaseModel):
    """Respon DELETE /v1/session/{user_id}."""

    user_id: str
    cleared: List[str] = Field(
        ...,
        description="Key-key yang berhasil dihapus dari Redis.",
        examples=[["history", "summary", "entity_cache", "context_cache"]],
    )
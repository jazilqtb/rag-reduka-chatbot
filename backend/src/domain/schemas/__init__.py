"""
Pydantic schemas untuk UTBK Tutor RAG API.

Package ini dipecah per resource (chat, document, session, health) supaya
tiap file tetap fokus. Import flat tetap didukung untuk backward compat:

    from src.domain.schemas import ChatRequest, DocumentItem, ErrorResponse

Tetap berfungsi meskipun class-nya sekarang ada di submodul.

Konvensi penamaan:
  *Request  → payload yang dikirim client ke RAG service
  *Response → payload yang dikembalikan RAG service ke client
  *Item     → sub-schema yang dipakai di dalam response lain
"""

# ── Common ──────────────────────────────────────────────────────────────────
from src.domain.schemas.common import ErrorResponse

# ── Chat ────────────────────────────────────────────────────────────────────
from src.domain.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ResponseMeta,
    SourceItem,
)

# ── Document ────────────────────────────────────────────────────────────────
from src.domain.schemas.document import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
    IngestJobResponse,
    IngestJobStatusResponse,
    IngestRequest,
)

# ── Session ─────────────────────────────────────────────────────────────────
from src.domain.schemas.session import (
    MessageItem,
    SessionClearResponse,
    SessionHistoryResponse,
)

# ── Health ──────────────────────────────────────────────────────────────────
from src.domain.schemas.health import (
    ComponentStatus,
    HealthDetailedResponse,
    HealthResponse,
)

# ── Validators (jarang di-import, tapi tersedia jika butuh) ─────────────────
from src.domain.schemas.validators import (
    RE_FILE_ID,
    RE_FILE_NAME,
    RE_JOB_ID,
    RE_SESSION_ID,
    RE_USER_ID,
    validate_file_id,
    validate_filename,
    validate_job_id,
    validate_session_id,
    validate_user_id,
)


__all__ = [
    # Common
    "ErrorResponse",
    # Chat
    "ChatRequest", "ChatResponse", "ResponseMeta", "SourceItem",
    # Document
    "DocumentDeleteResponse", "DocumentItem", "DocumentListResponse",
    "DocumentUploadResponse", "IngestJobResponse", "IngestJobStatusResponse",
    "IngestRequest",
    # Session
    "MessageItem", "SessionClearResponse", "SessionHistoryResponse",
    # Health
    "ComponentStatus", "HealthDetailedResponse", "HealthResponse",
    # Validators
    "RE_FILE_ID", "RE_FILE_NAME", "RE_JOB_ID", "RE_SESSION_ID", "RE_USER_ID",
    "validate_file_id", "validate_filename", "validate_job_id",
    "validate_session_id", "validate_user_id",
]
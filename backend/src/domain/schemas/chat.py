"""
Schemas untuk endpoint /v1/chat.

  ChatRequest   - payload masuk dari client
  ChatResponse  - payload keluar ke client
  SourceItem    - sub-schema citation metadata
  ResponseMeta  - sub-schema performance metadata
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from .validators import validate_user_id, validate_session_id


# ══════════════════════════════════════════════════════════════════════════════
# SUB-SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SourceItem(BaseModel):
    """Metadata dokumen sumber yang dipakai AI sebagai referensi."""

    subject:     str = Field(..., description="Mata pelajaran. Contoh: Penalaran Umum")
    jenis_ujian: str = Field(..., description="Jenis ujian. Contoh: Tryout 1")
    id_soal:     str = Field(..., description="Nomor soal. Contoh: '3'")
    source:      str = Field(..., description="Nama file PDF sumber.")


class ResponseMeta(BaseModel):
    """Metadata performa yang dikembalikan bersama setiap respon chat."""

    latency_ms: int = Field(..., description="Waktu proses total dalam milidetik.")


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Payload dari client ke RAG untuk generate respon chatbot."""

    user_id: str = Field(
        ...,
        description="ID unik siswa. Format: usr_{alphanum_underscore}, 4-53 char.",
        examples=["usr_student001"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "ID sesi percakapan. Format: sess_{alphanum_underscore}. "
            "Jika tidak diisi, sistem generate otomatis dan dikembalikan di response."
        ),
        examples=["sess_abc123xyz"],
    )
    query: str = Field(
        ...,
        description="Pertanyaan siswa terkait soal UTBK.",
        min_length=2,
        max_length=2000,
        examples=["Jelaskan soal nomor 3 penalaran umum kak"],
    )

    @field_validator("user_id")
    @classmethod
    def check_user_id(cls, v: str) -> str:
        return validate_user_id(v)

    @field_validator("session_id")
    @classmethod
    def check_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return validate_session_id(v)

    @field_validator("query")
    @classmethod
    def check_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query tidak boleh berisi hanya whitespace.")
        return v


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE
# ══════════════════════════════════════════════════════════════════════════════

class ChatResponse(BaseModel):
    """Respon lengkap dari RAG ke client."""

    session_id: str = Field(
        ...,
        description="ID sesi — kembalikan ke client untuk request berikutnya.",
    )
    answer:  str = Field(..., description="Teks jawaban dari Tutor AI.")
    sources: List[SourceItem] = Field(
        default=[],
        description="Daftar sumber referensi yang dipakai.",
    )
    meta: Optional[ResponseMeta] = Field(default=None)
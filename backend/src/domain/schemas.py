"""
Semua Pydantic schema untuk request (inbound) dan response (outbound).
 
Konvensi penamaan:
  *Request  → payload yang dikirim BE ke RAG service
  *Response → payload yang dikembalikan RAG service ke BE
  *Item     → sub-schema yang dipakai di dalam response lain
"""
 
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# ══════════════════════════════════════════════════════════════════════════════
# REGEX VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
 
_RE_USER_ID    = re.compile(r"^usr_[a-zA-Z0-9_]{3,49}$")
_RE_SESSION_ID = re.compile(r"^sess_[a-zA-Z0-9_]{4,49}$")
_RE_FILE_NAME  = re.compile(r"^(soal|jawaban)_[a-zA-Z0-9_]{1,50}\.pdf$")
_RE_FILE_ID    = re.compile(r"^file_[a-zA-Z0-9_]{1,80}$")
_RE_JOB_ID     = re.compile(r"^job_[a-zA-Z0-9_]{1,80}$")


def _validate_user_id(v: str) -> str:
    v = v.strip()
    if not _RE_USER_ID.match(v):
        raise ValueError(
            "user_id harus format 'usr_<alphanum_underscore>', panjang 4-53 karakter. "
            f"Contoh: usr_student123. Diterima: '{v}'"
        )
    return v
 
 
def _validate_session_id(v: str) -> str:
    v = v.strip()
    if not _RE_SESSION_ID.match(v):
        raise ValueError(
            "session_id harus format 'sess_<alphanum_underscore>', panjang 5-54 karakter. "
            f"Contoh: sess_abc123. Diterima: '{v}'"
        )
    return v

class SourceItem(BaseModel):
    """Metadata dokumen sumber yang dipakai AI sebagai referensi."""
    subject:     str = Field(..., description="Mata pelajaran. Contoh: Penalaran Umum")
    jenis_ujian: str = Field(..., description="Jenis ujian. Contoh: Tryout 1")
    id_soal:     str = Field(..., description="Nomor soal. Contoh: '3'")
    source:      str = Field(..., description="Nama file PDF sumber.")

class ChatRequest(BaseModel):
    """Payload dari BE ke RAG untuk generate respon chatbot."""
 
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
        return _validate_user_id(v)
 
    @field_validator("session_id")
    @classmethod
    def check_session_id(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return _validate_session_id(v)
 
    @field_validator("query")
    @classmethod
    def check_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query tidak boleh berisi hanya whitespace.")
        return v
    
class ResponseMeta(BaseModel):
    """Metadata performa yang dikembalikan bersama setiap respon chat."""
 
    latency_ms: int = Field(..., description="Waktu proses total dalam milidetik.")
 
 
class ChatResponse(BaseModel):
    """Respon lengkap dari RAG ke BE."""
 
    session_id: str = Field(..., description="ID sesi — kembalikan ke BE untuk request berikutnya.")
    answer:     str = Field(..., description="Teks jawaban dari Tutor AI.")
    sources:    List[SourceItem]  = Field(default=[], description="Daftar sumber referensi yang dipakai.")
    meta:       Optional[ResponseMeta] = Field(default=None)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT — Upload
# ══════════════════════════════════════════════════════════════════════════════
 
class DocumentUploadResponse(BaseModel):
    """Respon setelah file berhasil diupload."""
 
    file_id:     str = Field(..., description="ID unik file. Simpan untuk referensi ingest/delete.")
    filename:    str = Field(..., description="Nama file yang disimpan di server.")
    doc_type:    str = Field(..., description="'soal' atau 'jawaban'.")
    jenis_ujian: str = Field(..., description="Label ujian yang diberikan saat upload.")
    size_bytes:  int = Field(..., description="Ukuran file dalam bytes.")
    status:      str = Field(default="uploaded")
    message:     str = Field(
        default="File berhasil diupload. Panggil POST /v1/documents/ingest untuk memproses."
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT — Ingest
# ══════════════════════════════════════════════════════════════════════════════
 
class IngestRequest(BaseModel):
    """
    Payload untuk memulai proses ingestion.
    Minimal salah satu dari file_ids atau ingest_all_pending harus aktif.
    """
 
    file_ids:            List[str] = Field(
        default=[],
        description="Daftar file_id yang ingin diingest. Kosongkan jika pakai ingest_all_pending.",
    )
    ingest_all_pending:  bool = Field(
        default=False,
        description="Jika true, proses semua file yang belum diingest (file_ids diabaikan).",
    )
 
    @model_validator(mode="after")
    def check_at_least_one(self) -> "IngestRequest":
        if not self.file_ids and not self.ingest_all_pending:
            raise ValueError(
                "Isi minimal salah satu: 'file_ids' (list file_id) "
                "atau 'ingest_all_pending: true'."
            )
        return self
 
    @field_validator("file_ids")
    @classmethod
    def check_file_ids(cls, v: List[str]) -> List[str]:
        for fid in v:
            if not _RE_FILE_ID.match(fid.strip()):
                raise ValueError(
                    f"Format file_id tidak valid: '{fid}'. "
                    f"Harus format 'file_<alphanum_underscore>'."
                )
        return [fid.strip() for fid in v]
 
 
class IngestJobResponse(BaseModel):
    """Respon saat job ingestion berhasil dibuat (async)."""
 
    job_id:             str = Field(..., description="ID job. Gunakan untuk polling status via GET /v1/documents/ingest/{job_id}.")
    status:             str = Field(default="processing")
    files_queued:       int = Field(..., description="Jumlah file yang akan diproses.")
    estimated_seconds:  int = Field(default=120, description="Estimasi durasi proses (detik).")
 
 
class IngestJobStatusResponse(BaseModel):
    """Status job ingestion saat di-polling."""
 
    job_id:           str
    status:           str = Field(..., description="'processing' | 'done' | 'failed'")
    files_queued:     int
    files_processed:  int = Field(default=0)
    files_failed:     int = Field(default=0)
    errors:           List[str] = Field(default=[])
    created_at:       str
    completed_at:     Optional[str] = None
 
 
# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT — List & Delete
# ══════════════════════════════════════════════════════════════════════════════
 
class DocumentItem(BaseModel):
    """Metadata satu file dokumen dalam listing."""
 
    file_id:      str
    filename:     str
    doc_type:     str = Field(..., description="'soal' atau 'jawaban'")
    jenis_ujian:  str
    size_bytes:   int
    ingested:     bool
    uploaded_at:  str
    ingested_at:  Optional[str] = None
    chunk_count:  int = Field(default=0, description="Jumlah soal/chunk yang berhasil diingest.")
 
 
class DocumentListResponse(BaseModel):
    total:  int
    page:   int
    limit:  int
    items:  List[DocumentItem]
 
 
class DocumentDeleteResponse(BaseModel):
    file_id:                str
    deleted_from_storage:   bool
    deleted_from_vectordb:  bool
    chunks_removed:         int
    message:                str
 
 
# ══════════════════════════════════════════════════════════════════════════════
# SESSION
# ══════════════════════════════════════════════════════════════════════════════
 
class MessageItem(BaseModel):
    """Satu pesan dalam history percakapan."""
 
    role:       str  = Field(..., description="'human' atau 'ai'")
    content:    str
    timestamp:  Optional[str] = None
 
 
class SessionHistoryResponse(BaseModel):
    user_id:       str
    session_id:    str
    message_count: int
    summary:       Optional[str] = Field(default=None, description="Ringkasan percakapan lama jika ada.")
    messages:      List[MessageItem]
 
 
class SessionClearResponse(BaseModel):
    user_id: str
    cleared: List[str] = Field(
        ...,
        description="Key-key yang berhasil dihapus dari Redis.",
        examples=[["history", "summary", "entity_cache", "context_cache"]],
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════
 
class HealthResponse(BaseModel):
    status:    str = Field(..., description="'ok' | 'degraded' | 'down'")
    timestamp: str
 
 
class ComponentStatus(BaseModel):
    status:     str = Field(..., description="'ok' | 'error'")
    latency_ms: Optional[int]   = None
    detail:     Optional[str]   = None
 
 
class HealthDetailedResponse(BaseModel):
    status:     str
    timestamp:  str
    components: Dict[str, ComponentStatus]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# ERROR
# ══════════════════════════════════════════════════════════════════════════════
 
class ErrorResponse(BaseModel):
    """Format error standar untuk semua endpoint."""
 
    error:   str = Field(..., description="Kode error singkat. Contoh: 'validation_error'")
    message: str = Field(..., description="Penjelasan human-readable.")
    detail:  Optional[Any] = Field(default=None, description="Detail tambahan jika ada (misal: field yang gagal validasi).")
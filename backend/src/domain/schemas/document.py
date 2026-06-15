"""
Schemas untuk endpoint /v1/documents/*.

  Upload   : DocumentUploadResponse
  Ingest   : IngestRequest, IngestJobResponse, IngestJobStatusResponse
  List     : DocumentItem, DocumentListResponse
  Delete   : DocumentDeleteResponse
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from .validators import RE_FILE_ID


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
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
        default="File berhasil diupload. Panggil POST /v1/documents/ingest untuk memproses.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════════════════

class IngestRequest(BaseModel):
    """
    Payload untuk memulai proses ingestion.
    Minimal salah satu dari file_ids atau ingest_all_pending harus aktif.
    """

    file_ids: List[str] = Field(
        default=[],
        description="Daftar file_id yang ingin diingest. Kosongkan jika pakai ingest_all_pending.",
    )
    ingest_all_pending: bool = Field(
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
            if not RE_FILE_ID.match(fid.strip()):
                raise ValueError(
                    f"Format file_id tidak valid: '{fid}'. "
                    "Harus format 'file_<alphanum_underscore>'."
                )
        return [fid.strip() for fid in v]


class IngestJobResponse(BaseModel):
    """Respon saat job ingestion berhasil dibuat (async)."""

    job_id:            str = Field(
        ...,
        description="ID job. Gunakan untuk polling status via GET /v1/documents/ingest/{job_id}.",
    )
    status:            str = Field(default="processing")
    files_queued:      int = Field(..., description="Jumlah file yang akan diproses.")
    estimated_seconds: int = Field(default=120, description="Estimasi durasi proses (detik).")


class IngestJobStatusResponse(BaseModel):
    """Status job ingestion saat di-polling."""

    job_id:          str
    status:          str       = Field(..., description="'processing' | 'done' | 'failed'")
    files_queued:    int
    files_processed: int       = Field(default=0)
    files_failed:    int       = Field(default=0)
    errors:          List[str] = Field(default=[])
    created_at:      str
    completed_at:    Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# LIST & DELETE
# ══════════════════════════════════════════════════════════════════════════════

class DocumentItem(BaseModel):
    """Metadata satu file dokumen dalam listing."""

    file_id:     str
    filename:    str
    doc_type:    str = Field(..., description="'soal' atau 'jawaban'")
    jenis_ujian: str
    size_bytes:  int
    ingested:    bool
    uploaded_at: str
    ingested_at: Optional[str] = None
    chunk_count: int = Field(default=0, description="Jumlah soal/chunk yang berhasil diingest.")


class DocumentListResponse(BaseModel):
    total: int
    page:  int
    limit: int
    items: List[DocumentItem]


class DocumentDeleteResponse(BaseModel):
    file_id:               str
    deleted_from_storage:  bool
    deleted_from_vectordb: bool
    chunks_removed:        int
    message:               str
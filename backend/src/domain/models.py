"""
SQLAlchemy ORM models untuk PostgreSQL.

IMPORTANT: File `infra/postgres/init.sql` adalah SOURCE OF TRUTH untuk schema
database. Model di sini HANYA mirror untuk operasi read/write — tidak pernah
dipakai untuk create_all().

Constraint CHECK yang ada di __table_args__ adalah mirror dari init.sql untuk:
  - Dokumentasi langsung di kode Python
  - Enforcement tambahan jika ada developer yang test pakai create_all()
    di SQLite/in-memory untuk unit test

Selalu update file ini DAN init.sql secara konsisten saat schema berubah.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.db.base import Base


# ══════════════════════════════════════════════════════════════════════════════
# Document
# ══════════════════════════════════════════════════════════════════════════════

class Document(Base):
    """
    Metadata persisten untuk file PDF yang diupload (soal/jawaban UTBK).
    Menggantikan Redis `doc:meta:{file_id}` HASH.
    """

    __tablename__ = "documents"

    file_id:           Mapped[str] = mapped_column(String(80),  primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path:       Mapped[str] = mapped_column(Text,        nullable=False)
    file_type:         Mapped[str] = mapped_column(String(20),  nullable=False)
    jenis_ujian:       Mapped[str] = mapped_column(String(100), nullable=False)
    mime_type:         Mapped[str] = mapped_column(String(50),  nullable=False, default="application/pdf")
    size_bytes:        Mapped[int] = mapped_column(BigInteger,  nullable=False)
    status:            Mapped[str] = mapped_column(String(20),  nullable=False, default="uploaded")
    chunk_count:       Mapped[int] = mapped_column(Integer,     nullable=False, default=0)
    error_message:     Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    uploaded_at:       Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    ingested_at:       Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_at:        Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            r"file_id ~ '^file_[a-zA-Z0-9_]+$'",
            name="documents_file_id_format",
        ),
        CheckConstraint(
            "file_type IN ('soal', 'jawaban')",
            name="documents_file_type_valid",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'ingested', 'failed', 'deleted')",
            name="documents_status_valid",
        ),
        CheckConstraint("size_bytes > 0",   name="documents_size_positive"),
        CheckConstraint("chunk_count >= 0", name="documents_chunk_count_nonneg"),
        Index("idx_documents_uploaded_at", "uploaded_at"),
        Index("idx_documents_file_type",   "file_type"),
        Index("idx_documents_jenis_ujian", "jenis_ujian"),
    )

    def __repr__(self) -> str:
        return (
            f"<Document(file_id={self.file_id!r}, type={self.file_type!r}, "
            f"jenis={self.jenis_ujian!r}, status={self.status!r}, "
            f"chunks={self.chunk_count})>"
        )

    @property
    def is_ingested(self) -> bool:
        """Convenience flag — backward compat dengan response schema."""
        return self.status == "ingested"


# ══════════════════════════════════════════════════════════════════════════════
# IngestJob
# ══════════════════════════════════════════════════════════════════════════════

class IngestJob(Base):
    """
    Audit trail untuk job ingestion async (PDF -> ChromaDB pipeline).
    Menggantikan Redis `ingest:job:{job_id}` HASH.

    Catatan: Mutex untuk satu job berjalan sekaligus TETAP di Redis
    (key `ingest:lock`) karena SET NX lebih cepat & atomic.
    """

    __tablename__ = "ingest_jobs"

    job_id:          Mapped[str]      = mapped_column(String(80), primary_key=True)
    status:          Mapped[str]      = mapped_column(String(20), nullable=False, default="pending")
    file_ids:        Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    total_files:     Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    processed_files: Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    failed_files:    Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    total_chunks:    Mapped[int]      = mapped_column(Integer, nullable=False, default=0)
    errors:          Mapped[List[str]] = mapped_column(JSONB, nullable=False, default=list)
    error_message:   Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at:      Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    completed_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            r"job_id ~ '^job_[a-zA-Z0-9_]+$'",
            name="ingest_jobs_job_id_format",
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ingest_jobs_status_valid",
        ),
        CheckConstraint(
            "total_files >= 0 AND processed_files >= 0 "
            "AND failed_files >= 0 AND total_chunks >= 0",
            name="ingest_jobs_counts_nonneg",
        ),
        Index("idx_ingest_jobs_status",     "status"),
        Index("idx_ingest_jobs_started_at", "started_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<IngestJob(job_id={self.job_id!r}, status={self.status!r}, "
            f"progress={self.processed_files}/{self.total_files})>"
        )
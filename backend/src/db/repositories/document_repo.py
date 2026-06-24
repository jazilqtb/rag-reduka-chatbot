"""
Repository untuk tabel `documents`.

Pattern: tiap method commit per call (autonomous). Kalau butuh batch
transaksi (multiple writes atomic), gunakan session langsung atau wrap
beberapa call dalam transactional_session().

Untuk read-only methods (get/list), tidak ada commit.
"""

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.domain.models import Document


class DocumentRepository:
    """CRUD operations untuk Document."""

    def __init__(self, session: Session):
        self.session = session

    # ── CREATE ──────────────────────────────────────────────────────────────

    def create(
        self,
        *,
        file_id: str,
        original_filename: str,
        stored_path: str,
        file_type: str,
        jenis_ujian: str,
        size_bytes: int,
        mime_type: str = "application/pdf",
    ) -> Document:
        """
        Insert new document row. Commits on success.

        Raises:
            sqlalchemy.exc.IntegrityError: jika file_id duplikat, melanggar
                                            CHECK constraint, atau filename
                                            sudah ada (unique partial index).
        """
        doc = Document(
            file_id           = file_id,
            original_filename = original_filename,
            stored_path       = stored_path,
            file_type         = file_type,
            jenis_ujian       = jenis_ujian,
            mime_type         = mime_type,
            size_bytes        = size_bytes,
            status            = "uploaded",
            chunk_count       = 0,
        )
        self.session.add(doc)
        self.session.commit()
        self.session.refresh(doc)
        return doc

    # ── READ ────────────────────────────────────────────────────────────────

    def get_by_id(self, file_id: str, include_deleted: bool = False) -> Optional[Document]:
        """Get single document by file_id. Returns None if not found / soft-deleted."""
        doc = self.session.get(Document, file_id)
        if doc is None:
            return None
        if not include_deleted and doc.deleted_at is not None:
            return None
        return doc

    def find_by_filename(
        self,
        filename: str,
        include_deleted: bool = False,
    ) -> Optional[Document]:
        """
        Find document by original_filename. Returns None if not found.
        Berguna untuk cek duplikat saat upload.
        """
        stmt = select(Document).where(Document.original_filename == filename)
        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))
        return self.session.scalar(stmt)

    def list_all(
        self,
        *,
        file_type:   Optional[str] = None,
        jenis_ujian: Optional[str] = None,
        status:      Optional[str] = None,
        include_deleted: bool = False,
        page:  int = 1,
        limit: int = 20,
    ) -> Tuple[List[Document], int]:
        """
        List documents with filtering & pagination.

        Returns:
            (documents, total_count)
        """
        stmt = select(Document)

        if not include_deleted:
            stmt = stmt.where(Document.deleted_at.is_(None))
        if file_type is not None:
            stmt = stmt.where(Document.file_type == file_type)
        if jenis_ujian is not None:
            stmt = stmt.where(Document.jenis_ujian == jenis_ujian)
        if status is not None:
            stmt = stmt.where(Document.status == status)

        # Count total (clone the query without order/limit/offset)
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = self.session.scalar(count_stmt) or 0

        # Apply pagination + sorting
        offset = (page - 1) * limit
        stmt   = stmt.order_by(Document.uploaded_at.desc()).limit(limit).offset(offset)
        docs   = list(self.session.scalars(stmt).all())

        return docs, total

    def list_pending_soal(self) -> List[Document]:
        """
        Convenience: list file 'soal' yang belum diingest.
        Dipakai oleh endpoint /v1/documents/ingest dengan ingest_all_pending=True.
        """
        stmt = (
            select(Document)
            .where(Document.deleted_at.is_(None))
            .where(Document.file_type == "soal")
            .where(Document.status == "uploaded")
            .order_by(Document.uploaded_at.asc())
        )
        return list(self.session.scalars(stmt).all())

    def count_by_status(self, status: str) -> int:
        """Count active documents with given status."""
        stmt = (
            select(func.count())
            .select_from(Document)
            .where(Document.deleted_at.is_(None))
            .where(Document.status == status)
        )
        return self.session.scalar(stmt) or 0

    # ── UPDATE ──────────────────────────────────────────────────────────────

    def update_status(
        self,
        file_id: str,
        *,
        status: str,
        chunk_count:   Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Document]:
        """
        Update status of a document. Sets ingested_at automatically when
        transitioning to 'ingested'.

        Returns:
            Updated Document, or None if not found / already soft-deleted.
        """
        doc = self.get_by_id(file_id)
        if doc is None:
            return None

        doc.status = status
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        if error_message is not None:
            doc.error_message = error_message
        if status == "ingested" and doc.ingested_at is None:
            doc.ingested_at = datetime.now(timezone.utc)

        self.session.commit()
        self.session.refresh(doc)
        return doc

    # ── DELETE ──────────────────────────────────────────────────────────────

    def soft_delete(self, file_id: str) -> Optional[Document]:
        """
        Mark document as deleted (sets deleted_at + status='deleted').
        Row tetap ada di DB untuk audit.

        Returns:
            Deleted Document, or None if not found.
        """
        doc = self.session.get(Document, file_id)
        if doc is None or doc.deleted_at is not None:
            return None

        doc.deleted_at = datetime.now(timezone.utc)
        doc.status     = "deleted"
        self.session.commit()
        self.session.refresh(doc)
        return doc

    def hard_delete(self, file_id: str) -> bool:
        """
        Permanently remove document row from DB.
        Gunakan ini hanya untuk cleanup / testing.

        Returns:
            True jika row dihapus, False jika tidak ditemukan.
        """
        doc = self.session.get(Document, file_id)
        if doc is None:
            return False
        self.session.delete(doc)
        self.session.commit()
        return True
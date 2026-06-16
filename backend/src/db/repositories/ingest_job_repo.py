"""
Repository untuk tabel `ingest_jobs`.

Job ingestion punya state machine:
    pending -> running -> completed | failed | cancelled

Method update_progress() handle update partial — pass hanya field yang berubah,
field lain biarkan None (tidak akan ditimpa).
"""

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.domain.models import IngestJob


class IngestJobRepository:
    """CRUD operations untuk IngestJob."""

    # State machine terminal — kalau status di-set ke salah satu ini,
    # completed_at di-stempel otomatis.
    TERMINAL_STATES = ("completed", "failed", "cancelled")

    def __init__(self, session: Session):
        self.session = session

    # ── CREATE ──────────────────────────────────────────────────────────────

    def create(self, *, job_id: str, file_ids: List[str]) -> IngestJob:
        """
        Create new ingest job with status='pending'.

        Raises:
            sqlalchemy.exc.IntegrityError: jika job_id duplikat.
        """
        job = IngestJob(
            job_id      = job_id,
            status      = "pending",
            file_ids    = file_ids,
            total_files = len(file_ids),
        )
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    # ── READ ────────────────────────────────────────────────────────────────

    def get_by_id(self, job_id: str) -> Optional[IngestJob]:
        """Get single job by job_id. Returns None if not found."""
        return self.session.get(IngestJob, job_id)

    def list_recent(self, limit: int = 20) -> List[IngestJob]:
        """List recent jobs ordered by started_at descending."""
        stmt = (
            select(IngestJob)
            .order_by(IngestJob.started_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())

    def list_active(self) -> List[IngestJob]:
        """List jobs currently in pending or running state."""
        stmt = (
            select(IngestJob)
            .where(IngestJob.status.in_(("pending", "running")))
            .order_by(IngestJob.started_at.asc())
        )
        return list(self.session.scalars(stmt).all())

    # ── UPDATE ──────────────────────────────────────────────────────────────

    def update_progress(
        self,
        job_id: str,
        *,
        status:          Optional[str]       = None,
        processed_files: Optional[int]       = None,
        failed_files:    Optional[int]       = None,
        total_chunks:    Optional[int]       = None,
        errors:          Optional[List[str]] = None,
        error_message:   Optional[str]       = None,
    ) -> Optional[IngestJob]:
        """
        Partial update — hanya field yang dikirim (non-None) yang di-update.

        Saat status berubah ke terminal state (completed/failed/cancelled),
        `completed_at` otomatis di-set ke NOW().

        Returns:
            Updated IngestJob, atau None jika job_id tidak ditemukan.
        """
        job = self.get_by_id(job_id)
        if job is None:
            return None

        if status is not None:
            job.status = status
            if status in self.TERMINAL_STATES and job.completed_at is None:
                job.completed_at = datetime.now(timezone.utc)

        if processed_files is not None:
            job.processed_files = processed_files
        if failed_files is not None:
            job.failed_files = failed_files
        if total_chunks is not None:
            job.total_chunks = total_chunks
        if errors is not None:
            job.errors = errors
        if error_message is not None:
            job.error_message = error_message

        self.session.commit()
        self.session.refresh(job)
        return job

    def increment_progress(
        self,
        job_id: str,
        *,
        processed_delta: int = 0,
        failed_delta:    int = 0,
        chunks_delta:    int = 0,
        new_errors:      Optional[List[str]] = None,
    ) -> Optional[IngestJob]:
        """
        Atomic-ish increment counter (read-modify-write within same session).

        Dipakai dari background worker yang process file satu per satu.
        Bukan benar-benar atomic di level DB — kalau butuh strict concurrent,
        pakai SELECT ... FOR UPDATE atau UPDATE ... SET x = x + N.

        Returns:
            Updated IngestJob, atau None jika job_id tidak ditemukan.
        """
        job = self.get_by_id(job_id)
        if job is None:
            return None

        if processed_delta:
            job.processed_files = (job.processed_files or 0) + processed_delta
        if failed_delta:
            job.failed_files = (job.failed_files or 0) + failed_delta
        if chunks_delta:
            job.total_chunks = (job.total_chunks or 0) + chunks_delta
        if new_errors:
            job.errors = (job.errors or []) + new_errors

        self.session.commit()
        self.session.refresh(job)
        return job

    # ── DELETE ──────────────────────────────────────────────────────────────

    def hard_delete(self, job_id: str) -> bool:
        """
        Permanently remove job row. Gunakan untuk cleanup/testing.
        Job records biasanya tidak dihapus untuk audit trail.
        """
        job = self.session.get(IngestJob, job_id)
        if job is None:
            return False
        self.session.delete(job)
        self.session.commit()
        return True
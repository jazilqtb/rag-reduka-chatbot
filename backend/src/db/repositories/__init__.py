"""
Repository pattern untuk akses database.

Setiap repository mengabstrak SQL/ORM detail dari service layer.
Service tidak panggil SQLAlchemy langsung — mereka panggil:

    doc_repo = DocumentRepository(db_session)
    doc = doc_repo.create(file_id=..., ...)

Keuntungan:
  - Mudah di-mock untuk unit test
  - Kalau ganti ORM/DB nanti, hanya repository yang berubah
  - Service layer fokus ke business logic
"""

from src.db.repositories.document_repo   import DocumentRepository
from src.db.repositories.ingest_job_repo import IngestJobRepository


__all__ = [
    "DocumentRepository",
    "IngestJobRepository",
]
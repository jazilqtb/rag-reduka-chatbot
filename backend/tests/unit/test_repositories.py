"""
Integration tests untuk repository (DocumentRepository, IngestJobRepository).

Pakai SQLite in-memory via db_session fixture dari conftest.
Setiap test mulai dengan DB bersih (rollback di akhir test).
"""

import pytest
from datetime import datetime, timezone


pytestmark = pytest.mark.integration


# ══════════════════════════════════════════════════════════════════════════════
# DocumentRepository
# ══════════════════════════════════════════════════════════════════════════════

class TestDocumentRepository:

    def test_create_returns_document(self, document_repo):
        doc = document_repo.create(
            file_id           = "file_abc123",
            original_filename = "soal_test.pdf",
            stored_path       = "/data/raw_docs/soal_test.pdf",
            file_type         = "soal",
            jenis_ujian       = "Tryout 1",
            size_bytes        = 1024,
        )
        assert doc.file_id == "file_abc123"
        assert doc.status == "uploaded"
        assert doc.chunk_count == 0
        assert doc.ingested_at is None
        assert doc.deleted_at is None

    def test_get_by_id_returns_none_when_missing(self, document_repo):
        assert document_repo.get_by_id("file_notfound") is None

    def test_get_by_id_returns_doc_when_present(self, document_repo):
        document_repo.create(
            file_id="file_x", original_filename="soal_x.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        doc = document_repo.get_by_id("file_x")
        assert doc is not None
        assert doc.original_filename == "soal_x.pdf"

    def test_find_by_filename(self, document_repo):
        document_repo.create(
            file_id="file_x", original_filename="soal_unique.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        found = document_repo.find_by_filename("soal_unique.pdf")
        assert found is not None
        assert found.file_id == "file_x"

        # Not found returns None
        assert document_repo.find_by_filename("does_not_exist.pdf") is None

    def test_update_status_to_ingested_sets_timestamp(self, document_repo):
        document_repo.create(
            file_id="file_x", original_filename="soal_x.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        updated = document_repo.update_status(
            "file_x", status="ingested", chunk_count=15,
        )
        assert updated.status == "ingested"
        assert updated.chunk_count == 15
        assert updated.ingested_at is not None

    def test_update_status_failed_does_not_set_ingested_at(self, document_repo):
        document_repo.create(
            file_id="file_x", original_filename="soal_x.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        updated = document_repo.update_status(
            "file_x", status="failed", error_message="parse error",
        )
        assert updated.status == "failed"
        assert updated.error_message == "parse error"
        assert updated.ingested_at is None

    def test_soft_delete(self, document_repo):
        document_repo.create(
            file_id="file_x", original_filename="soal_x.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        deleted = document_repo.soft_delete("file_x")
        assert deleted.status == "deleted"
        assert deleted.deleted_at is not None

        # Subsequent get returns None (filtered out)
        assert document_repo.get_by_id("file_x") is None
        # But include_deleted=True still returns
        assert document_repo.get_by_id("file_x", include_deleted=True) is not None

    def test_list_filter_by_type(self, document_repo):
        document_repo.create(
            file_id="file_s1", original_filename="soal_a.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        document_repo.create(
            file_id="file_s2", original_filename="soal_b.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 2", size_bytes=200,
        )
        document_repo.create(
            file_id="file_j1", original_filename="jawaban_a.pdf",
            stored_path="/p", file_type="jawaban",
            jenis_ujian="Tryout 1", size_bytes=50,
        )

        docs, total = document_repo.list_all(file_type="soal")
        assert total == 2
        assert all(d.file_type == "soal" for d in docs)

    def test_list_filter_by_jenis_ujian(self, document_repo):
        document_repo.create(
            file_id="file_a", original_filename="soal_a.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        document_repo.create(
            file_id="file_b", original_filename="soal_b.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 2", size_bytes=100,
        )
        docs, total = document_repo.list_all(jenis_ujian="Tryout 1")
        assert total == 1
        assert docs[0].file_id == "file_a"

    def test_list_pending_soal(self, document_repo):
        # soal uploaded
        document_repo.create(
            file_id="file_s1", original_filename="soal_a.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        # soal already ingested
        document_repo.create(
            file_id="file_s2", original_filename="soal_b.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        document_repo.update_status("file_s2", status="ingested", chunk_count=10)
        # jawaban (not soal) — should not appear
        document_repo.create(
            file_id="file_j", original_filename="jawaban_a.pdf",
            stored_path="/p", file_type="jawaban",
            jenis_ujian="Tryout 1", size_bytes=50,
        )

        pending = document_repo.list_pending_soal()
        assert len(pending) == 1
        assert pending[0].file_id == "file_s1"


# ══════════════════════════════════════════════════════════════════════════════
# IngestJobRepository
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestJobRepository:

    def test_create_job(self, ingest_job_repo):
        job = ingest_job_repo.create(
            job_id   = "job_abc123",
            file_ids = ["file_x", "file_y"],
        )
        assert job.status == "pending"
        assert job.total_files == 2
        assert job.processed_files == 0
        assert job.completed_at is None

    def test_update_to_running_no_completed_timestamp(self, ingest_job_repo):
        ingest_job_repo.create(job_id="job_x", file_ids=["file_a"])
        job = ingest_job_repo.update_progress("job_x", status="running")
        assert job.status == "running"
        assert job.completed_at is None

    def test_update_to_completed_sets_timestamp(self, ingest_job_repo):
        ingest_job_repo.create(job_id="job_x", file_ids=["file_a"])
        job = ingest_job_repo.update_progress(
            "job_x", status="completed", processed_files=1, total_chunks=10,
        )
        assert job.status == "completed"
        assert job.completed_at is not None
        assert job.processed_files == 1
        assert job.total_chunks == 10

    def test_update_to_failed_sets_timestamp(self, ingest_job_repo):
        ingest_job_repo.create(job_id="job_x", file_ids=["file_a"])
        job = ingest_job_repo.update_progress(
            "job_x", status="failed",
            failed_files=1, error_message="oops",
        )
        assert job.status == "failed"
        assert job.completed_at is not None
        assert job.error_message == "oops"

    def test_increment_progress(self, ingest_job_repo):
        ingest_job_repo.create(job_id="job_x", file_ids=["a", "b", "c"])
        ingest_job_repo.increment_progress("job_x", processed_delta=1, chunks_delta=5)
        ingest_job_repo.increment_progress(
            "job_x", processed_delta=1, chunks_delta=8, new_errors=["timeout on b"]
        )
        job = ingest_job_repo.get_by_id("job_x")
        assert job.processed_files == 2
        assert job.total_chunks == 13
        assert job.errors == ["timeout on b"]

    def test_list_active_excludes_terminal(self, ingest_job_repo):
        ingest_job_repo.create(job_id="job_p", file_ids=["x"])  # pending
        ingest_job_repo.create(job_id="job_r", file_ids=["y"])
        ingest_job_repo.update_progress("job_r", status="running")
        ingest_job_repo.create(job_id="job_c", file_ids=["z"])
        ingest_job_repo.update_progress("job_c", status="completed")

        active = ingest_job_repo.list_active()
        assert {j.job_id for j in active} == {"job_p", "job_r"}

    def test_update_nonexistent_returns_none(self, ingest_job_repo):
        result = ingest_job_repo.update_progress("job_nonexistent", status="running")
        assert result is None
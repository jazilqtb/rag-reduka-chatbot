"""
Integration tests untuk endpoint /v1/documents/* — pakai FastAPI TestClient.

Tests cover:
  - Auth (401 / 403)
  - Path parameter validation (422)
  - Happy path GET /v1/documents (empty + with data)
  - Validation errors di upload (400)
  - File_id format validation di DELETE (422)
  - Job_id not found (404)

NOT covered (perlu LLM/ChromaDB real):
  - Full upload flow yang ingest ke ChromaDB
  - Full ingest job execution

Untuk itu pakai marker @pytest.mark.llm (skipped by default).
"""

import io

import pytest


pytestmark = pytest.mark.integration


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

class TestAuth:

    def test_missing_api_key_returns_401(self, test_client):
        r = test_client.get("/v1/documents")
        assert r.status_code == 401
        assert "missing_api_key" in r.text

    def test_invalid_api_key_returns_403(self, test_client):
        r = test_client.get("/v1/documents", headers={"X-API-Key": "wrong"})
        assert r.status_code == 403
        assert "invalid_api_key" in r.text


# ══════════════════════════════════════════════════════════════════════════════
# GET /v1/documents
# ══════════════════════════════════════════════════════════════════════════════

class TestListDocuments:

    def test_empty_returns_zero_total(self, test_client, api_key_header):
        r = test_client.get("/v1/documents", headers=api_key_header)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 0
        assert data["items"] == []
        assert data["page"]  == 1
        assert data["limit"] == 20

    def test_with_data_returns_items(self, test_client, api_key_header, document_repo):
        # Seed test data
        document_repo.create(
            file_id="file_test1", original_filename="soal_a.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        document_repo.create(
            file_id="file_test2", original_filename="jawaban_a.pdf",
            stored_path="/p", file_type="jawaban",
            jenis_ujian="Tryout 1", size_bytes=50,
        )

        r = test_client.get("/v1/documents", headers=api_key_header)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 2
        assert {item["file_id"] for item in data["items"]} == {"file_test1", "file_test2"}

    def test_filter_by_doc_type(self, test_client, api_key_header, document_repo):
        document_repo.create(
            file_id="file_s", original_filename="soal_a.pdf",
            stored_path="/p", file_type="soal",
            jenis_ujian="Tryout 1", size_bytes=100,
        )
        document_repo.create(
            file_id="file_j", original_filename="jawaban_a.pdf",
            stored_path="/p", file_type="jawaban",
            jenis_ujian="Tryout 1", size_bytes=50,
        )

        r = test_client.get(
            "/v1/documents?doc_type=soal", headers=api_key_header
        )
        data = r.json()
        assert data["total"] == 1
        assert data["items"][0]["file_id"] == "file_s"


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/documents/upload (validation only — full flow needs real DB write)
# ══════════════════════════════════════════════════════════════════════════════

class TestUploadValidation:

    def test_invalid_filename_pattern(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/upload",
            headers=api_key_header,
            files={"file": ("namanya_aneh.pdf", b"%PDF-1.4 test", "application/pdf")},
            data ={"doc_type": "soal", "jenis_ujian": "Tryout Test"},
        )
        assert r.status_code == 400
        assert "invalid_filename" in r.text

    def test_invalid_doc_type(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/upload",
            headers=api_key_header,
            files={"file": ("soal_a.pdf", b"%PDF-1.4 test", "application/pdf")},
            data ={"doc_type": "buku", "jenis_ujian": "Tryout 1"},
        )
        assert r.status_code == 400
        assert "invalid_doc_type" in r.text

    def test_doctype_filename_mismatch(self, test_client, api_key_header):
        # filename starts with "soal_" but doc_type="jawaban"
        r = test_client.post(
            "/v1/documents/upload",
            headers=api_key_header,
            files={"file": ("soal_a.pdf", b"%PDF-1.4 test", "application/pdf")},
            data ={"doc_type": "jawaban", "jenis_ujian": "Tryout 1"},
        )
        assert r.status_code == 400
        assert "doctype_filename_mismatch" in r.text

    def test_non_pdf_content(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/upload",
            headers=api_key_header,
            files={"file": ("soal_a.pdf", b"this is not a real pdf", "application/pdf")},
            data ={"doc_type": "soal", "jenis_ujian": "Tryout 1"},
        )
        assert r.status_code == 400
        assert "invalid_file_type" in r.text

    def test_empty_jenis_ujian(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/upload",
            headers=api_key_header,
            files={"file": ("soal_a.pdf", b"%PDF-1.4 test", "application/pdf")},
            data ={"doc_type": "soal", "jenis_ujian": "   "},
        )
        assert r.status_code == 400
        assert "invalid_jenis_ujian" in r.text


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /v1/documents/{file_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestDeleteDocument:

    def test_invalid_file_id_format_returns_422(self, test_client, api_key_header):
        r = test_client.delete(
            "/v1/documents/bad-format", headers=api_key_header,
        )
        assert r.status_code == 422
        assert "invalid_file_id" in r.text

    def test_file_id_not_found_returns_404(self, test_client, api_key_header):
        r = test_client.delete(
            "/v1/documents/file_notexist123", headers=api_key_header,
        )
        assert r.status_code == 404
        assert "file_not_found" in r.text


# ══════════════════════════════════════════════════════════════════════════════
# GET /v1/documents/ingest/{job_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestStatus:

    def test_invalid_job_id_format(self, test_client, api_key_header):
        r = test_client.get(
            "/v1/documents/ingest/badformat", headers=api_key_header,
        )
        assert r.status_code == 422

    def test_job_not_found(self, test_client, api_key_header):
        r = test_client.get(
            "/v1/documents/ingest/job_notexist123", headers=api_key_header,
        )
        assert r.status_code == 404
        assert "job_not_found" in r.text


# ══════════════════════════════════════════════════════════════════════════════
# POST /v1/documents/ingest
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestRequest:

    def test_empty_body_returns_422(self, test_client, api_key_header):
        # Both file_ids empty AND ingest_all_pending=False → schema validation
        r = test_client.post(
            "/v1/documents/ingest", headers=api_key_header,
            json={"file_ids": [], "ingest_all_pending": False},
        )
        assert r.status_code == 422

    def test_unknown_file_id_returns_404(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/ingest", headers=api_key_header,
            json={"file_ids": ["file_doesnotexist"]},
        )
        assert r.status_code == 404
        assert "file_not_found" in r.text

    def test_ingest_all_pending_with_no_files(self, test_client, api_key_header):
        r = test_client.post(
            "/v1/documents/ingest", headers=api_key_header,
            json={"ingest_all_pending": True},
        )
        # No 'soal' uploaded yet → no_pending_files (400)
        assert r.status_code == 400
        assert "no_pending_files" in r.text
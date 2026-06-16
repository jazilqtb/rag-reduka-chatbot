"""
Endpoints: Document Management
  POST   /v1/documents/upload            — Upload file PDF soal/jawaban
  POST   /v1/documents/ingest            — Mulai job ingestion ke ChromaDB (async)
  GET    /v1/documents/ingest/{job_id}   — Polling status job ingestion
  GET    /v1/documents                   — List semua dokumen terdaftar
  DELETE /v1/documents/{file_id}         — Hapus dokumen dari storage + ChromaDB

Catatan IngestionService:
  IngestionService.run() saat ini memproses SEMUA file di raw_docs dan me-reset
  ChromaDB secara penuh (shutil.rmtree di __init__). Ini adalah desain existing.
  Implikasinya: selama job ingestion berlangsung (~1-3 menit), query chatbot
  mungkin mendapat hasil kosong dari ChromaDB. Client perlu handle kondisi ini.
  Rekomendasi future: modifikasi IngestionService untuk incremental ingestion.

Stage 4 changes:
  - Metadata document & ingest job DIPINDAHKAN dari Redis HASH ke PostgreSQL.
  - Mutex ingest tetap di Redis (key: `ingest:lock`) karena SET NX lebih cepat.
  - Helper Redis (_get_doc_meta, _save_doc_meta, dst) dihapus.
  - Status mapping: DB pakai pending/running/completed/failed, API tetap
    pakai processing/done/failed untuk backward compat dengan client.
"""

import os
import json
import re
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from redis import Redis

from src.api.deps import (
    get_document_repository,
    get_ingest_job_repository,
    get_redis,
    require_api_key,
    valid_file_id,
    valid_job_id,
)
from src.core.config import settings
from src.core.logger import get_logger
from src.core.security import generate_file_id, generate_job_id
from src.db.repositories import DocumentRepository, IngestJobRepository
from src.domain.models import Document, IngestJob
from src.domain.schemas import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
    IngestJobResponse,
    IngestJobStatusResponse,
    IngestRequest,
)

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger("endpoint.document")

# ── Redis key (mutex saja, metadata sudah pindah ke Postgres) ─────────────────
_INGEST_LOCK_KEY = "ingest:lock"

# ── Validasi ──────────────────────────────────────────────────────────────────
_MAX_SIZE    = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
_RE_FILENAME = re.compile(r"^(soal|jawaban)_[a-zA-Z0-9_]{1,50}\.pdf$")


# ══════════════════════════════════════════════════════════════════════════════
# MAPPERS: ORM model -> Pydantic response schema
# ══════════════════════════════════════════════════════════════════════════════

# Mapping status DB <-> API
# DB pakai state machine yang lebih kaya, API pakai 3 state agar backward
# compatible dengan client lama.
_DB_TO_API_STATUS = {
    "pending":    "processing",
    "running":    "processing",
    "completed":  "done",
    "failed":     "failed",
    "cancelled":  "failed",
}


def _doc_to_item(doc: Document) -> DocumentItem:
    """Konversi Document ORM ke DocumentItem response schema."""
    return DocumentItem(
        file_id     = doc.file_id,
        filename    = doc.original_filename,
        doc_type    = doc.file_type,
        jenis_ujian = doc.jenis_ujian,
        size_bytes  = doc.size_bytes,
        ingested    = doc.is_ingested,
        uploaded_at = doc.uploaded_at.isoformat() if doc.uploaded_at else "",
        ingested_at = doc.ingested_at.isoformat() if doc.ingested_at else None,
        chunk_count = doc.chunk_count,
    )


def _job_to_status_response(job: IngestJob) -> IngestJobStatusResponse:
    """Konversi IngestJob ORM ke IngestJobStatusResponse."""
    return IngestJobStatusResponse(
        job_id          = job.job_id,
        status          = _DB_TO_API_STATUS.get(job.status, job.status),
        files_queued    = job.total_files,
        files_processed = job.processed_files,
        files_failed    = job.failed_files,
        errors          = list(job.errors or []),
        created_at      = job.started_at.isoformat() if job.started_at else "",
        completed_at    = job.completed_at.isoformat() if job.completed_at else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_chunk_count_from_debug(filename: str) -> int:
    """
    Hitung jumlah soal (chunk) dari file debug JSON yang dihasilkan IngestionService.
    File debug: data/debug/debug_{stem}.json

    TODO Stage 5: IngestionService harus mengembalikan chunk_count langsung
    sehingga tidak perlu re-parse debug file.
    """
    try:
        stem       = filename.replace(".pdf", "")
        debug_path = settings.DATA_DIR / "debug" / f"debug_{stem}.json"
        if not debug_path.exists():
            return 0
        with open(debug_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND INGESTION JOB
# ══════════════════════════════════════════════════════════════════════════════

def _run_ingestion_job(job_id: str, file_ids: List[str], redis: Redis) -> None:
    """
    Jalankan IngestionService di background thread.
    Status job di-update di PostgreSQL via IngestJobRepository.

    Karena thread tidak punya akses ke FastAPI Depends, kita buat session
    manual dari SessionLocal. Session di-close di finally block.

    CATATAN: IngestionService.run() memproses SEMUA file di raw_docs dan
    me-reset ChromaDB. file_ids dipakai untuk update metadata setelah selesai,
    bukan sebagai filter ingestion (batasan desain existing IngestionService).
    """
    # Import di dalam fungsi agar tidak circular & tidak load LLM saat startup
    from src.db.session import SessionLocal
    from src.services.ingestion_service import IngestionService

    db = SessionLocal()
    doc_repo = DocumentRepository(db)
    job_repo = IngestJobRepository(db)

    try:
        logger.info(f"[Ingest] Job '{job_id}' mulai. Files: {file_ids}")
        job_repo.update_progress(job_id, status="running")

        ingestor = IngestionService()
        ingestor.run()

        # Update metadata setiap file: status=ingested + chunk_count dari debug JSON
        processed   = 0
        total_chunk = 0
        for fid in file_ids:
            doc = doc_repo.get_by_id(fid)
            if doc is None:
                continue
            chunk_count = _get_chunk_count_from_debug(doc.original_filename)
            doc_repo.update_status(
                fid,
                status      = "ingested",
                chunk_count = chunk_count,
            )
            processed   += 1
            total_chunk += chunk_count

        job_repo.update_progress(
            job_id,
            status          = "completed",
            processed_files = processed,
            total_chunks    = total_chunk,
        )
        logger.info(
            f"[Ingest] Job '{job_id}' selesai. Processed: {processed} files, "
            f"{total_chunk} chunks total."
        )

    except Exception as e:
        logger.error(f"[Ingest] Job '{job_id}' GAGAL: {e}")
        try:
            job_repo.update_progress(
                job_id,
                status        = "failed",
                failed_files  = len(file_ids),
                error_message = str(e),
                errors        = [str(e)],
            )
        except Exception as inner:
            logger.error(f"[Ingest] Gagal update job status ke failed: {inner}")

    finally:
        # Lepas lock ingestion agar job baru bisa dimulai
        try:
            redis.delete(_INGEST_LOCK_KEY)
        except Exception:
            pass
        db.close()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validasi file gagal"},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        409: {"model": ErrorResponse, "description": "File sudah ada"},
    },
    summary="Upload file PDF soal atau jawaban",
    description=(
        "Upload satu file PDF. Nama file **wajib** mengikuti pola: "
        "`soal_<identifier>.pdf` atau `jawaban_<identifier>.pdf`. "
        "Identifier hanya boleh berisi huruf, angka, dan underscore (maks 50 char). "
        "File yang diupload belum langsung masuk ke ChromaDB — "
        "panggil `POST /v1/documents/ingest` setelah semua pasangan soal+jawaban terupload."
    ),
)
async def upload_document(
    file:        UploadFile = File(..., description="File PDF. Nama wajib: soal_X.pdf atau jawaban_X.pdf"),
    doc_type:    str        = Form(..., description="'soal' atau 'jawaban'"),
    jenis_ujian: str        = Form(..., description="Label ujian, contoh: 'Tryout 1'"),
    doc_repo:    DocumentRepository = Depends(get_document_repository),
    _:           None       = Depends(require_api_key),
) -> DocumentUploadResponse:

    # ── Validasi doc_type ─────────────────────────────────────────────────────
    doc_type = doc_type.strip().lower()
    if doc_type not in ("soal", "jawaban"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_doc_type", "message": "doc_type harus 'soal' atau 'jawaban'."},
        )

    jenis_ujian = jenis_ujian.strip()
    if not jenis_ujian or len(jenis_ujian) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_jenis_ujian", "message": "jenis_ujian wajib diisi, maks 100 karakter."},
        )

    # ── Validasi nama file ────────────────────────────────────────────────────
    filename = file.filename or ""
    if not _RE_FILENAME.match(filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "invalid_filename",
                "message": (
                    f"Nama file tidak valid: '{filename}'. "
                    "Harus mengikuti pola: soal_<alphanum_underscore>.pdf "
                    "atau jawaban_<alphanum_underscore>.pdf (identifier maks 50 char)."
                ),
            },
        )

    # ── Validasi doc_type konsisten dengan nama file ──────────────────────────
    if not filename.startswith(f"{doc_type}_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "doctype_filename_mismatch",
                "message": (
                    f"doc_type='{doc_type}' tapi nama file '{filename}' "
                    f"tidak diawali '{doc_type}_'. Sesuaikan keduanya."
                ),
            },
        )

    # ── Validasi MIME type (cek magic bytes, bukan hanya extension) ───────────
    header_bytes = await file.read(5)
    await file.seek(0)
    if header_bytes[:4] != b"%PDF":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_file_type", "message": "File bukan PDF yang valid."},
        )

    # ── Baca konten + validasi ukuran ────────────────────────────────────────
    content = await file.read()
    if len(content) > _MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error":   "file_too_large",
                "message": f"Ukuran file melebihi batas {settings.MAX_UPLOAD_SIZE_MB}MB.",
            },
        )

    # ── Cek duplikat via PostgreSQL ───────────────────────────────────────────
    existing = doc_repo.find_by_filename(filename)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error":   "file_exists",
                "message": (
                    f"File '{filename}' sudah ada (file_id: {existing.file_id}). "
                    "Hapus dulu via DELETE /v1/documents/{file_id} jika ingin replace."
                ),
            },
        )

    # ── Simpan file ke disk ───────────────────────────────────────────────────
    upload_dir = settings.DATA_DIR / "raw_docs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path  = upload_dir / filename

    try:
        with open(dest_path, "wb") as f_out:
            f_out.write(content)
    except Exception as e:
        logger.error(f"Gagal menyimpan file '{filename}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "storage_error", "message": "Gagal menyimpan file ke server."},
        )

    # ── INSERT metadata ke PostgreSQL ─────────────────────────────────────────
    file_id = generate_file_id(filename)
    try:
        doc = doc_repo.create(
            file_id           = file_id,
            original_filename = filename,
            stored_path       = str(dest_path),
            file_type         = doc_type,
            jenis_ujian       = jenis_ujian,
            size_bytes        = len(content),
        )
    except Exception as e:
        # Rollback: hapus file dari disk supaya tidak orphan
        logger.error(f"[Upload] INSERT documents gagal: {e}. Rolling back disk write.")
        try:
            os.remove(dest_path)
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "db_error", "message": "Gagal menyimpan metadata dokumen."},
        )

    logger.info(f"[Upload] File '{filename}' berhasil diupload. file_id='{file_id}'")
    return DocumentUploadResponse(
        file_id     = doc.file_id,
        filename    = doc.original_filename,
        doc_type    = doc.file_type,
        jenis_ujian = doc.jenis_ujian,
        size_bytes  = doc.size_bytes,
    )


@router.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "Tidak ada file valid untuk diingest"},
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "file_id tidak ditemukan"},
        409: {"model": ErrorResponse, "description": "Job ingestion sedang berjalan"},
    },
    summary="Mulai proses ingestion ke ChromaDB (async)",
    description=(
        "Jalankan ingestion secara asinkron. Gunakan `job_id` yang dikembalikan "
        "untuk polling status via `GET /v1/documents/ingest/{job_id}`. "
        "⚠️ Hanya satu job ingestion yang bisa berjalan dalam satu waktu. "
        "⚠️ Pastikan semua pasangan soal + jawaban sudah diupload sebelum ingest."
    ),
)
async def ingest_documents(
    req:      IngestRequest,
    redis:    Redis              = Depends(get_redis),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    job_repo: IngestJobRepository = Depends(get_ingest_job_repository),
    _:        None               = Depends(require_api_key),
) -> IngestJobResponse:

    # ── Cek apakah ada job yang sedang berjalan ───────────────────────────────
    if redis.exists(_INGEST_LOCK_KEY):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error":   "ingest_in_progress",
                "message": "Job ingestion sedang berjalan. Tunggu selesai sebelum memulai yang baru.",
            },
        )

    # ── Tentukan file yang akan diproses ──────────────────────────────────────
    if req.ingest_all_pending:
        pending_docs = doc_repo.list_pending_soal()
        target_ids   = [d.file_id for d in pending_docs]
        if not target_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error":   "no_pending_files",
                    "message": "Tidak ada file 'soal' yang pending ingestion.",
                },
            )
    else:
        target_ids = req.file_ids
        for fid in target_ids:
            if doc_repo.get_by_id(fid) is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error":   "file_not_found",
                        "message": (
                            f"file_id '{fid}' tidak ditemukan. "
                            "Upload dulu via POST /v1/documents/upload."
                        ),
                    },
                )

    # ── Validasi: setiap file soal harus punya pasangan jawaban ──────────────
    upload_dir = settings.DATA_DIR / "raw_docs"
    warnings   = []
    for fid in target_ids:
        doc = doc_repo.get_by_id(fid)
        if doc is None:
            continue
        filename = doc.original_filename
        if filename.startswith("soal_"):
            jawaban_filename = filename.replace("soal_", "jawaban_", 1)
            if not (upload_dir / jawaban_filename).exists():
                warnings.append(
                    f"File jawaban '{jawaban_filename}' tidak ditemukan untuk '{filename}'. "
                    "Soal akan diingest tanpa kunci jawaban."
                )
                logger.warning(f"[Ingest] Pasangan jawaban tidak ada: {jawaban_filename}")

    # ── Buat job record di Postgres & lock di Redis ───────────────────────────
    job_id = generate_job_id()
    try:
        job_repo.create(job_id=job_id, file_ids=target_ids)
        if warnings:
            job_repo.update_progress(job_id, errors=warnings)
    except Exception as e:
        logger.error(f"[Ingest] Gagal create job di DB: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "db_error", "message": "Gagal membuat job ingestion."},
        )

    # Lock di Redis: TTL 10 menit sebagai safety (akan di-delete saat job selesai)
    redis.set(_INGEST_LOCK_KEY, job_id, ex=600)

    # Spawn background thread untuk eksekusi ingestion
    thread = threading.Thread(
        target=_run_ingestion_job,
        args=(job_id, target_ids, redis),
        daemon=True,
    )
    thread.start()

    logger.info(f"[Ingest] Job '{job_id}' dimulai untuk {len(target_ids)} file.")

    return IngestJobResponse(
        job_id       = job_id,
        files_queued = len(target_ids),
    )


@router.get(
    "/ingest/{job_id}",
    response_model=IngestJobStatusResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "job_id tidak ditemukan"},
    },
    summary="Polling status job ingestion",
)
async def get_ingest_status(
    job_id:   str                = Depends(valid_job_id),
    job_repo: IngestJobRepository = Depends(get_ingest_job_repository),
    _:        None               = Depends(require_api_key),
) -> IngestJobStatusResponse:

    job = job_repo.get_by_id(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "job_not_found",
                "message": f"job_id '{job_id}' tidak ditemukan.",
            },
        )
    return _job_to_status_response(job)


@router.get(
    "",
    response_model=DocumentListResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
    },
    summary="List semua dokumen terdaftar",
)
async def list_documents(
    doc_type:    Optional[str]      = None,
    jenis_ujian: Optional[str]      = None,
    page:        int                = 1,
    limit:       int                = 20,
    doc_repo:    DocumentRepository = Depends(get_document_repository),
    _:           None               = Depends(require_api_key),
) -> DocumentListResponse:

    page  = max(page, 1)
    limit = max(1, min(limit, 100))

    docs, total = doc_repo.list_all(
        file_type   = doc_type.lower() if doc_type else None,
        jenis_ujian = jenis_ujian,
        page        = page,
        limit       = limit,
    )

    return DocumentListResponse(
        total = total,
        page  = page,
        limit = limit,
        items = [_doc_to_item(d) for d in docs],
    )


@router.delete(
    "/{file_id}",
    response_model=DocumentDeleteResponse,
    responses={
        401: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse, "description": "file_id tidak ditemukan"},
        409: {"model": ErrorResponse, "description": "Tidak bisa hapus saat ingestion berjalan"},
    },
    summary="Hapus dokumen dari storage dan ChromaDB",
)
async def delete_document(
    file_id:  str                = Depends(valid_file_id),
    redis:    Redis              = Depends(get_redis),
    doc_repo: DocumentRepository = Depends(get_document_repository),
    _:        None               = Depends(require_api_key),
) -> DocumentDeleteResponse:

    if redis.exists(_INGEST_LOCK_KEY):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error":   "ingest_in_progress",
                "message": "Tidak bisa menghapus file saat ingestion sedang berjalan.",
            },
        )

    doc = doc_repo.get_by_id(file_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "file_not_found",
                "message": f"file_id '{file_id}' tidak ditemukan.",
            },
        )

    filename        = doc.original_filename
    deleted_storage = False
    deleted_vector  = False
    chunks_removed  = 0

    # ── Hapus dari disk ───────────────────────────────────────────────────────
    file_path = settings.DATA_DIR / "raw_docs" / filename
    if file_path.exists():
        try:
            os.remove(file_path)
            deleted_storage = True
            logger.info(f"[Delete] File dihapus dari disk: {filename}")
        except Exception as e:
            logger.error(f"[Delete] Gagal hapus dari disk: {e}")
    else:
        deleted_storage = True  # File sudah tidak ada, anggap sukses

    # ── Hapus dari ChromaDB (berdasarkan metadata source == filename) ─────────
    try:
        from langchain_chroma import Chroma
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        emb = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_query",
            google_api_key=settings.GOOGLE_API_KEY,
        )
        vector_store = Chroma(
            collection_name="UTBK_TUTOR_KNOWLEDGE",  # Stage 1 decision
            embedding_function=emb,
            persist_directory=str(settings.CHROMA_PERSIST_DIR),
        )
        result  = vector_store._collection.get(
            where={"source": {"$eq": filename}},
            include=[],
        )
        doc_ids = result.get("ids", [])
        if doc_ids:
            vector_store._collection.delete(ids=doc_ids)
            chunks_removed = len(doc_ids)
            deleted_vector = True
            logger.info(f"[Delete] {chunks_removed} chunk dihapus dari ChromaDB untuk '{filename}'")
        else:
            deleted_vector = True  # Tidak ada di ChromaDB, anggap sukses
    except Exception as e:
        logger.error(f"[Delete] Gagal hapus dari ChromaDB: {e}")

    # ── Soft delete metadata di PostgreSQL ────────────────────────────────────
    doc_repo.soft_delete(file_id)

    return DocumentDeleteResponse(
        file_id               = file_id,
        deleted_from_storage  = deleted_storage,
        deleted_from_vectordb = deleted_vector,
        chunks_removed        = chunks_removed,
        message               = (
            f"File '{filename}' berhasil dihapus. "
            f"{chunks_removed} chunk dihapus dari ChromaDB."
        ),
    )
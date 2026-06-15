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
  mungkin mendapat hasil kosong dari ChromaDB. BE perlu handle kondisi ini.
  Rekomendasi future: modifikasi IngestionService untuk incremental ingestion.
"""

import os
import json
import re
import time
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from redis import Redis

from src.api.deps import get_redis, require_api_key, valid_file_id, valid_job_id
from src.core.config import settings
from src.core.logger import get_logger
from src.core.security import generate_file_id, generate_job_id
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

# ── Konstanta Redis key ───────────────────────────────────────────────────────
_DOC_META_PREFIX   = "doc:meta:"    # HASH per file_id
_DOC_INDEX_KEY     = "doc:index"    # SET semua file_id
_INGEST_JOB_PREFIX = "ingest:job:"  # HASH per job_id
_INGEST_LOCK_KEY   = "ingest:lock"  # STRING — mutex ingestion

# ── Validasi ──────────────────────────────────────────────────────────────────
_MAX_SIZE    = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
_RE_FILENAME = re.compile(r"^(soal|jawaban)_[a-zA-Z0-9_]{1,50}\.pdf$")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _get_doc_meta(redis: Redis, file_id: str) -> Optional[dict]:
    """Ambil metadata dokumen dari Redis HASH. Return None jika tidak ada."""
    try:
        data = redis.hgetall(f"{_DOC_META_PREFIX}{file_id}")
        return data if data else None
    except Exception:
        return None


def _list_all_docs(redis: Redis) -> List[dict]:
    """Ambil semua metadata dokumen terdaftar dari Redis SET + HASH."""
    try:
        file_ids = redis.smembers(_DOC_INDEX_KEY)
        docs = []
        for fid in file_ids:
            meta = _get_doc_meta(redis, fid)
            if meta:
                docs.append(meta)
        return docs
    except Exception:
        return []


def _save_doc_meta(redis: Redis, file_id: str, meta: dict) -> None:
    """Simpan atau update metadata dokumen di Redis HASH, daftarkan ke SET index."""
    try:
        redis.hset(f"{_DOC_META_PREFIX}{file_id}", mapping=meta)
        redis.sadd(_DOC_INDEX_KEY, file_id)
    except Exception as e:
        logger.error(f"Gagal simpan doc meta ke Redis: {e}")


def _delete_doc_meta(redis: Redis, file_id: str) -> None:
    """Hapus metadata dokumen dari Redis HASH + SET index."""
    try:
        redis.delete(f"{_DOC_META_PREFIX}{file_id}")
        redis.srem(_DOC_INDEX_KEY, file_id)
    except Exception as e:
        logger.warning(f"Gagal hapus doc meta dari Redis: {e}")


def _get_chunk_count_from_debug(filename: str) -> int:
    """
    Hitung jumlah soal (chunk) dari file debug JSON yang dihasilkan IngestionService.
    File debug: data/debug/debug_{stem}.json
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
    Status job di-update di Redis HASH.

    CATATAN: IngestionService.run() memproses SEMUA file di raw_docs dan
    me-reset ChromaDB. file_ids dipakai untuk update metadata setelah selesai,
    bukan sebagai filter ingestion (batasan desain existing IngestionService).
    """
    job_key = f"{_INGEST_JOB_PREFIX}{job_id}"

    # Import di dalam fungsi agar tidak circular dan tidak load LLM saat startup
    from src.services.ingestion_service import IngestionService

    try:
        logger.info(f"[Ingest] Job '{job_id}' mulai. Files: {file_ids}")
        redis.hset(job_key, mapping={"status": "processing"})

        ingestor = IngestionService()
        ingestor.run()

        # Update metadata setiap file: ingested=true + chunk_count dari debug JSON
        processed = 0
        for fid in file_ids:
            meta = _get_doc_meta(redis, fid)
            if not meta:
                continue
            filename    = meta.get("filename", "")
            chunk_count = _get_chunk_count_from_debug(filename)
            redis.hset(
                f"{_DOC_META_PREFIX}{fid}",
                mapping={
                    "ingested":    "true",
                    "ingested_at": datetime.utcnow().isoformat(),
                    "chunk_count": str(chunk_count),
                },
            )
            processed += 1

        redis.hset(
            job_key,
            mapping={
                "status":          "done",
                "files_processed": str(processed),
                "files_failed":    "0",
                "completed_at":    datetime.utcnow().isoformat(),
            },
        )
        logger.info(f"[Ingest] Job '{job_id}' selesai. Processed: {processed} files.")

    except Exception as e:
        logger.error(f"[Ingest] Job '{job_id}' GAGAL: {e}")
        redis.hset(
            job_key,
            mapping={
                "status":          "failed",
                "files_failed":    str(len(file_ids)),
                "errors":          json.dumps([str(e)]),
                "completed_at":    datetime.utcnow().isoformat(),
            },
        )
    finally:
        # Lepas lock ingestion agar job baru bisa dimulai
        redis.delete(_INGEST_LOCK_KEY)


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
    redis:       Redis      = Depends(get_redis),
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

    # ── Cek duplikat (nama file sama sudah ada di registry) ───────────────────
    for doc in _list_all_docs(redis):
        if doc.get("filename") == filename:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error":   "file_exists",
                    "message": (
                        f"File '{filename}' sudah ada (file_id: {doc['file_id']}). "
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

    # ── Simpan metadata ke Redis ──────────────────────────────────────────────
    file_id = generate_file_id(filename)
    meta = {
        "file_id":     file_id,
        "filename":    filename,
        "doc_type":    doc_type,
        "jenis_ujian": jenis_ujian,
        "size_bytes":  str(len(content)),
        "ingested":    "false",
        "chunk_count": "0",
        "uploaded_at": datetime.utcnow().isoformat(),
        "ingested_at": "",
    }
    _save_doc_meta(redis, file_id, meta)

    logger.info(f"[Upload] File '{filename}' berhasil diupload. file_id='{file_id}'")
    return DocumentUploadResponse(
        file_id     = file_id,
        filename    = filename,
        doc_type    = doc_type,
        jenis_ujian = jenis_ujian,
        size_bytes  = len(content),
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
    req:   IngestRequest,
    redis: Redis = Depends(get_redis),
    _:     None  = Depends(require_api_key),
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
        all_docs   = _list_all_docs(redis)
        target_ids = [
            d["file_id"] for d in all_docs
            if d.get("ingested") == "false" and d.get("doc_type") == "soal"
        ]
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
            if not _get_doc_meta(redis, fid):
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
        meta     = _get_doc_meta(redis, fid)
        filename = (meta or {}).get("filename", "")
        if filename.startswith("soal_"):
            jawaban_filename = filename.replace("soal_", "jawaban_", 1)
            if not (upload_dir / jawaban_filename).exists():
                warnings.append(
                    f"File jawaban '{jawaban_filename}' tidak ditemukan untuk '{filename}'. "
                    "Soal akan diingest tanpa kunci jawaban."
                )
                logger.warning(f"[Ingest] Pasangan jawaban tidak ada: {jawaban_filename}")

    # ── Buat job di Redis & jalankan background thread ────────────────────────
    job_id  = generate_job_id()
    job_key = f"{_INGEST_JOB_PREFIX}{job_id}"

    redis.hset(
        job_key,
        mapping={
            "job_id":          job_id,
            "status":          "processing",
            "files_queued":    str(len(target_ids)),
            "files_processed": "0",
            "files_failed":    "0",
            "errors":          json.dumps(warnings),
            "created_at":      datetime.utcnow().isoformat(),
            "completed_at":    "",
        },
    )
    redis.expire(job_key, 86400)             # Job record TTL: 24 jam
    redis.set(_INGEST_LOCK_KEY, job_id, ex=600)  # Lock TTL: 10 menit (safety)

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
    job_id: str   = Depends(valid_job_id),
    redis:  Redis = Depends(get_redis),
    _:      None  = Depends(require_api_key),
) -> IngestJobStatusResponse:

    job_key = f"{_INGEST_JOB_PREFIX}{job_id}"
    data    = redis.hgetall(job_key)

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "job_not_found",
                "message": f"job_id '{job_id}' tidak ditemukan atau sudah expired (>24 jam).",
            },
        )

    errors_raw = data.get("errors", "[]")
    try:
        errors = json.loads(errors_raw)
    except Exception:
        errors = [errors_raw] if errors_raw else []

    return IngestJobStatusResponse(
        job_id          = data.get("job_id", job_id),
        status          = data.get("status", "unknown"),
        files_queued    = int(data.get("files_queued", 0)),
        files_processed = int(data.get("files_processed", 0)),
        files_failed    = int(data.get("files_failed", 0)),
        errors          = errors,
        created_at      = data.get("created_at", ""),
        completed_at    = data.get("completed_at") or None,
    )


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
    doc_type:    Optional[str] = None,
    jenis_ujian: Optional[str] = None,
    page:        int           = 1,
    limit:       int           = 20,
    redis:       Redis         = Depends(get_redis),
    _:           None          = Depends(require_api_key),
) -> DocumentListResponse:

    page  = max(page, 1)
    limit = max(1, min(limit, 100))

    all_docs = _list_all_docs(redis)

    if doc_type:
        all_docs = [d for d in all_docs if d.get("doc_type", "").lower() == doc_type.lower()]
    if jenis_ujian:
        all_docs = [d for d in all_docs if d.get("jenis_ujian", "").lower() == jenis_ujian.lower()]

    all_docs.sort(key=lambda d: d.get("uploaded_at", ""), reverse=True)

    total  = len(all_docs)
    offset = (page - 1) * limit
    paged  = all_docs[offset : offset + limit]

    items = [
        DocumentItem(
            file_id     = d["file_id"],
            filename    = d["filename"],
            doc_type    = d["doc_type"],
            jenis_ujian = d["jenis_ujian"],
            size_bytes  = int(d.get("size_bytes", 0)),
            ingested    = d.get("ingested", "false") == "true",
            uploaded_at = d.get("uploaded_at", ""),
            ingested_at = d.get("ingested_at") or None,
            chunk_count = int(d.get("chunk_count", 0)),
        )
        for d in paged
    ]

    return DocumentListResponse(total=total, page=page, limit=limit, items=items)


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
    file_id: str   = Depends(valid_file_id),
    redis:   Redis = Depends(get_redis),
    _:       None  = Depends(require_api_key),
) -> DocumentDeleteResponse:

    if redis.exists(_INGEST_LOCK_KEY):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error":   "ingest_in_progress",
                "message": "Tidak bisa menghapus file saat ingestion sedang berjalan.",
            },
        )

    meta = _get_doc_meta(redis, file_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":   "file_not_found",
                "message": f"file_id '{file_id}' tidak ditemukan.",
            },
        )

    filename        = meta.get("filename", "")
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
            collection_name="RAG_REDUKA_DOC_KNOWLEDGE",
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

    # ── Hapus dari Redis registry ─────────────────────────────────────────────
    _delete_doc_meta(redis, file_id)

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
-- ============================================================================
-- UTBK Tutor RAG System - PostgreSQL Schema
-- ----------------------------------------------------------------------------
-- This file is the SOURCE OF TRUTH for the database schema.
-- It is auto-executed by the postgres container on first startup
-- (mounted at /docker-entrypoint-initdb.d/01-init.sql).
--
-- A human-readable copy is also kept at docs/schema.sql for documentation.
-- When you change the schema here, update docs/schema.sql to match
-- (or run `make schema-dump`).
-- ============================================================================


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: documents
-- ----------------------------------------------------------------------------
-- Persistent metadata for uploaded PDF files (soal & jawaban).
-- Replaces the former Redis `doc:meta:{file_id}` HASH and `doc:index` SET.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    file_id           VARCHAR(80)  PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_path       TEXT         NOT NULL,
    file_type         VARCHAR(20)  NOT NULL,
    jenis_ujian       VARCHAR(100) NOT NULL,
    mime_type         VARCHAR(50)  NOT NULL DEFAULT 'application/pdf',
    size_bytes        BIGINT       NOT NULL,
    status            VARCHAR(20)  NOT NULL DEFAULT 'uploaded',
    chunk_count       INTEGER      NOT NULL DEFAULT 0,
    error_message     TEXT,
    uploaded_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    ingested_at       TIMESTAMPTZ,
    deleted_at        TIMESTAMPTZ,

    CONSTRAINT documents_file_id_format
        CHECK (file_id ~ '^file_[a-zA-Z0-9_]+$'),
    CONSTRAINT documents_file_type_valid
        CHECK (file_type IN ('soal', 'jawaban')),
    CONSTRAINT documents_status_valid
        CHECK (status IN ('uploaded', 'ingested', 'failed', 'deleted')),
    CONSTRAINT documents_size_positive
        CHECK (size_bytes > 0),
    CONSTRAINT documents_chunk_count_nonneg
        CHECK (chunk_count >= 0),
    CONSTRAINT documents_jenis_ujian_nonempty
        CHECK (length(trim(jenis_ujian)) > 0)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_documents_status_active
    ON documents (status)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at
    ON documents (uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_documents_file_type
    ON documents (file_type);

CREATE INDEX IF NOT EXISTS idx_documents_jenis_ujian
    ON documents (jenis_ujian);

-- Unique constraint: tidak boleh ada 2 file aktif dengan filename sama
-- (deleted_at IS NULL berarti partial unique index — file yang sudah dihapus
-- soft tidak dihitung).
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_filename_active
    ON documents (original_filename)
    WHERE deleted_at IS NULL;

-- Column comments (visible via \d+ in psql)
COMMENT ON TABLE  documents IS
    'Metadata persisten untuk file PDF yang diupload (soal/jawaban UTBK).';
COMMENT ON COLUMN documents.file_id IS
    'Unique identifier dengan format: file_{alphanum_underscore}';
COMMENT ON COLUMN documents.file_type IS
    'Tipe file: "soal" (PDF pertanyaan) atau "jawaban" (PDF kunci jawaban)';
COMMENT ON COLUMN documents.jenis_ujian IS
    'Label ujian yang diberikan user saat upload. Contoh: "Tryout 1", "UTBK 2024"';
COMMENT ON COLUMN documents.status IS
    'Lifecycle: uploaded -> ingested | failed. "deleted" untuk soft delete.';
COMMENT ON COLUMN documents.chunk_count IS
    'Jumlah chunk yang dihasilkan saat ingestion ke ChromaDB (0 sampai diingest)';
COMMENT ON COLUMN documents.deleted_at IS
    'Timestamp soft delete. NULL berarti aktif.';


-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE: ingest_jobs
-- ----------------------------------------------------------------------------
-- Audit trail untuk job ingestion async (PDF -> ChromaDB pipeline).
-- Replaces the former Redis `ingest:job:{job_id}` HASH.
--
-- Note: Mutex untuk menyamakan hanya satu job berjalan sekaligus tetap
-- menggunakan Redis SET NX (key: `ingest:lock`) karena lebih cepat & atomic.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id           VARCHAR(80)  PRIMARY KEY,
    status           VARCHAR(20)  NOT NULL DEFAULT 'pending',
    file_ids         JSONB        NOT NULL DEFAULT '[]'::jsonb,
    total_files      INTEGER      NOT NULL DEFAULT 0,
    processed_files  INTEGER      NOT NULL DEFAULT 0,
    failed_files     INTEGER      NOT NULL DEFAULT 0,
    total_chunks     INTEGER      NOT NULL DEFAULT 0,
    errors           JSONB        NOT NULL DEFAULT '[]'::jsonb,
    error_message    TEXT,
    started_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,

    CONSTRAINT ingest_jobs_job_id_format
        CHECK (job_id ~ '^job_[a-zA-Z0-9_]+$'),
    CONSTRAINT ingest_jobs_status_valid
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT ingest_jobs_counts_nonneg
        CHECK (
            total_files     >= 0
            AND processed_files >= 0
            AND failed_files    >= 0
            AND total_chunks    >= 0
        )
);

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_status
    ON ingest_jobs (status);

CREATE INDEX IF NOT EXISTS idx_ingest_jobs_started_at
    ON ingest_jobs (started_at DESC);

COMMENT ON TABLE  ingest_jobs IS
    'Audit log untuk job ingestion async PDF ke ChromaDB.';
COMMENT ON COLUMN ingest_jobs.file_ids IS
    'Array file_id (JSON) yang diproses dalam job ini';
COMMENT ON COLUMN ingest_jobs.errors IS
    'Array error message (JSON) yang terjadi selama proses';
COMMENT ON COLUMN ingest_jobs.status IS
    'State machine: pending -> running -> completed | failed | cancelled';


-- ─────────────────────────────────────────────────────────────────────────────
-- FUTURE: Reserved space for phase 2 tables
-- ----------------------------------------------------------------------------
-- conversations  : archive sesi chat setelah Redis TTL expired
-- messages       : detail per pesan (untuk analytics & quality eval)
-- usage_metrics  : cost & latency tracking per request
-- ─────────────────────────────────────────────────────────────────────────────
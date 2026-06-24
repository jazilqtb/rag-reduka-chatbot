"""
Test Fixtures — Shared infrastructure untuk unit & integration tests.

Fixtures yang disediakan:
  - fake_redis           : fakeredis client (in-memory, behaves like real Redis)
  - db_engine            : SQLite in-memory engine dengan tables created
  - db_session           : SQLAlchemy Session yang di-rollback tiap test
  - document_repo        : DocumentRepository wrapped around db_session
  - ingest_job_repo      : IngestJobRepository wrapped around db_session
  - test_client          : FastAPI TestClient dengan dependency override
  - api_key_header       : header X-API-Key valid untuk request

Catatan SQLite vs PostgreSQL:
  Beberapa CHECK constraint di model pakai regex operator '~' yang PostgreSQL-
  specific. Saat create_all() di SQLite, constraint regex di-strip otomatis
  (lihat _create_tables_sqlite_compatible). Validasi format di app level tetap
  jalan via Pydantic schema.
"""

import os
from typing import Generator

# Set env vars BEFORE importing app code (config.py read them at import time)
os.environ.setdefault("GOOGLE_API_KEY",     "test-gemini-key")
os.environ.setdefault("API_KEY",            "test-api-key-32-chars-minimum-length")
os.environ.setdefault("POSTGRES_PASSWORD",  "test-password")

import pytest
import fakeredis
from fastapi.testclient import TestClient
from sqlalchemy import CheckConstraint, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON


# ── Helper: strip PostgreSQL-only constructs untuk SQLite test ───────────────

def _make_sqlite_compatible(metadata) -> None:
    """
    Strip CHECK constraints dengan regex operator '~' (Postgres-only)
    dan ganti JSONB → JSON supaya bisa di-create_all() di SQLite.

    Ini test-only modification — production tetap pakai Postgres dengan
    semua constraint asli dari init.sql.
    """
    for table in metadata.tables.values():
        # Strip CHECK dengan regex operator
        new_constraints = set()
        for c in table.constraints:
            if isinstance(c, CheckConstraint):
                sql_text = str(c.sqltext)
                if "~" in sql_text:
                    continue  # skip regex CHECKs
            new_constraints.add(c)
        table.constraints = new_constraints

        # Replace JSONB with JSON
        for col in table.columns:
            if isinstance(col.type, JSONB):
                col.type = JSON()


# ══════════════════════════════════════════════════════════════════════════════
# FAKE REDIS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def fake_redis() -> fakeredis.FakeRedis:
    """
    Fresh in-memory Redis per test. Behaves like real Redis 7.

    Note: fakeredis returns bytes by default. Kalau app code pakai
    decode_responses=True, gunakan FakeRedis(decode_responses=True).
    """
    return fakeredis.FakeRedis(decode_responses=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE (SQLite in-memory)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def db_engine():
    """
    SQLite in-memory engine (per session). Tables di-create sekali di awal,
    di-drop saat session end.
    """
    from src.db.base import Base
    from src.domain import models  # ensure all models registered

    # Strip Postgres-only stuff for SQLite compat
    _make_sqlite_compatible(Base.metadata)

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    """
    SQLAlchemy session per test dengan transaction rollback di akhir.
    Setiap test mulai dengan DB bersih (no data dari test lain).
    """
    connection  = db_engine.connect()
    transaction = connection.begin()

    Session_ = sessionmaker(bind=connection, autoflush=False, autocommit=False, expire_on_commit=False)
    session  = Session_()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def document_repo(db_session):
    """DocumentRepository terikat ke db_session fixture."""
    from src.db.repositories import DocumentRepository
    return DocumentRepository(db_session)


@pytest.fixture
def ingest_job_repo(db_session):
    """IngestJobRepository terikat ke db_session fixture."""
    from src.db.repositories import IngestJobRepository
    return IngestJobRepository(db_session)


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI TEST CLIENT
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_client(db_session, fake_redis, monkeypatch, tmp_path) -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient dengan semua external dependency di-mock:
      - DB session   → SQLite in-memory (per test)
      - Redis        → fakeredis
      - ChatService  → stub (no LLM call)

    Note: Test yang butuh real LLM/embedding harus pakai marker @pytest.mark.llm
    dan skip secara default.
    """
    # Pastikan DATA_DIR menunjuk ke folder temp supaya upload tests tidak
    # tulis ke disk produksi.
    from src.core import config as config_module
    monkeypatch.setattr(config_module.settings, "DATA_DIR",           tmp_path)
    monkeypatch.setattr(config_module.settings, "CHROMA_PERSIST_DIR", tmp_path / "vector_store")

    # Pre-create data subdirs
    (tmp_path / "raw_docs").mkdir(exist_ok=True)
    (tmp_path / "debug").mkdir(exist_ok=True)
    (tmp_path / "vector_store").mkdir(exist_ok=True)

    # Stub ChatService so app boot doesn't try to init real LLM
    from src.services import chat_service as cs_module

    class _StubChatService:
        def __init__(self): pass

    monkeypatch.setattr(cs_module, "ChatService", _StubChatService)

    # Import main AFTER monkeypatching
    from main import app
    from src.api.deps import get_db_session, get_redis, _get_chat_service_singleton, _get_redis_singleton

    # Override dependencies
    def _override_db_session():
        try:
            yield db_session
        finally:
            pass  # rollback handled by db_session fixture

    def _override_redis():
        return fake_redis

    app.dependency_overrides[get_db_session] = _override_db_session
    app.dependency_overrides[get_redis]      = _override_redis

    # Also patch the cached singletons so non-Depends code paths also see fake.
    _get_redis_singleton.cache_clear()
    monkeypatch.setattr(
        "src.api.deps._get_redis_singleton",
        lambda: fake_redis,
    )

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def api_key_header() -> dict:
    """Header X-API-Key valid (matches API_KEY env set di atas)."""
    return {"X-API-Key": "test-api-key-32-chars-minimum-length"}
"""
Database engine + session factory untuk PostgreSQL.

Dua cara akses session:

  1. FastAPI dependency:
        from src.db.session import get_db_session
        def endpoint(db: Session = Depends(get_db_session)): ...

  2. Context manager (di luar request, mis. background task atau script):
        from src.db.session import transactional_session
        with transactional_session() as db:
            doc = doc_repo.create(...)

Engine pakai connection pool default SQLAlchemy + pool_pre_ping supaya
koneksi mati otomatis di-recycle.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("db.session")


# ── Database URL ────────────────────────────────────────────────────────────
# Dibangun dari Settings supaya satu sumber konfigurasi (.env / env var).
DATABASE_URL = (
    f"postgresql+psycopg2://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
)


# ── Engine ──────────────────────────────────────────────────────────────────
# pool_size       : jumlah koneksi yang dipertahankan
# max_overflow    : koneksi tambahan jika pool penuh
# pool_pre_ping   : test koneksi dengan SELECT 1 sebelum dipakai (handle
#                    koneksi yang mati karena timeout DB/network)
# echo            : True untuk log semua SQL (development debugging only)
engine = create_engine(
    DATABASE_URL,
    pool_size     = 5,
    max_overflow  = 10,
    pool_pre_ping = True,
    echo          = False,
)


# ── Session Factory ─────────────────────────────────────────────────────────
# autocommit=False    : transaksi eksplisit (default SQLAlchemy 2.0)
# autoflush=False     : tidak auto-flush sebelum query — kita kontrol manual
# expire_on_commit=False : object tetap accessible setelah commit
SessionLocal = sessionmaker(
    bind             = engine,
    autocommit       = False,
    autoflush        = False,
    expire_on_commit = False,
)


# ══════════════════════════════════════════════════════════════════════════════
# Dependency untuk FastAPI
# ══════════════════════════════════════════════════════════════════════════════

def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency: yields a database session per request.

    Usage:
        @router.get("/foo")
        def endpoint(db: Session = Depends(get_db_session)):
            ...

    Catatan: dependency ini TIDAK auto-commit. Repository methods yang
    commit per call (lihat docstring di setiap repo). Kalau butuh batch
    transaksi, panggil session.commit()/rollback() manual.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# Context manager untuk non-FastAPI (background tasks, scripts, CLI)
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def transactional_session() -> Generator[Session, None, None]:
    """
    Context manager untuk operasi DB di luar FastAPI request.

    Auto-commit on success, rollback on exception.

    Usage:
        with transactional_session() as db:
            doc_repo = DocumentRepository(db)
            doc_repo.create(...)
            # commit happens automatically on exit
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# Health check helper
# ══════════════════════════════════════════════════════════════════════════════

def check_db_connection() -> bool:
    """
    Test koneksi DB dengan SELECT 1. Dipakai di /v1/health/detailed.
    Tidak raise exception — return True/False.
    """
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"[DB] Connection check failed: {e}")
        return False
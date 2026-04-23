"""
Utilitas keamanan terpusat.

  - _RE_SESSION_ID : regex validator session_id (dipakai session.py + deps.py)
  - verify_api_key : constant-time comparison (anti timing attack)
  - check_rate_limit: Redis sliding window per user_id
  - generate_session_id / generate_file_id / generate_job_id
"""

import hmac
import re
import time
import uuid
from redis import Redis
from fastapi import HTTPException, status

from src.core.config import settings


# ══════════════════════════════════════════════════════════════════════════════
# REGEX VALIDATORS
# Didefinisikan di sini agar bisa diimport oleh session.py, deps.py, schemas.py
# tanpa circular dependency.
# ══════════════════════════════════════════════════════════════════════════════

_RE_SESSION_ID = re.compile(r"^sess_[a-zA-Z0-9_]{4,49}$")
"""
Format session_id yang valid: sess_<alphanum_underscore>, panjang total 5-54 char.
Contoh valid  : sess_abc123xyz, sess_a1b2c3d4e5f6
Contoh invalid: abc123, session_xyz, sess_
"""


# ══════════════════════════════════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════════════════════════════════

def verify_api_key(provided_key: str) -> bool:
    """
    Bandingkan API key menggunakan hmac.compare_digest untuk mencegah
    timing attack. Mengembalikan True jika cocok.
    """
    return hmac.compare_digest(
        provided_key.encode("utf-8"),
        settings.API_KEY.encode("utf-8"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# RATE LIMITING — Redis Sliding Window
# ══════════════════════════════════════════════════════════════════════════════

_RATELIMIT_PREFIX = "ratelimit:chat"


def check_rate_limit(
    user_id:    str,
    redis:      Redis,
    limit:      int = None,
    window_sec: int = None,
) -> None:
    """
    Cek apakah user_id telah melewati batas request dalam satu window waktu.
    Menggunakan Redis sorted set (timestamp sebagai score) untuk sliding window.

    Raises HTTPException 429 jika limit terlampaui.

    Args:
        user_id    : ID unik siswa
        redis      : Redis connection
        limit      : maks request per window (default: settings.RATE_LIMIT_CHAT_MAX)
        window_sec : durasi window dalam detik (default: settings.REDIS_RATELIMIT_TTL)
    """
    limit      = limit      or settings.RATE_LIMIT_CHAT_MAX
    window_sec = window_sec or settings.REDIS_RATELIMIT_TTL

    key = f"{_RATELIMIT_PREFIX}:{user_id}"
    now = time.time()

    pipe = redis.pipeline()
    # Hapus entry yang sudah di luar window
    pipe.zremrangebyscore(key, 0, now - window_sec)
    # Tambah entry baru (gunakan uuid pendek agar score tidak collision)
    pipe.zadd(key, {f"{now}:{uuid.uuid4().hex[:8]}": now})
    # Hitung total dalam window
    pipe.zcard(key)
    # Set TTL agar key tidak menumpuk selamanya
    pipe.expire(key, window_sec)
    results = pipe.execute()

    count = results[2]
    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error":   "rate_limit_exceeded",
                "message": (
                    f"Terlalu banyak request. "
                    f"Maks {limit} request per {window_sec} detik."
                ),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# ID GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def generate_session_id() -> str:
    """
    Generate session_id unik dengan format sess_{hex16}.
    Contoh: sess_3f7a2b1c9e4d8f0a
    """
    return f"sess_{uuid.uuid4().hex[:16]}"


def generate_file_id(filename: str) -> str:
    """
    Generate file_id deterministik dari nama file + uuid pendek.
    Format: file_{stem}_{uuid6}
    Contoh: file_soal_tryout1_a3f9c2

    Args:
        filename: nama file lengkap, misal "soal_tryout1.pdf"
    """
    stem       = filename.replace(".pdf", "").replace(" ", "_")[:40]
    short_uuid = uuid.uuid4().hex[:6]
    return f"file_{stem}_{short_uuid}"


def generate_job_id() -> str:
    """
    Generate job_id untuk ingestion job.
    Format: job_{unix_timestamp}_{uuid6}
    Contoh: job_1745497200_a3f9c2
    """
    ts = int(time.time())
    return f"job_{ts}_{uuid.uuid4().hex[:6]}"
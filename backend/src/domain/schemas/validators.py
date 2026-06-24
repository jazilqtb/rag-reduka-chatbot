"""
Regex patterns dan validator functions yang dipakai di schema-schema lain.

Diekstrak ke modul terpisah supaya:
  - Bisa di-reuse antar schema tanpa circular import
  - Bisa di-test independen
  - Satu sumber kebenaran untuk format ID
"""

import re

# ══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

RE_USER_ID    = re.compile(r"^usr_[a-zA-Z0-9_]{3,49}$")
RE_SESSION_ID = re.compile(r"^sess_[a-zA-Z0-9_]{4,49}$")
RE_FILE_NAME  = re.compile(r"^(soal|jawaban)_[a-zA-Z0-9_]{1,50}\.pdf$")
RE_FILE_ID    = re.compile(r"^file_[a-zA-Z0-9_]{1,80}$")
RE_JOB_ID     = re.compile(r"^job_[a-zA-Z0-9_]{1,80}$")


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def validate_user_id(v: str) -> str:
    """Validasi format user_id. Raises ValueError jika tidak valid."""
    v = v.strip()
    if not RE_USER_ID.match(v):
        raise ValueError(
            "user_id harus format 'usr_<alphanum_underscore>', panjang 4-53 karakter. "
            f"Contoh: usr_student123. Diterima: '{v}'"
        )
    return v


def validate_session_id(v: str) -> str:
    """Validasi format session_id. Raises ValueError jika tidak valid."""
    v = v.strip()
    if not RE_SESSION_ID.match(v):
        raise ValueError(
            "session_id harus format 'sess_<alphanum_underscore>', panjang 5-54 karakter. "
            f"Contoh: sess_abc123. Diterima: '{v}'"
        )
    return v


def validate_file_id(v: str) -> str:
    """Validasi format file_id. Raises ValueError jika tidak valid."""
    v = v.strip()
    if not RE_FILE_ID.match(v):
        raise ValueError(
            f"file_id tidak valid: '{v}'. Format: file_<alphanum_underscore>."
        )
    return v


def validate_job_id(v: str) -> str:
    """Validasi format job_id. Raises ValueError jika tidak valid."""
    v = v.strip()
    if not RE_JOB_ID.match(v):
        raise ValueError(
            f"job_id tidak valid: '{v}'. Format: job_<alphanum_underscore>."
        )
    return v


def validate_filename(v: str) -> str:
    """Validasi format nama file PDF (soal/jawaban). Raises ValueError jika tidak valid."""
    v = v.strip()
    if not RE_FILE_NAME.match(v):
        raise ValueError(
            f"Nama file tidak valid: '{v}'. "
            "Format: '(soal|jawaban)_<nama>.pdf'."
        )
    return v
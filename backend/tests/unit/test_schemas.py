"""
Unit tests untuk Pydantic schemas (src/domain/schemas/).

Cover:
  - ChatRequest field validators
  - IngestRequest model_validator (at least one of file_ids/ingest_all_pending)
  - ID format validators (user_id, session_id, file_id, job_id)
  - Edge cases: whitespace, empty, mixed valid+invalid lists
"""

import pytest
from pydantic import ValidationError

from src.domain.schemas import (
    ChatRequest,
    IngestRequest,
    validate_file_id,
    validate_session_id,
    validate_user_id,
)


pytestmark = pytest.mark.unit


# ══════════════════════════════════════════════════════════════════════════════
# ChatRequest
# ══════════════════════════════════════════════════════════════════════════════

class TestChatRequest:

    def test_valid_minimal(self):
        req = ChatRequest(user_id="usr_test123", query="Halo")
        assert req.user_id == "usr_test123"
        assert req.session_id is None
        assert req.query == "Halo"

    def test_valid_with_session(self):
        req = ChatRequest(
            user_id="usr_abc",
            session_id="sess_xyz1234",
            query="Jelaskan nomor 3",
        )
        assert req.session_id == "sess_xyz1234"

    def test_user_id_too_short(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="usr_x", query="hi")

    def test_user_id_wrong_prefix(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="user_abc123", query="hi")

    def test_session_id_invalid_format(self):
        with pytest.raises(ValidationError):
            ChatRequest(
                user_id="usr_test123",
                session_id="invalid_sess",
                query="hi",
            )

    def test_query_min_length(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="usr_test123", query="a")

    def test_query_max_length(self):
        with pytest.raises(ValidationError):
            ChatRequest(user_id="usr_test123", query="x" * 2001)

    def test_query_whitespace_only(self):
        with pytest.raises(ValidationError) as exc:
            ChatRequest(user_id="usr_test123", query="   ")
        assert "whitespace" in str(exc.value).lower()

    def test_query_strips_whitespace(self):
        req = ChatRequest(user_id="usr_test123", query="  halo  ")
        assert req.query == "halo"


# ══════════════════════════════════════════════════════════════════════════════
# IngestRequest
# ══════════════════════════════════════════════════════════════════════════════

class TestIngestRequest:

    def test_with_file_ids(self):
        req = IngestRequest(file_ids=["file_abc", "file_def"])
        assert req.file_ids == ["file_abc", "file_def"]
        assert req.ingest_all_pending is False

    def test_with_ingest_all_pending(self):
        req = IngestRequest(ingest_all_pending=True)
        assert req.ingest_all_pending is True
        assert req.file_ids == []

    def test_both_empty_raises(self):
        with pytest.raises(ValidationError) as exc:
            IngestRequest()
        assert "minimal salah satu" in str(exc.value).lower()

    def test_invalid_file_id_format(self):
        with pytest.raises(ValidationError):
            IngestRequest(file_ids=["bad_format"])

    def test_one_valid_one_invalid_rejected(self):
        """Semua harus valid; satu invalid → reject."""
        with pytest.raises(ValidationError):
            IngestRequest(file_ids=["file_ok", "not_valid"])

    def test_file_ids_stripped(self):
        req = IngestRequest(file_ids=["  file_abc  "])
        assert req.file_ids == ["file_abc"]


# ══════════════════════════════════════════════════════════════════════════════
# Validators (standalone functions)
# ══════════════════════════════════════════════════════════════════════════════

class TestValidators:

    @pytest.mark.parametrize("uid", [
        "usr_abc",
        "usr_123",
        "usr_test_user_001",
        "usr_" + "x" * 49,  # boundary: 53 chars total
    ])
    def test_user_id_valid(self, uid):
        assert validate_user_id(uid) == uid

    @pytest.mark.parametrize("uid", [
        "abc",            # no prefix
        "usr_",           # empty body
        "usr_xy",         # too short (body < 3 char)
        "user_abc",       # wrong prefix
        "USR_abc",        # uppercase prefix
        "usr_abc!",       # special char
        "usr_" + "x" * 50,  # too long
    ])
    def test_user_id_invalid(self, uid):
        with pytest.raises(ValueError):
            validate_user_id(uid)

    @pytest.mark.parametrize("sid", [
        "sess_abc1",      # boundary: body = 4 chars
        "sess_test123",
        "sess_a_b_c_d_e_f",
    ])
    def test_session_id_valid(self, sid):
        assert validate_session_id(sid) == sid

    @pytest.mark.parametrize("sid", [
        "session_abc",
        "sess_abc",       # body too short (< 4)
        "sess_",
        "abc",
    ])
    def test_session_id_invalid(self, sid):
        with pytest.raises(ValueError):
            validate_session_id(sid)

    def test_file_id_valid(self):
        assert validate_file_id("file_abc123") == "file_abc123"
        assert validate_file_id("  file_xyz  ") == "file_xyz"  # strip whitespace

    def test_file_id_invalid(self):
        with pytest.raises(ValueError):
            validate_file_id("file-abc")  # dash not allowed
        with pytest.raises(ValueError):
            validate_file_id("notfile_abc")
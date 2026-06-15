"""
test_api.py — Test otomatis semua endpoint RAG Reduka API.

Cara pakai:
    python test_api.py                        # test semua, skip chat LLM
    python test_api.py --with-llm             # include endpoint /chat (panggil Gemini)
    python test_api.py --base-url http://...  # URL berbeda (default: localhost:8000)

Dependensi:
    pip install httpx   (sudah include di environment.yml)
"""

import argparse
import json
import sys
import time
from typing import Any

import httpx

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_API_KEY  = "reduka-secret-key"   # Harus sama dengan API_KEY di .env

# Akun test — pastikan format sesuai regex di schemas.py
TEST_USER_ID    = "usr_test001"
TEST_SESSION_ID = "sess_testxyz1234"

# ── Warna terminal ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"
SKIP = f"{YELLOW}SKIP{RESET}"


# ── Helpers ───────────────────────────────────────────────────────────────────

class TestRunner:
    def __init__(self, base_url: str, api_key: str):
        self.base    = base_url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.client  = httpx.Client(base_url=self.base, headers=self.headers, timeout=60)
        self.results : list[dict] = []

    def _record(self, name: str, ok: bool, status: int, detail: str, latency_ms: int):
        self.results.append({
            "name": name, "ok": ok,
            "status": status, "detail": detail, "latency_ms": latency_ms,
        })
        tag = PASS if ok else FAIL
        print(f"  [{tag}] {name:<52} {status:>3}  {latency_ms:>5}ms  {DIM}{detail[:60]}{RESET}")

    def check(
        self,
        name:            str,
        method:          str,
        path:            str,
        expect_status:   int,
        body:            Any  = None,
        files:           Any  = None,
        extra_headers:   dict = None,
        assert_keys:     list = None,   # pastikan key ada di response JSON
        assert_values:   dict = None,   # pastikan key=value di response JSON
        skip:            bool = False,
        skip_reason:     str  = "",
    ) -> dict:
        if skip:
            print(f"  [{SKIP}] {name:<52}  {DIM}{skip_reason}{RESET}")
            self.results.append({"name": name, "ok": True, "status": 0,
                                 "detail": "skipped", "latency_ms": 0})
            return {}

        headers = dict(self.headers)
        if extra_headers:
            headers.update(extra_headers)

        t0 = time.perf_counter()
        try:
            if files:
                # multipart — jangan set Content-Type manual
                resp = self.client.request(method, path, data=body, files=files, headers=headers)
            elif body is not None:
                resp = self.client.request(method, path, json=body, headers=headers)
            else:
                resp = self.client.request(method, path, headers=headers)
        except Exception as e:
            self._record(name, False, 0, f"Connection error: {e}", 0)
            return {}

        latency_ms = int((time.perf_counter() - t0) * 1000)

        try:
            data = resp.json()
        except Exception:
            data = {}

        ok = resp.status_code == expect_status

        # Validasi isi response
        if ok and assert_keys:
            for k in assert_keys:
                if k not in data:
                    ok = False
                    self._record(name, False, resp.status_code,
                                 f"Key '{k}' tidak ada di response", latency_ms)
                    return data

        if ok and assert_values:
            for k, v in assert_values.items():
                if data.get(k) != v:
                    ok = False
                    self._record(name, False, resp.status_code,
                                 f"Expected {k}={v!r}, got {data.get(k)!r}", latency_ms)
                    return data

        detail = str(data)[:80] if not ok else (
            ", ".join(f"{k}={data[k]!r}" for k in (assert_keys or [])[:3]) or "OK"
        )
        self._record(name, ok, resp.status_code, detail, latency_ms)
        return data

    def summary(self):
        total   = len(self.results)
        passed  = sum(1 for r in self.results if r["ok"])
        failed  = total - passed
        skipped = sum(1 for r in self.results if r["detail"] == "skipped")

        print(f"\n{'═'*72}")
        print(f"{BOLD}  SUMMARY{RESET}")
        print(f"{'═'*72}")
        print(f"  Total  : {total}")
        print(f"  {GREEN}Passed : {passed}{RESET}")
        if failed:
            print(f"  {RED}Failed : {failed}{RESET}")
            print(f"\n  {BOLD}Failed tests:{RESET}")
            for r in self.results:
                if not r["ok"] and r["detail"] != "skipped":
                    print(f"    {RED}✗{RESET} {r['name']}")
                    print(f"        status={r['status']}, detail={r['detail']}")
        if skipped:
            print(f"  {YELLOW}Skipped: {skipped}{RESET}")
        print(f"{'═'*72}\n")
        return failed == 0

    def close(self):
        self.client.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SUITES
# ═══════════════════════════════════════════════════════════════════════════════

def run_health(t: TestRunner):
    print(f"\n{BOLD}── [1] HEALTH ──────────────────────────────────────────────────────{RESET}")

    t.check(
        "GET /v1/health  → 200",
        "GET", "/v1/health", 200,
        assert_keys=["status", "timestamp"],
        assert_values={"status": "ok"},
    )
    data = t.check(
        "GET /v1/health/detailed  → 200",
        "GET", "/v1/health/detailed", 200,
        assert_keys=["status", "components"],
    )
    if data:
        components = data.get("components", {})
        for comp, info in components.items():
            status_val = info.get("status", "?")
            color      = GREEN if status_val == "ok" else RED
            latency    = f"{info.get('latency_ms', '-')}ms" if info.get("latency_ms") else ""
            detail     = info.get("detail", "")[:60]
            print(f"      {color}{'●'}{RESET} {comp:<12} {status_val:<8} {latency:<8} {DIM}{detail}{RESET}")


def run_auth(t: TestRunner):
    print(f"\n{BOLD}── [2] AUTH & VALIDASI ─────────────────────────────────────────────{RESET}")

    # Tanpa API key
    t.check(
        "POST /v1/chat tanpa API key  → 401",
        "POST", "/v1/chat",
        expect_status=401,
        body={"user_id": TEST_USER_ID, "query": "test"},
        extra_headers={"X-API-Key": ""},
    )
    # API key salah
    t.check(
        "POST /v1/chat API key salah  → 403",
        "POST", "/v1/chat",
        expect_status=403,
        body={"user_id": TEST_USER_ID, "query": "test"},
        extra_headers={"X-API-Key": "wrong-key-xyz"},
    )
    # user_id format salah
    t.check(
        "POST /v1/chat user_id invalid  → 422",
        "POST", "/v1/chat",
        expect_status=422,
        body={"user_id": "badformat", "query": "test"},
    )
    # query terlalu pendek
    t.check(
        "POST /v1/chat query terlalu pendek  → 422",
        "POST", "/v1/chat",
        expect_status=422,
        body={"user_id": TEST_USER_ID, "query": "x"},
    )


def run_chat(t: TestRunner, with_llm: bool):
    print(f"\n{BOLD}── [3] CHAT ─────────────────────────────────────────────────────────{RESET}")

    if not with_llm:
        t.check(
            "POST /v1/chat → 200  (LLM call)",
            "POST", "/v1/chat", 200,
            skip=True, skip_reason="Gunakan --with-llm untuk mengaktifkan (memanggil Gemini API)",
        )
        return

    # Pertama: query dengan nomor soal (harusnya kena L1 Regex + exact search)
    data = t.check(
        "POST /v1/chat soal spesifik  → 200",
        "POST", "/v1/chat", 200,
        body={
            "user_id":    TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "query":      "Jelaskan soal nomor 3 penalaran umum kak.",
        },
        assert_keys=["session_id", "answer", "sources", "meta"],
    )
    if data:
        print(f"      {DIM}session_id : {data.get('session_id')}{RESET}")
        print(f"      {DIM}latency_ms : {data.get('meta', {}).get('latency_ms')}ms{RESET}")
        print(f"      {DIM}sources    : {len(data.get('sources', []))} dokumen{RESET}")
        ans = data.get("answer", "")
        print(f"      {DIM}answer     : {ans[:120]}...{RESET}")

    # Follow-up: tanpa nomor soal (harusnya kena L2 similarity atau L3 Redis cache)
    t.check(
        "POST /v1/chat follow-up (tanpa nomor)  → 200",
        "POST", "/v1/chat", 200,
        body={
            "user_id":    TEST_USER_ID,
            "session_id": TEST_SESSION_ID,
            "query":      "Kenapa bisa jawabannya seperti itu kak?",
        },
        assert_keys=["session_id", "answer"],
    )

    # Auto-generate session_id (tidak kirim session_id)
    data2 = t.check(
        "POST /v1/chat tanpa session_id (auto-generate)  → 200",
        "POST", "/v1/chat", 200,
        body={
            "user_id": TEST_USER_ID,
            "query":   "Apa itu penalaran umum?",
        },
        assert_keys=["session_id", "answer"],
    )
    if data2:
        sid = data2.get("session_id", "")
        ok_format = sid.startswith("sess_")
        print(f"      {GREEN if ok_format else RED}{'✓' if ok_format else '✗'}{RESET} "
              f"session_id auto-generated: {sid}")


def run_session(t: TestRunner, with_llm: bool):
    print(f"\n{BOLD}── [4] SESSION ──────────────────────────────────────────────────────{RESET}")

    if not with_llm:
        t.check(
            "GET /v1/session/{user_id}/history  → 200",
            "GET", f"/v1/session/{TEST_USER_ID}/history?session_id={TEST_SESSION_ID}",
            expect_status=200, skip=True,
            skip_reason="Jalankan dengan --with-llm agar ada history dulu",
        )
    else:
        data = t.check(
            "GET /v1/session history  → 200",
            "GET", f"/v1/session/{TEST_USER_ID}/history?session_id={TEST_SESSION_ID}",
            expect_status=200,
            assert_keys=["user_id", "session_id", "messages", "message_count"],
        )
        if data:
            print(f"      {DIM}message_count : {data.get('message_count')}{RESET}")
            print(f"      {DIM}summary       : {(data.get('summary') or '(belum ada)')[:80]}{RESET}")

    # History sesi tidak ada → 404
    t.check(
        "GET /v1/session sesi tidak ada  → 404",
        "GET", f"/v1/session/{TEST_USER_ID}/history?session_id=sess_tidakadaxyz",
        expect_status=404,
    )

    # user_id format salah → 422
    t.check(
        "GET /v1/session user_id invalid  → 422",
        "GET", "/v1/session/badid/history?session_id=sess_abc1234",
        expect_status=422,
    )

    if with_llm:
        # Clear session setelah test
        t.check(
            "DELETE /v1/session/{user_id}  → 200",
            "DELETE", f"/v1/session/{TEST_USER_ID}?session_id={TEST_SESSION_ID}",
            expect_status=200,
            assert_keys=["user_id", "cleared"],
        )


def run_documents(t: TestRunner):
    print(f"\n{BOLD}── [5] DOCUMENTS ────────────────────────────────────────────────────{RESET}")

    # List dokumen
    data = t.check(
        "GET /v1/documents  → 200",
        "GET", "/v1/documents",
        expect_status=200,
        assert_keys=["total", "page", "limit", "items"],
    )
    if data:
        print(f"      {DIM}total dokumen terdaftar: {data.get('total')}{RESET}")

    # Upload file tidak valid (bukan PDF)
    t.check(
        "POST /v1/documents/upload bukan PDF  → 400",
        "POST", "/v1/documents/upload",
        expect_status=400,
        files={"file": ("soal_test.pdf", b"bukan pdf sama sekali", "application/pdf")},
        body={"doc_type": "soal", "jenis_ujian": "Tryout Test"},
    )

    # Upload nama file tidak sesuai pola
    t.check(
        "POST /v1/documents/upload nama file salah  → 400",
        "POST", "/v1/documents/upload",
        expect_status=400,
        files={"file": ("namafileaaneh.pdf", b"%PDF-1.4 test", "application/pdf")},
        body={"doc_type": "soal", "jenis_ujian": "Tryout Test"},
    )

    # doc_type tidak valid
    t.check(
        "POST /v1/documents/upload doc_type invalid  → 400",
        "POST", "/v1/documents/upload",
        expect_status=400,
        files={"file": ("soal_test.pdf", b"%PDF-1.4 test", "application/pdf")},
        body={"doc_type": "invalid", "jenis_ujian": "Tryout Test"},
    )

    # Ingest request kosong → 422
    t.check(
        "POST /v1/documents/ingest body kosong  → 422",
        "POST", "/v1/documents/ingest",
        expect_status=422,
        body={"file_ids": [], "ingest_all_pending": False},
    )

    # file_id tidak ada → 404
    t.check(
        "DELETE /v1/documents/file_tidakada  → 404",
        "DELETE", "/v1/documents/file_tidakadaxyz123",
        expect_status=404,
    )

    # file_id format salah → 422
    t.check(
        "DELETE /v1/documents/badformat  → 422",
        "DELETE", "/v1/documents/badformat",
        expect_status=422,
    )

    # Polling job tidak ada → 404
    t.check(
        "GET /v1/documents/ingest/job_tidakada  → 404",
        "GET", "/v1/documents/ingest/job_00000000_aaaaaa",
        expect_status=404,
    )


def run_rate_limit(t: TestRunner, with_llm: bool):
    """Kirim request berulang untuk memicu 429."""
    print(f"\n{BOLD}── [6] RATE LIMIT ────────────────────────────────────────────────── {RESET}")

    if not with_llm:
        t.check(
            "Rate limit 429 simulation",
            "POST", "/v1/chat", 429,
            skip=True,
            skip_reason="Skip tanpa --with-llm (perlu 30+ request ke /chat)",
        )
        return

    print(f"  {YELLOW}Mengirim 32 request berturut-turut ke /v1/chat...{RESET}")
    hit_429 = False
    for i in range(32):
        resp = t.client.post(
            "/v1/chat",
            json={"user_id": "usr_ratelimitest", "query": f"test rate limit {i}"},
        )
        if resp.status_code == 429:
            hit_429 = True
            print(f"  {GREEN}429 diterima pada request ke-{i+1}{RESET}")
            break

    if hit_429:
        t.results.append({"name": "Rate limit 429 triggered", "ok": True,
                          "status": 429, "detail": "OK", "latency_ms": 0})
        print(f"  [{PASS}] Rate limit 429 triggered")
    else:
        t.results.append({"name": "Rate limit 429 triggered", "ok": False,
                          "status": 200, "detail": "429 tidak pernah muncul dalam 32 request", "latency_ms": 0})
        print(f"  [{FAIL}] Rate limit tidak bekerja — 429 tidak pernah muncul")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test otomatis RAG Reduka API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help=f"Base URL server (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help="API key (harus sama dengan .env)")
    parser.add_argument("--with-llm", action="store_true",
                        help="Aktifkan test yang memanggil Gemini API (ada biaya token)")
    args = parser.parse_args()

    print(f"\n{'═'*72}")
    print(f"{BOLD}  RAG REDUKA — API TEST SUITE{RESET}")
    print(f"{'═'*72}")
    print(f"  Base URL  : {args.base_url}")
    print(f"  API Key   : {args.api_key[:8]}{'*' * (len(args.api_key) - 8)}")
    print(f"  LLM mode  : {'ON (memanggil Gemini)' if args.with_llm else 'OFF (skip endpoint chat)'}")
    print(f"  User ID   : {TEST_USER_ID}")
    print(f"  Session ID: {TEST_SESSION_ID}")
    print(f"{'─'*72}")
    print(f"  {'TEST NAME':<52} {'STS':>3}  {'MS':>6}  DETAIL")
    print(f"{'─'*72}")

    t = TestRunner(args.base_url, args.api_key)

    try:
        run_health(t)
        run_auth(t)
        run_chat(t, args.with_llm)
        run_session(t, args.with_llm)
        run_documents(t)
        run_rate_limit(t, args.with_llm)
    finally:
        t.close()

    success = t.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
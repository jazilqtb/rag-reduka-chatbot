"""
History Service — Manajemen Riwayat Percakapan di Redis

Diekstrak dari ChatService di Stage 5 supaya:
  - ChatService fokus ke orkestrasi LLM + retrieve
  - History logic (load/save/summarize) bisa di-test independen
  - Kalau nanti pindah dari Redis ke Postgres untuk archive, hanya file ini
    yang berubah

Hybrid history strategy:
  - MAX_RECENT_MESSAGES pesan terakhir dikirim FULL ke LLM
  - Pesan lama yang di luar window di-ringkas via rolling summary
  - Summary di-update incremental setiap SUMMARY_TRIGGER pesan baru

Redis key structure (per user + session):
  chat:messages:{user_id}:{session_id}        → Redis LIST of JSON messages
  chat:summary:{user_id}:{session_id}         → STRING ringkasan
  chat:summarized_upto:{user_id}:{session_id} → STRING int (indeks terakhir yg di-summary)

TTL semua key: REDIS_CHAT_TTL (default 24 jam), di-refresh setiap pesan baru.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis import Redis

from src.core.config import settings
from src.core.logger import get_logger


# ── Konstanta ─────────────────────────────────────────────────────────────────
MAX_RECENT_MESSAGES = 10   # Pesan terakhir yang dikirim full ke LLM
SUMMARY_TRIGGER     = 20   # Picu summarize setelah history melebihi N pesan


class HistoryService:
    """
    Kelola riwayat percakapan di Redis + rolling summary.

    Dependency:
      redis : Redis client (singleton dari deps.py)
      llm   : LLM untuk summarization (boleh sama dengan main chat LLM)
    """

    def __init__(self, redis: Redis, llm: Any):
        self.redis  = redis
        self.llm    = llm
        self.logger = get_logger("HistoryService")

    # ══════════════════════════════════════════════════════════════════════════
    # REDIS KEY HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _key_messages(self, user_id: str, session_id: str) -> str:
        return f"chat:messages:{user_id}:{session_id}"

    def _key_summary(self, user_id: str, session_id: str) -> str:
        return f"chat:summary:{user_id}:{session_id}"

    def _key_summarized_upto(self, user_id: str, session_id: str) -> str:
        return f"chat:summarized_upto:{user_id}:{session_id}"

    def _refresh_ttl(self, user_id: str, session_id: str) -> None:
        """Refresh TTL semua key session agar tidak expired selama aktif dipakai."""
        ttl = settings.REDIS_CHAT_TTL
        for key in [
            self._key_messages(user_id, session_id),
            self._key_summary(user_id, session_id),
            self._key_summarized_upto(user_id, session_id),
        ]:
            try:
                self.redis.expire(key, ttl)
            except Exception as e:
                self.logger.warning(f"[History] Gagal refresh TTL '{key}': {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # MESSAGE PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def load_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """
        Muat semua pesan sesi dari Redis LIST sebagai LangChain BaseMessage.
        Setiap entry Redis adalah JSON: {role, content, timestamp}
        """
        key = self._key_messages(user_id, session_id)
        try:
            raw_list = self.redis.lrange(key, 0, -1)
        except Exception as e:
            self.logger.warning(f"[History] Gagal load messages dari Redis: {e}")
            return []

        messages: List[BaseMessage] = []
        for raw in raw_list:
            try:
                data = json.loads(raw)
                if data["role"] == "human":
                    messages.append(HumanMessage(content=data["content"]))
                else:
                    messages.append(AIMessage(content=data["content"]))
            except Exception:
                continue
        return messages

    def load_messages_with_meta(self, user_id: str, session_id: str) -> List[Dict]:
        """
        Muat semua pesan beserta timestamp — dipakai oleh session endpoint.
        Mengembalikan list dict mentah (tidak di-convert ke BaseMessage).
        """
        key = self._key_messages(user_id, session_id)
        try:
            raw_list = self.redis.lrange(key, 0, -1)
            return [json.loads(r) for r in raw_list]
        except Exception as e:
            self.logger.warning(f"[History] Gagal load messages with meta: {e}")
            return []

    def append_exchange(
        self,
        user_id:    str,
        session_id: str,
        query:      str,
        answer:     str,
    ) -> None:
        """
        Tambahkan pasangan human+AI message ke Redis LIST dalam satu pipeline.
        TTL key di-refresh.
        """
        key = self._key_messages(user_id, session_id)
        ts  = datetime.utcnow().isoformat()
        try:
            pipe = self.redis.pipeline()
            pipe.rpush(
                key,
                json.dumps({"role": "human", "content": query,  "timestamp": ts}, ensure_ascii=False),
            )
            pipe.rpush(
                key,
                json.dumps({"role": "ai",    "content": answer, "timestamp": ts}, ensure_ascii=False),
            )
            pipe.expire(key, settings.REDIS_CHAT_TTL)
            pipe.execute()
        except Exception as e:
            self.logger.warning(f"[History] Gagal append exchange: {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # ROLLING SUMMARY
    # ══════════════════════════════════════════════════════════════════════════

    def try_summarize(self, user_id: str, session_id: str) -> None:
        """
        Jika total pesan >= SUMMARY_TRIGGER, ringkas pesan lama yang berada
        di luar window MAX_RECENT_MESSAGES secara incremental.

        Biaya: 1× LLM call ringan (~200 token), hanya dipicu pada sesi panjang.
        """
        messages = self.load_messages(user_id, session_id)
        if len(messages) < SUMMARY_TRIGGER:
            return

        cutoff = len(messages) - MAX_RECENT_MESSAGES

        try:
            already_done = int(self.redis.get(self._key_summarized_upto(user_id, session_id)) or 0)
        except Exception:
            already_done = 0

        if cutoff <= already_done:
            return

        new_messages = messages[already_done:cutoff]
        new_convo    = "\n".join(
            f"{'Siswa' if isinstance(m, HumanMessage) else 'Tutor AI'}: {m.content}"
            for m in new_messages
        )

        existing_summary = self.redis.get(self._key_summary(user_id, session_id)) or ""
        if existing_summary:
            prompt_text = (
                f"Ringkasan percakapan sebelumnya:\n{existing_summary}\n\n"
                f"Lanjutan percakapan baru:\n{new_convo}\n\n"
                f"Perbarui ringkasan menjadi maksimal 3 kalimat. "
                f"Fokus: soal nomor berapa dan materi apa yang sudah dibahas."
            )
        else:
            prompt_text = (
                f"Percakapan antara siswa dan Tutor AI:\n{new_convo}\n\n"
                f"Buat ringkasan singkat (maks 3 kalimat). "
                f"Fokus: soal nomor berapa dan materi apa yang sudah dibahas."
            )

        try:
            response = self.llm.invoke(prompt_text)
            summary  = response.content.strip()
            pipe = self.redis.pipeline()
            pipe.set(self._key_summary(user_id, session_id),         summary, ex=settings.REDIS_CHAT_TTL)
            pipe.set(self._key_summarized_upto(user_id, session_id), str(cutoff), ex=settings.REDIS_CHAT_TTL)
            pipe.execute()
            self.logger.debug(
                f"[History] Summary diperbarui untuk '{user_id}:{session_id}' "
                f"(s/d pesan ke-{cutoff})."
            )
        except Exception as e:
            self.logger.warning(f"[History] Gagal membuat summary: {e}")

    def get_llm_context(
        self,
        user_id: str,
        session_id: str,
    ) -> Tuple[str, List[BaseMessage]]:
        """
        Kembalikan (summary_lama, pesan_terakhir) untuk dimasukkan ke prompt LLM.

        summary_lama   : ringkasan percakapan di luar window MAX_RECENT_MESSAGES
        pesan_terakhir : MAX_RECENT_MESSAGES pesan terbaru (full BaseMessage)
        """
        messages = self.load_messages(user_id, session_id)
        recent = (
            messages[-MAX_RECENT_MESSAGES:]
            if len(messages) > MAX_RECENT_MESSAGES
            else messages
        )
        try:
            summary = self.redis.get(self._key_summary(user_id, session_id)) or ""
        except Exception:
            summary = ""
        return summary, recent

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION OPERATIONS (dipakai endpoint /v1/session)
    # ══════════════════════════════════════════════════════════════════════════

    def get_summary(self, user_id: str, session_id: str) -> Optional[str]:
        """Kembalikan summary sesi jika ada. None jika belum ada/expired."""
        try:
            return self.redis.get(self._key_summary(user_id, session_id))
        except Exception:
            return None

    def clear_session(self, user_id: str, session_id: str) -> List[str]:
        """
        Hapus semua key Redis terkait satu sesi:
          - history pesan
          - summary
          - summarized_upto counter
          - entity cache (RetrieveService)
          - context cache (RetrieveService)

        Mengembalikan list label key yang berhasil dihapus.
        """
        cleared: List[str] = []
        keys_chat = {
            "history":    self._key_messages(user_id, session_id),
            "summary":    self._key_summary(user_id, session_id),
            "summarized": self._key_summarized_upto(user_id, session_id),
        }
        # Key milik RetrieveService (entity & context per-user, bukan per-session)
        keys_retrieve = {
            "entity_cache":  f"entity:{user_id}",
            "context_cache": f"context:{user_id}",
        }

        all_keys = {**keys_chat, **keys_retrieve}
        for label, key in all_keys.items():
            try:
                deleted = self.redis.delete(key)
                if deleted:
                    cleared.append(label)
            except Exception as e:
                self.logger.warning(f"[Session] Gagal hapus key '{key}': {e}")

        self.logger.info(
            f"[Session] Cleared untuk user='{user_id}', session='{session_id}': {cleared}"
        )
        return cleared
"""
RAG Chat Service — Tutor AI Reduka (UTBK SNBT)

Tanggung jawab ChatService HANYA:
  1. Orkestrasi: panggil RetrieveService → format docs → panggil LLM
  2. Hybrid history management persisten di Redis (full messages + rolling summary)
  3. Retry logic dengan exponential backoff untuk semua LLM call

History Redis key structure (per user + session):
  chat:messages:{user_id}:{session_id}       → Redis LIST of JSON messages
  chat:summary:{user_id}:{session_id}        → string ringkasan
  chat:summarized_upto:{user_id}:{session_id}→ int string (indeks terakhir yang di-summary)
  TTL semua key: REDIS_CHAT_TTL (default 24 jam), di-refresh setiap pesan baru.
"""

import yaml
import json
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from redis import Redis

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.schemas import ChatResponse, ResponseMeta, SourceItem
from src.services.retrieve_service import RetrieveService

# ── Konstanta ─────────────────────────────────────────────────────────────────
MAX_RETRIES         = 3    # Maksimal percobaan ulang LLM
BASE_RETRY_DELAY    = 2    # Detik awal delay (×2 setiap percobaan)
MAX_RECENT_MESSAGES = 10   # Pesan terakhir yang dikirim penuh ke LLM
SUMMARY_TRIGGER     = 20   # Picu summarize setelah history melebihi N pesan


class ChatService:
    def __init__(self):
        self.logger = get_logger("ChatService")

        # ── Redis ─────────────────────────────────────────────────────────────
        self.redis = Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )

        # ── Retrieve module ───────────────────────────────────────────────────
        self.retrieve_service = RetrieveService()

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
        )

        # ── Prompt ───────────────────────────────────────────────────────────
        self.chat_prompt = ""
        try:
            with open(settings.PROMPT_DIR, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.chat_prompt = data.get("rag_chat_prompt", "")
            self.logger.info("Prompt berhasil dimuat.")
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}. Menggunakan prompt default.")
            self.chat_prompt = (
                "Kamu adalah Tutor AI dari Bimbel Reduka yang membantu siswa "
                "memahami soal UTBK SNBT. Jawab HANYA berdasarkan konteks yang diberikan."
            )

        self.logger.info("ChatService berhasil diinisialisasi.")

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
            self.redis.expire(key, ttl)

    # ══════════════════════════════════════════════════════════════════════════
    # MESSAGE PERSISTENCE
    # ══════════════════════════════════════════════════════════════════════════

    def _load_messages(self, user_id: str, session_id: str) -> List[BaseMessage]:
        """
        Muat semua pesan sesi dari Redis LIST.
        Setiap entry adalah JSON: {role, content, timestamp}
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

    def _load_messages_with_meta(self, user_id: str, session_id: str) -> List[Dict]:
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

    def _append_messages(
        self,
        user_id:    str,
        session_id: str,
        query:      str,
        answer:     str,
    ) -> None:
        """Tambahkan pasangan human+AI message ke Redis LIST."""
        key = self._key_messages(user_id, session_id)
        ts  = datetime.utcnow().isoformat()
        pipe = self.redis.pipeline()
        pipe.rpush(key, json.dumps({"role": "human", "content": query,  "timestamp": ts}, ensure_ascii=False))
        pipe.rpush(key, json.dumps({"role": "ai",    "content": answer, "timestamp": ts}, ensure_ascii=False))
        pipe.expire(key, settings.REDIS_CHAT_TTL)
        pipe.execute()

    # ══════════════════════════════════════════════════════════════════════════
    # HYBRID HISTORY MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _try_summarize_old_messages(self, user_id: str, session_id: str) -> None:
        """
        Jika total pesan >= SUMMARY_TRIGGER, ringkas pesan lama yang berada
        di luar window MAX_RECENT_MESSAGES secara incremental.

        Biaya: 1× LLM call ringan (~200 token), hanya dipicu pada sesi panjang.
        """
        messages = self._load_messages(user_id, session_id)
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
            pipe.set(self._key_summary(user_id, session_id), summary, ex=settings.REDIS_CHAT_TTL)
            pipe.set(self._key_summarized_upto(user_id, session_id), str(cutoff), ex=settings.REDIS_CHAT_TTL)
            pipe.execute()
            self.logger.debug(
                f"[History] Summary diperbarui untuk '{user_id}:{session_id}' "
                f"(s/d pesan ke-{cutoff})."
            )
        except Exception as e:
            self.logger.warning(f"[History] Gagal membuat summary: {e}")

    def _get_llm_history_context(
        self, user_id: str, session_id: str
    ) -> Tuple[str, List[BaseMessage]]:
        """
        Kembalikan (summary_lama, pesan_terakhir) untuk prompt LLM.
          summary_lama   : ringkasan semua percakapan di luar window
          pesan_terakhir : MAX_RECENT_MESSAGES pesan terbaru (full)
        """
        messages = self._load_messages(user_id, session_id)
        recent   = (messages[-MAX_RECENT_MESSAGES:]
                    if len(messages) > MAX_RECENT_MESSAGES else messages)
        try:
            summary = self.redis.get(self._key_summary(user_id, session_id)) or ""
        except Exception:
            summary = ""
        return summary, recent

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _build_sources(docs: List[Document]) -> List[SourceItem]:
        return [
            SourceItem(
                subject     = doc.metadata.get("subject", ""),
                jenis_ujian = doc.metadata.get("jenis_ujian", ""),
                id_soal     = str(doc.metadata.get("id_soal", "")),
                source      = doc.metadata.get("source", ""),
            )
            for doc in docs
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # RETRY LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _invoke_with_retry(self, chain: Any, inputs: Dict) -> str:
        last_error: Exception = RuntimeError("Unknown error")
        for attempt in range(MAX_RETRIES):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                last_error = e
                wait       = BASE_RETRY_DELAY * (2 ** attempt)
                if "429" in str(e):
                    self.logger.warning(
                        f"[LLM] Rate limit (429). Retry {attempt+1}/{MAX_RETRIES} dalam {wait}s..."
                    )
                else:
                    self.logger.warning(
                        f"[LLM] Error: {e}. Retry {attempt+1}/{MAX_RETRIES} dalam {wait}s..."
                    )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM gagal setelah {MAX_RETRIES} percobaan. Error terakhir: {last_error}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RESPONSE GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def generate_response(
        self,
        query:      str,
        user_id:    str,
        session_id: str,
    ) -> ChatResponse:
        self.logger.info(f"[Chat] Memproses query — user: '{user_id}', session: '{session_id}'")
        t_start = time.perf_counter()

        # ── 1. Retrieve konteks ───────────────────────────────────────────────
        docs: List[Document] = self.retrieve_service.search(user_id=user_id, query=query)

        context_text = self._format_docs(docs) if docs else (
            "Tidak ada konteks spesifik yang ditemukan untuk pertanyaan ini."
        )
        sources_list = self._build_sources(docs)

        # ── 2. Update summary history jika perlu ─────────────────────────────
        self._try_summarize_old_messages(user_id, session_id)

        # ── 3. Bangun konteks history untuk LLM ──────────────────────────────
        old_summary, recent_messages = self._get_llm_history_context(user_id, session_id)

        # ── 4. Bangun system content ──────────────────────────────────────────
        system_content = self.chat_prompt + f"\n\nKonteks Referensi:\n{context_text}"
        if old_summary:
            system_content += (
                f"\n\nRingkasan percakapan sebelumnya dengan siswa ini:\n{old_summary}"
            )

        # ── 5. Prompt + chain ─────────────────────────────────────────────────
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_content}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        chain = prompt | self.llm | StrOutputParser()

        # ── 6. Invoke dengan retry ────────────────────────────────────────────
        try:
            answer = self._invoke_with_retry(
                chain,
                {
                    "system_content": system_content,
                    "chat_history":   recent_messages,
                    "input":          query,
                },
            )

            # ── 7. Persist history ke Redis ───────────────────────────────────
            self._append_messages(user_id, session_id, query, answer)
            self._refresh_ttl(user_id, session_id)

            latency_ms = int((time.perf_counter() - t_start) * 1000)
            self.logger.info(
                f"[Chat] Respon OK — user: '{user_id}', session: '{session_id}', "
                f"latency: {latency_ms}ms"
            )

            return ChatResponse(
                session_id = session_id,
                answer     = answer,
                sources    = sources_list,
                meta       = ResponseMeta(latency_ms=latency_ms),
            )

        except Exception as e:
            self.logger.error(
                f"[Chat] Gagal generate respon — user: '{user_id}', "
                f"session: '{session_id}': {e}"
            )
            latency_ms = int((time.perf_counter() - t_start) * 1000)
            return ChatResponse(
                session_id = session_id,
                answer     = "Maaf, terjadi kesalahan saat memproses pertanyaanmu. Coba tanyakan lagi ya! 🙏",
                sources    = [],
                meta       = ResponseMeta(latency_ms=latency_ms),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION MANAGEMENT (dipakai oleh session endpoint)
    # ══════════════════════════════════════════════════════════════════════════

    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict]:
        """Kembalikan semua pesan beserta metadata untuk session endpoint."""
        return self._load_messages_with_meta(user_id, session_id)

    def get_session_summary(self, user_id: str, session_id: str) -> Optional[str]:
        """Kembalikan summary sesi jika ada."""
        try:
            return self.redis.get(self._key_summary(user_id, session_id))
        except Exception:
            return None

    def clear_session(self, user_id: str, session_id: str) -> List[str]:
        """
        Hapus semua data sesi dari Redis:
          - history pesan
          - summary
          - summarized_upto counter
          - entity cache (RetrieveService)
          - context cache (RetrieveService)

        Mengembalikan list key yang berhasil dihapus.
        """
        cleared = []
        keys_chat = {
            "history":     self._key_messages(user_id, session_id),
            "summary":     self._key_summary(user_id, session_id),
            "summarized":  self._key_summarized_upto(user_id, session_id),
        }
        # Key milik RetrieveService
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


# ══════════════════════════════════════════════════════════════════════════════
# TESTING — 3 MODE INTERAKTIF
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    import json
    from src.core.security import generate_session_id

    PRICE_INPUT_PER_1M  = 0.075
    PRICE_OUTPUT_PER_1M = 0.30
    KURS_IDR            = 16_000

    def _estimate_cost(input_chars: int, output_chars: int) -> dict:
        inp = input_chars  / 4
        out = output_chars / 4
        usd = (inp / 1_000_000 * PRICE_INPUT_PER_1M) + (out / 1_000_000 * PRICE_OUTPUT_PER_1M)
        return {"input_tokens": int(inp), "output_tokens": int(out),
                "cost_usd": usd, "cost_idr": usd * KURS_IDR}

    print("╔═══════════════════════════════════════════════════════╗")
    print("║        CHATBOT REDUKA — TUTOR AI UTBK SNBT            ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  Mode 1 : Chat murni       (tanpa log)                ║")
    print("║  Mode 2 : Chat + Log INFO  (ada pemisah seksi)        ║")
    print("║  Mode 3 : Full debug + Context + History + Biaya      ║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    while True:
        _m = input("Pilih mode [1/2/3]: ").strip()
        if _m in ("1", "2", "3"):
            MODE = int(_m)
            break
        print("  ⚠️  Input tidak valid.\n")

    if MODE == 1:
        logging.disable(logging.CRITICAL)
    elif MODE == 2:
        logging.disable(logging.DEBUG)
    else:
        logging.disable(logging.NOTSET)

    if MODE != 1:
        print("\n⏳ Menginisialisasi ChatService...\n")

    chat_service = ChatService()
    print()
    USER_ID    = input("Masukkan User ID (Enter = 'usr_test001'): ").strip() or "usr_test001"
    SESSION_ID = generate_session_id()

    print(f"\n{'─'*60}")
    print(f"  ✅  Mode {MODE} aktif  |  User: {USER_ID}  |  Session: {SESSION_ID}")
    print(f"  💡  Ketik 'exit' atau 'stop' untuk keluar")
    print(f"{'─'*60}\n")

    while True:
        try:
            query = input("🧑 Kamu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar. Semangat belajar! 💪")
            break

        if query.lower() in ("exit", "stop", "quit", "keluar"):
            print("Sampai jumpa! Semangat UTBK-nya! 🚀")
            break
        if not query:
            continue

        response = chat_service.generate_response(query=query, user_id=USER_ID, session_id=SESSION_ID)

        if MODE == 1:
            print(f"\n🤖 AI: {response.answer}\n")

        elif MODE == 2:
            S = "─" * 65
            print(f"\n{S}")
            print(f"  📥 QUERY    : {query}")
            print(f"  ⏱️  Latency  : {response.meta.latency_ms}ms")
            print(S)
            print(f"  💬 RESPON AI:")
            print(f"  {response.answer}")
            print(f"{S}\n")

        else:
            S1, S2 = "═" * 80, "─" * 80
            all_msgs = chat_service.get_session_messages(USER_ID, SESSION_ID)
            summary  = chat_service.get_session_summary(USER_ID, SESSION_ID) or ""

            _ctx_chars  = sum(len(s.source) for s in response.sources)
            _hist_chars = sum(len(m.get("content", "")) for m in all_msgs)
            _metrics    = _estimate_cost(
                len(query) + _ctx_chars + _hist_chars + len(chat_service.chat_prompt),
                len(response.answer)
            )

            print(f"\n{S1}")
            print("  🔍 [1] SOURCES:")
            if response.sources:
                print(json.dumps([s.model_dump() for s in response.sources], indent=4, ensure_ascii=False))
            else:
                print("  (Tidak ada konteks spesifik)")

            print(S2)
            print(f"  🕰️  [2] CHAT HISTORY ({len(all_msgs)} pesan):")
            for _msg in all_msgs[-10:]:
                _role = "🧑 User" if _msg["role"] == "human" else "🤖 AI  "
                _txt  = _msg["content"][:220] + "…" if len(_msg["content"]) > 220 else _msg["content"]
                print(f"    {_role}: {_txt}")
            if summary:
                print(f"\n  📝 SUMMARY: {summary}")

            print(S2)
            print("  💬 [3] RESPON AI:")
            print(f"  {response.answer}")

            print(S2)
            print("  📊 [4] METRIK:")
            print(f"    ⏱️  Latency    : {response.meta.latency_ms}ms")
            print(f"    🪙  Est. Token : {_metrics['input_tokens']} input | {_metrics['output_tokens']} output")
            print(f"    💵  Est. Biaya : ${_metrics['cost_usd']:.6f} (~Rp {_metrics['cost_idr']:.2f})")
            print(f"    📚  History    : {len(all_msgs)} pesan di Redis")
            print(f"    📝  Summary    : {'Ada ✓' if summary else 'Belum ada'}")
            print(f"{S1}\n")
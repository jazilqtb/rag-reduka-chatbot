"""
RAG Chat Service

Tanggung jawab ChatService sekarang lebih fokus (Stage 5):
  1. Orkestrasi: panggil RetrieveService → format docs → panggil LLM
  2. Retry logic dengan exponential backoff untuk LLM call
  3. Delegasi history management ke HistoryService

History management (Redis I/O + rolling summary) dipindahkan ke
src/services/history_service.py supaya:
  - File ini lebih kecil dan fokus
  - History bisa di-test tanpa LLM mahal
  - Single Responsibility Principle
"""

import time
import yaml
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from redis import Redis

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.schemas import ChatResponse, ResponseMeta, SourceItem
from src.services.history_service import HistoryService
from src.services.retrieve_service import RetrieveService


# ── Konstanta ─────────────────────────────────────────────────────────────────
MAX_RETRIES      = 3   # Maksimal percobaan ulang LLM
BASE_RETRY_DELAY = 2   # Detik awal delay (×2 setiap percobaan)


class ChatService:
    """
    Orchestrator untuk RAG chatbot:
      retrieve context → build prompt → invoke LLM → persist history.
    """

    def __init__(self):
        self.logger = get_logger("ChatService")

        # ── Redis (shared dengan HistoryService) ──────────────────────────────
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

        # ── History service (composes Redis + LLM) ────────────────────────────
        # LLM dipakai HistoryService HANYA untuk summarization (jarang dipicu).
        self.history_service = HistoryService(redis=self.redis, llm=self.llm)

        # ── Prompt ────────────────────────────────────────────────────────────
        self.chat_prompt = ""
        try:
            with open(settings.PROMPT_DIR, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.chat_prompt = data.get("rag_chat_prompt", "")
            self.logger.info("Prompt berhasil dimuat.")
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}. Menggunakan prompt default.")
            self.chat_prompt = (
                "Kamu adalah Tutor AI yang membantu siswa memahami soal UTBK SNBT. "
                "Jawab HANYA berdasarkan konteks yang diberikan."
            )

        self.logger.info("ChatService berhasil diinisialisasi.")

    # ══════════════════════════════════════════════════════════════════════════
    # STATIC HELPERS
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
    # RESPONSE GENERATION (orchestrator)
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

        # ── 2. Update summary history jika perlu (delegate) ──────────────────
        self.history_service.try_summarize(user_id, session_id)

        # ── 3. Bangun konteks history untuk LLM (delegate) ───────────────────
        old_summary, recent_messages = self.history_service.get_llm_context(user_id, session_id)

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

            # ── 7. Persist history (delegate) ──────────────────────────────────
            self.history_service.append_exchange(user_id, session_id, query, answer)
            self.history_service._refresh_ttl(user_id, session_id)

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
    # SESSION MANAGEMENT — Delegate ke HistoryService
    # Backward-compatible API supaya endpoint session.py tidak perlu diubah.
    # ══════════════════════════════════════════════════════════════════════════

    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict]:
        """Kembalikan semua pesan beserta metadata untuk session endpoint."""
        return self.history_service.load_messages_with_meta(user_id, session_id)

    def get_session_summary(self, user_id: str, session_id: str):
        """Kembalikan summary sesi jika ada."""
        return self.history_service.get_summary(user_id, session_id)

    def clear_session(self, user_id: str, session_id: str) -> List[str]:
        """Hapus semua data sesi dari Redis."""
        return self.history_service.clear_session(user_id, session_id)
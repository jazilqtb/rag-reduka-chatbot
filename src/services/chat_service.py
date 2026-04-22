"""
RAG Chat Service 
         
Tanggung jawab ChatService HANYA:
  1. Orkestrasi: panggil RetrieveService → format docs → panggil LLM
  2. Hybrid history management  (full storage in-memory + rolling summary)
  3. Retry logic dengan exponential backoff untuk semua LLM call

Seluruh pipeline retrieval (Regex NER → Exact Search → Similarity →
Redis Cache → LLM NER) dikelola sepenuhnya oleh RetrieveService.
"""

import yaml
import json
import time
import logging
from typing import Any, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

from src.core.config import settings
from src.core.logger import get_logger
from src.domain.schemas import ChatRequest, ChatResponse
from src.services.retrieve_service import RetrieveService

# ── Konstanta ─────────────────────────────────────────────────────────────────
MAX_RETRIES         = 3    # Maksimal percobaan ulang LLM
BASE_RETRY_DELAY    = 2    # Detik awal delay (×2 setiap percobaan)
MAX_RECENT_MESSAGES = 10   # Pesan terakhir yang dikirim penuh ke LLM
SUMMARY_TRIGGER     = 20   # Picu summarize setelah history melebihi N pesan


class ChatService:
    def __init__(self):
        self.logger = get_logger("ChatService")

        # ── Retrieve module ───────────────────────────────────────────────────
        self.retrieve_service = RetrieveService()

        # ── LLM (untuk generate respon & summarize history) ───────────────────
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
        )

        # ── Prompts ───────────────────────────────────────────────────────────
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

        # ── Session stores (in-memory) ────────────────────────────────────────
        self.store:            Dict[str, ChatMessageHistory] = {}  # Semua pesan
        self.summary_store:    Dict[str, str]                = {}  # Rolling summary
        self.summarized_up_to: Dict[str, int]                = {}  # Indeks terakhir summary

        self.logger.info("ChatService berhasil diinisialisasi.")

    # ══════════════════════════════════════════════════════════════════════════
    # SESSION HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def get_session_history(self, user_id: str) -> ChatMessageHistory:
        """Ambil atau buat ChatMessageHistory untuk user_id."""
        if user_id not in self.store:
            self.store[user_id] = ChatMessageHistory()
        return self.store[user_id]

    # ══════════════════════════════════════════════════════════════════════════
    # HYBRID HISTORY MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════

    def _try_summarize_old_messages(self, user_id: str) -> None:
        """
        Jika total pesan >= SUMMARY_TRIGGER, ringkas pesan-pesan lama yang
        berada di luar window MAX_RECENT_MESSAGES secara incremental.

        Biaya: 1× LLM call ringan (~200 token), hanya dipicu pada sesi panjang.
        Benefit: LLM tetap aware konteks soal yang pernah dibahas jauh sebelumnya.
        """
        messages = self.get_session_history(user_id).messages
        if len(messages) < SUMMARY_TRIGGER:
            return

        cutoff       = len(messages) - MAX_RECENT_MESSAGES
        already_done = self.summarized_up_to.get(user_id, 0)

        if cutoff <= already_done:
            return  # Tidak ada pesan baru untuk diringkas

        new_messages = messages[already_done:cutoff]
        new_convo    = "\n".join(
            f"{'Siswa' if m.type == 'human' else 'Tutor AI'}: {m.content}"
            for m in new_messages
        )

        existing_summary = self.summary_store.get(user_id, "")
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
            self.summary_store[user_id]    = response.content.strip()
            self.summarized_up_to[user_id] = cutoff
            self.logger.debug(
                f"[History] Summary diperbarui untuk '{user_id}' "
                f"(mencakup s/d pesan ke-{cutoff})."
            )
        except Exception as e:
            self.logger.warning(f"[History] Gagal membuat summary: {e}")

    def _get_llm_history_context(self, user_id: str) -> Tuple[str, List[BaseMessage]]:
        """
        Kembalikan (summary_lama, pesan_terakhir) untuk prompt LLM.
          summary_lama   : ringkasan semua percakapan di luar window terakhir
          pesan_terakhir : MAX_RECENT_MESSAGES pesan terbaru (full content)
        """
        messages = self.get_session_history(user_id).messages
        recent   = (messages[-MAX_RECENT_MESSAGES:]
                    if len(messages) > MAX_RECENT_MESSAGES else messages)
        summary  = self.summary_store.get(user_id, "")
        return summary, recent

    # ══════════════════════════════════════════════════════════════════════════
    # HELPER
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Gabungkan page_content semua dokumen menjadi satu string konteks."""
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _build_sources(docs: List[Document]) -> List[Dict]:
        """Ekstrak metadata ringkas dari dokumen untuk field sources response."""
        return [
            {
                "subject":     doc.metadata.get("subject", ""),
                "jenis_ujian": doc.metadata.get("jenis_ujian", ""),
                "id_soal":     doc.metadata.get("id_soal", ""),
                "source":      doc.metadata.get("source", ""),
            }
            for doc in docs
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # RETRY LOGIC
    # ══════════════════════════════════════════════════════════════════════════

    def _invoke_with_retry(self, chain: Any, inputs: Dict) -> str:
        """Invoke LangChain chain dengan exponential backoff retry."""
        last_error: Exception = RuntimeError("Unknown error")
        for attempt in range(MAX_RETRIES):
            try:
                return chain.invoke(inputs)
            except Exception as e:
                last_error = e
                wait       = BASE_RETRY_DELAY * (2 ** attempt)
                if "429" in str(e):
                    self.logger.warning(
                        f"[LLM] Rate limit (429). "
                        f"Retry {attempt+1}/{MAX_RETRIES} dalam {wait}s..."
                    )
                else:
                    self.logger.warning(
                        f"[LLM] Error: {e}. "
                        f"Retry {attempt+1}/{MAX_RETRIES} dalam {wait}s..."
                    )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM gagal setelah {MAX_RETRIES} percobaan. "
            f"Error terakhir: {last_error}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # RESPONSE GENERATION
    # ══════════════════════════════════════════════════════════════════════════

    def generate_response(self, query: str, user_id: str) -> ChatResponse:
        self.logger.info(f"[Chat] Memproses query untuk session: '{user_id}'")

        # ── 1. Retrieve konteks via RetrieveService ───────────────────────────
        docs: List[Document] = self.retrieve_service.search(
            user_id=user_id, query=query
        )

        context_text = self._format_docs(docs) if docs else (
            "Tidak ada konteks spesifik yang ditemukan untuk pertanyaan ini."
        )
        sources_list = self._build_sources(docs)

        # ── 2. Update summary history jika perlu ─────────────────────────────
        self._try_summarize_old_messages(user_id)

        # ── 3. Bangun konteks history untuk LLM ──────────────────────────────
        old_summary, recent_messages = self._get_llm_history_context(user_id)

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

            # ── 7. Simpan ke history ──────────────────────────────────────────
            history = self.get_session_history(user_id)
            history.add_user_message(query)
            history.add_ai_message(answer)

            self.logger.info(f"[Chat] Respon berhasil dibuat untuk session: '{user_id}'")
            return ChatResponse(answer=answer, sources=sources_list)

        except Exception as e:
            self.logger.error(
                f"[Chat] Gagal generate respon untuk session '{user_id}': {e}"
            )
            return ChatResponse(
                answer=(
                    "Maaf, terjadi kesalahan saat memproses pertanyaanmu. "
                    "Coba tanyakan lagi ya! 🙏"
                ),
                sources=[],
            )


# ══════════════════════════════════════════════════════════════════════════════
# TESTING — 3 MODE INTERAKTIF
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    import json

    # ── Konstanta estimasi biaya (Gemini 2.5 Flash) ───────────────────────────
    PRICE_INPUT_PER_1M  = 0.075   # USD per 1 juta token input
    PRICE_OUTPUT_PER_1M = 0.30    # USD per 1 juta token output
    KURS_IDR            = 16_000  # Sesuaikan dengan kurs aktual

    def _estimate_cost(input_chars: int, output_chars: int) -> Dict:
        """Estimasi kasar: 1 token ≈ 4 karakter."""
        inp = input_chars  / 4
        out = output_chars / 4
        usd = (inp / 1_000_000 * PRICE_INPUT_PER_1M) + (out / 1_000_000 * PRICE_OUTPUT_PER_1M)
        return {
            "input_tokens":  int(inp),
            "output_tokens": int(out),
            "cost_usd":      usd,
            "cost_idr":      usd * KURS_IDR,
        }

    # ── Banner & pilih mode ───────────────────────────────────────────────────
    print("╔═══════════════════════════════════════════════════════╗")
    print("║        CHATBOT REDUKA — TUTOR AI UTBK SNBT            ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  Mode 1 : Chat murni       (tanpa log)                ║")
    print("║  Mode 2 : Chat + Log       (INFO, ada pemisah seksi)  ║")
    print("║  Mode 3 : Chat + Log DEBUG + Context + History + Biaya║")
    print("╚═══════════════════════════════════════════════════════╝\n")

    while True:
        _m = input("Pilih mode [1/2/3]: ").strip()
        if _m in ("1", "2", "3"):
            MODE = int(_m)
            break
        print("  ⚠️  Input tidak valid. Masukkan 1, 2, atau 3.\n")

    # ── Konfigurasi logging sesuai mode ──────────────────────────────────────
    # Mode 1: matikan semua log agar output terminal bersih
    # Mode 2: tampilkan INFO ke atas saja (sembunyikan DEBUG yang verbose)
    # Mode 3: tampilkan semua level (DEBUG+)
    if MODE == 1:
        logging.disable(logging.CRITICAL)
    elif MODE == 2:
        logging.disable(logging.DEBUG)
    else:
        logging.disable(logging.NOTSET)

    # ── Inisialisasi service ──────────────────────────────────────────────────
    if MODE != 1:
        print("\n⏳ Menginisialisasi ChatService...\n")

    chat_service = ChatService()

    print()
    USER_ID = input("Masukkan User ID (Enter = 'user001'): ").strip() or "user001"

    print(f"\n{'─'*60}")
    print(f"  ✅  Mode {MODE} aktif  |  User ID: {USER_ID}")
    print(f"  💡  Ketik 'exit' atau 'stop' untuk keluar")
    print(f"{'─'*60}\n")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            query = input("🧑 Kamu: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nKeluar. Semangat belajar! 💪")
            break

        if query.lower() in ("exit", "stop", "quit", "keluar"):
            print("Sampai jumpa! Semangat persiapan UTBK-nya! 🚀")
            break

        if not query:
            continue

        # Snapshot history sebelum query (untuk estimasi token mode 3)
        _hist_before = chat_service.get_session_history(USER_ID).messages[:]

        _t0      = time.perf_counter()
        response = chat_service.generate_response(query=query, user_id=USER_ID)
        _elapsed = time.perf_counter() - _t0

        # ── MODE 1: Output bersih ─────────────────────────────────────────────
        if MODE == 1:
            print(f"\n🤖 AI: {response.answer}\n")

        # ── MODE 2: Query | Log (tercetak natural di atas) | Respon ──────────
        elif MODE == 2:
            # Log sudah tercetak secara natural oleh logger selama generate_response.
            # Pemisah di bawah memisahkan area log dari area respon.
            S = "─" * 65
            print(f"\n{S}")
            print(f"  📥 QUERY    : {query}")
            print(f"  ⏱️  Latency  : {_elapsed:.2f}s")
            print(S)
            print(f"  💬 RESPON AI:")
            print(f"  {response.answer}")
            print(f"{S}\n")

        # ── MODE 3: Debug lengkap ─────────────────────────────────────────────
        else:
            S1 = "═" * 80
            S2 = "─" * 80

            _hist_after = chat_service.get_session_history(USER_ID).messages
            _summary    = chat_service.summary_store.get(USER_ID, "")

            # Estimasi biaya: ambil konteks dari Redis cache milik RetrieveService
            _cached_docs = chat_service.retrieve_service._get_context_history(USER_ID)
            _ctx_chars   = sum(len(d.page_content) for d in _cached_docs)
            _hist_chars  = sum(len(m.content) for m in _hist_before)
            _sys_chars   = len(chat_service.chat_prompt)
            _in_chars    = len(query) + _ctx_chars + _hist_chars + _sys_chars
            _out_chars   = len(response.answer)
            _metrics     = _estimate_cost(_in_chars, _out_chars)

            print(f"\n{S1}")

            # [1] Sources
            print("  🔍 [1] CONTEXT (METADATA SUMBER):")
            if response.sources:
                print(json.dumps(response.sources, indent=4, ensure_ascii=False))
            else:
                print("  (Tidak ada konteks spesifik yang diambil)")

            print(S2)

            # [2] Chat History
            print(f"  🕰️  [2] CHAT HISTORY ({len(_hist_after)} pesan di memori):")
            if not _hist_after:
                print("  (Belum ada history)")
            else:
                for _msg in _hist_after:
                    _role = "🧑 User" if _msg.type == "human" else "🤖 AI  "
                    _txt  = _msg.content
                    if len(_txt) > 220:
                        _txt = _txt[:220] + "…"
                    print(f"    {_role}: {_txt}")

            if _summary:
                print(f"\n  📝 SUMMARY (ringkasan chat lama):")
                print(f"    {_summary}")

            print(S2)

            # [3] Respon AI
            print("  💬 [3] RESPON AI:")
            print(f"  {response.answer}")

            print(S2)

            # [4] Metrik
            print("  📊 [4] METRIK PERFORMA & BIAYA:")
            print(f"    ⏱️  Latency       : {_elapsed:.2f}s")
            print(f"    🪙  Est. Token    : "
                  f"{_metrics['input_tokens']} input | {_metrics['output_tokens']} output")
            print(f"    💵  Est. Biaya    : "
                  f"${_metrics['cost_usd']:.6f}  (~Rp {_metrics['cost_idr']:.2f})")
            print(f"    📚  History total : {len(_hist_after)} pesan")
            print(f"    📝  Summary aktif : "
                  f"{'Ya ✓' if _summary else 'Tidak (belum diperlukan)'}")
            print(f"{S1}\n")
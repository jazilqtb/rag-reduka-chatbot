import re
import yaml
import json
from typing import List, Optional
from redis import Redis


from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


from src.core.config import settings
from src.core.logger import get_logger
from src.services.regex_entities_extractor import RegexEntityExtractor


# ──────────────────────────────────────────────────────────────────────────────
# Konstanta Redis
# ──────────────────────────────────────────────────────────────────────────────
REDIS_ENTITY_TTL  = 1800   # 30 menit
REDIS_CONTEXT_TTL = 1800   # 30 menit
REDIS_ENTITY_KEY  = "entity:{user_id}"
REDIS_CONTEXT_KEY = "context:{user_id}"

class RetrieveService():
    """ 
    Strategi retrieval diurutkan dari biaya paling rendah ke tertinggi:
 
      [0 API call]  Regex + ChromaDB metadata filter (id_soal ditemukan)
      [Embed API]   ChromaDB similarity search        (regex gagal, coba semantik)
      [0 API call]  Redis entity cache + re-fetch     (similarity kosong = follow-up?)
      [Gen API]     LLM entity extractor              (truly last resort)
 
    Redis menyimpan DUA hal per user:
      entity:{user_id}  -> dict {"id_soal": str|None, "subject": str|None}
                           ~50 byte · dipakai sebagai metadata filter ChromaDB
      context:{user_id} -> JSON List[Document]
                           ~3-8 KB · fallback jika semua path lain gagal
    """

    def __init__(self):
        # initialize logger
        self.logger = get_logger("RetrieveService")

        self.r = Redis(
            host=getattr(settings, "REDIS_HOST", "localhost"),
            port=getattr(settings, "REDIS_PORT", 6379),
            decode_responses=True,
        )
        
        self.regex_entity_ext = RegexEntityExtractor()

        # initialize embedding_model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="RETRIEVAL_DOCUMENT",
            google_api_key=settings.GOOGLE_API_KEY
        )

        # Load LLM Entities Extractor
        self.llm_entities_extractor = ChatGoogleGenerativeAI(
            model = settings.GENAI_MODEL,
            api_key = settings.GOOGLE_API_KEY,
            temperature = 0.1
        )

        # Load entities extractor prompt
        try:
            with open(settings.PROMPT_DIR, 'r') as file:
                prompt_data = yaml.safe_load(file)
                self.entities_extractor_prompt = prompt_data.get(
                    "entities_extractor_prompt"
                )
        except Exception as e:
            self.logger.error(f"Gagal load entities extractor prompt: {e}")
            self.entities_extractor_prompt = ""

        # initialize vector_store
        self.vector_store = Chroma(
            collection_name="RAG_REDUKA_DOC_KNOWLEDGE",
            embedding_function=self.embedding_model,
            persist_directory=str(settings.CHROMA_PERSIST_DIR)
        )

 
    # -------------------------------------------------------------------------
    # Entity Extractors
    # -------------------------------------------------------------------------

    def _entities_parser_regex(self, query: str) -> list[dict]:
        """
        Ekstrak entitas menggunakan Regex — 0 API call, ~1ms.
        Mengembalikan list kosong jika pola tidak cocok.
 
        PENTING: gagal ekstrak != tidak ada entitas dalam query.
        Bisa juga frasa-nya tidak tercakup pola regex saat ini.
        Itulah mengapa lapis berikutnya adalah similarity search
        (bukan langsung asumsi follow-up).
        """
        return self.regex_entity_ext.extract_entities(query)

    def _entities_parser_llm(self, query: str) -> list[dict]:
        """
        Ekstrak entitas menggunakan LLM — 1 Generation API call, ~1-3 detik.
 
        Dipakai HANYA jika:
          1. Regex gagal mengekstrak
          2. Similarity search tidak menemukan dokumen relevan
          3. Redis entity cache kosong / expired
 
        Generation API jauh lebih mahal dari embedding API.
        Jika anggaran sangat ketat, method ini bisa dinonaktifkan;
        konsekuensinya chatbot mengembalikan list kosong pada kasus edge.
        """
        if not self.entities_extractor_prompt:
            return []
 
        cleaned = ""
        try:
            messages = [
                ("system", self.entities_extractor_prompt),
                ("human", query),
            ]
            response = self.llm_entities_extractor.invoke(messages)
            cleaned  = response.content.strip()
 
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            return [data] if isinstance(data, dict) else []
 
        except json.JSONDecodeError:
            # Fallback manual jika LLM mengembalikan JSON tidak valid
            pattern = (
                r'\{[^{}]*"id_soal"\s*:\s*"([^"]*)"\s*,\s*"subject"\s*:\s*([^,}]+)\}'
            )
            result = []
            for id_soal, subject_raw in re.findall(pattern, cleaned):
                subject = subject_raw.strip().strip('"')
                if subject in ("null", "None"):
                    subject = None
                result.append({"id_soal": id_soal, "subject": subject})
            return result
 
        except Exception as e:
            self.logger.error(f"[LLM Extractor] Error: {e}")
            return []


    # -------------------------------------------------------------------------
    # ChromaDB Search Methods
    # -------------------------------------------------------------------------
 
    def _similarity_search(
        self, query: str, k: int = 3, subject: Optional[str] = None
    ) -> List[Document]:
        """
        Semantic similarity search ke ChromaDB.
 
        BIAYA: 1 Gemini Embedding API call (query -> vektor).
        Embedding API ~10-50x lebih murah dari Generation API, tapi tetap
        ada biaya per call. Gunakan hanya saat exact search tidak bisa dipakai.
 
        Args:
            query   : teks query siswa
            k       : jumlah dokumen yang dikembalikan
            subject : jika tersedia, tambahkan metadata filter untuk precision
        """
        filter_dict = {"subject": subject} if subject else None
        self.logger.info(
            f"[SimilaritySearch] query='{query[:60]}', k={k}, "
            f"subject={subject or 'none'} | 1 Embedding API call"
        )
        try:
            results = self.vector_store.similarity_search(
                query=query, k=k, filter=filter_dict
            )
            self.logger.info(f"[SimilaritySearch] Ditemukan {len(results)} dokumen.")
            return results
        except Exception as e:
            self.logger.error(f"[SimilaritySearch] Error: {e}")
            return []
    
    def _exact_search(
        self, query: str, id_soal: str = "", subject: str = ""
    ) -> List[Document]:
        """
        Pencarian berdasarkan metadata exact match.
 
        Biaya tiered:
          id_soal + subject -> $and metadata filter   (0 API call)
          id_soal saja      -> metadata filter         (0 API call)
          subject saja      -> similarity + filter     (1 Embedding API call)
          keduanya kosong   -> pure similarity         (1 Embedding API call)
 
        Args:
            query   : teks query siswa (dipakai jika fallback ke similarity)
            id_soal : nomor soal, misal "12"
            subject : mata pelajaran, misal "Penalaran Matematika"
        """
        has_id      = bool(id_soal.strip())
        has_subject = bool(subject.strip())
 
        if has_id and has_subject:
            where   = {
                "$and": [
                    {"id_soal": {"$eq": id_soal.strip()}},
                    {"subject": {"$eq": subject.strip()}},
                ]
            }
            log_tag = f"id_soal={id_soal} & subject={subject}"
 
        elif has_id:
            where   = {"id_soal": {"$eq": id_soal.strip()}}
            log_tag = f"id_soal={id_soal}"
 
        elif has_subject:
            self.logger.info(
                f"[ExactSearch] subject-only -> similarity. subject={subject}"
            )
            return self._similarity_search(query, k=5, subject=subject.strip())
 
        else:
            self.logger.warning(
                "[ExactSearch] id_soal & subject kosong -> pure similarity search."
            )
            return self._similarity_search(query, k=3)
 
        self.logger.info(f"[ExactSearch] metadata filter: {log_tag} | 0 API call")
        try:
            raw   = self.vector_store._collection.get(
                where=where, include=["documents", "metadatas"]
            )
            docs  = raw.get("documents", [])
            metas = raw.get("metadatas", [])
 
            if not docs:
                self.logger.warning(
                    f"[ExactSearch] Tidak ada dokumen untuk filter: {log_tag}"
                )
                return []
 
            self.logger.info(f"[ExactSearch] Ditemukan {len(docs)} dokumen.")
            return [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(docs, metas)
            ]
        except Exception as e:
            self.logger.error(f"[ExactSearch] Error: {e}")
            return []
        
    # -------------------------------------------------------------------------
    # Redis Helpers
    # -------------------------------------------------------------------------
 
    def _redis_entity_key(self, user_id: str) -> str:
        return REDIS_ENTITY_KEY.format(user_id=user_id)
 
    def _redis_context_key(self, user_id: str) -> str:
        return REDIS_CONTEXT_KEY.format(user_id=user_id)
 
    def _save_entity_history(self, user_id: str, entity: dict) -> None:
        """
        Simpan satu dict entitas terakhir ke Redis.
 
        Isi    : {"id_soal": "12", "subject": "Penalaran Matematika"}
        Ukuran : ~50 byte per user
        Tujuan : metadata filter ChromaDB pada follow-up question berikutnya
        Catatan: TIDAK dikirim ke LLM — tidak menguras token.
        """
        if not entity.get("id_soal") and not entity.get("subject"):
            return
        try:
            self.r.setex(
                self._redis_entity_key(user_id),
                REDIS_ENTITY_TTL,
                json.dumps(entity, ensure_ascii=False),
            )
            self.logger.info(f"[Redis] Entity saved: {entity}")
        except Exception as e:
            self.logger.warning(f"[Redis] Gagal simpan entity: {e}")
 
    def _get_entity_history(self, user_id: str) -> Optional[dict]:
        """
        Ambil satu dict entitas terakhir dari Redis.
        Mengembalikan None jika tidak ada atau sudah expired (TTL habis).
        """
        try:
            data = self.r.get(self._redis_entity_key(user_id))
            if data:
                entity = json.loads(data)
                self.logger.info(f"[Redis] Entity loaded: {entity}")
                return entity
        except Exception as e:
            self.logger.warning(f"[Redis] Gagal ambil entity: {e}")
        return None
 
    def _save_context_history(self, user_id: str, docs: List[Document]) -> None:
        """
        Simpan dokumen konteks terakhir ke Redis sebagai JSON.
 
        Isi    : [{page_content: ..., metadata: ...}, ...]
        Ukuran : ~3-8 KB (rata-rata 3 soal x ~1-2 KB/soal)
        Tujuan : safety-net fallback saat semua strategi pencarian kosong
        Catatan: Dokumen ini yang dikirim ke LLM sebagai <konteks> dalam prompt.
                 Entity Redis hanya dipakai untuk query ChromaDB, bukan ke LLM.
        """
        if not docs:
            return
        try:
            payload = json.dumps(
                [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in docs
                ],
                ensure_ascii=False,
            )
            self.r.setex(
                self._redis_context_key(user_id),
                REDIS_CONTEXT_TTL,
                payload,
            )
            self.logger.info(f"[Redis] Context saved: {len(docs)} docs")
        except Exception as e:
            self.logger.warning(f"[Redis] Gagal simpan context: {e}")
 
    def _get_context_history(self, user_id: str) -> List[Document]:
        """
        Ambil dokumen konteks terakhir dari Redis.
        Mengembalikan list kosong jika tidak ada atau expired.
        """
        try:
            data = self.r.get(self._redis_context_key(user_id))
            if data:
                docs = [
                    Document(
                        page_content=d["page_content"], metadata=d["metadata"]
                    )
                    for d in json.loads(data)
                ]
                self.logger.info(
                    f"[Redis] Context loaded: {len(docs)} docs (fallback)"
                )
                return docs
        except Exception as e:
            self.logger.warning(f"[Redis] Gagal ambil context: {e}")
        return []
        
    # -------------------------------------------------------------------------
    # Main Search (Orchestrator)
    # -------------------------------------------------------------------------
 
    def search(self, user_id: str, query: str) -> List[Document]:
        """
        Orkestrasi pencarian konteks — 4 lapis diurutkan dari biaya terendah.
 
        Lapis 1  Regex + ChromaDB metadata filter          [0 API call]
          Regex berhasil + id_soal ditemukan.
          Exact search via _collection.get() — tanpa embedding.
 
        Lapis 2  ChromaDB similarity search                [1 Embedding API]
          Regex gagal (atau dokumen tidak ditemukan di ChromaDB).
          Regex gagal != tidak ada entitas; bisa frasa tidak umum.
          Similarity search mencoba memahami query saat ini secara semantik
          sebelum kita berasumsi bahwa ini adalah follow-up question.
 
        Lapis 3  Redis entity cache + re-fetch             [0 API call]
          Similarity kosong / tidak relevan.
          Baru sekarang kita asumsikan follow-up dari soal sebelumnya.
          Ambil entity dari Redis -> metadata filter ulang ke ChromaDB.
 
        Lapis 4  LLM entity extractor                      [1 Generation API]
          Semua lapis di atas gagal. Truly last resort.
          Generation API jauh lebih mahal dari embedding API.
 
        Setelah pencarian berhasil pada lapis manapun:
          - entity terakhir disimpan ke Redis (untuk follow-up berikutnya)
          - dokumen disimpan ke Redis (sebagai safety-net fallback)
 
        Args:
            user_id : ID unik siswa (dari ChatRequest.user_id)
            query   : teks pertanyaan siswa
 
        Returns:
            List[Document] — dokumen konteks untuk dikirim ke LLM prompt
        """
        contexts: List[Document] = []
 
        # -- Lapis 1: Regex + exact search (0 API call) -----------------------
        entities = self._entities_parser_regex(query.lower())
 
        if entities:
            self.logger.info(
                f"[Search:L1] Regex OK -> {len(entities)} entities: {entities}"
            )
            for entity in entities:
                docs = self._exact_search(
                    query=query,
                    id_soal=entity.get("id_soal") or "",
                    subject=entity.get("subject") or "",
                )
                contexts.extend(docs)
 
            if contexts:
                self._save_entity_history(user_id, entities[-1])
                self._save_context_history(user_id, contexts)
                return contexts
 
            # Regex menemukan entitas tapi dokumen belum ada di ChromaDB
            # (mis. soal belum diingest). Lanjut ke lapis 2.
            self.logger.warning(
                f"[Search:L1] Entitas ditemukan tapi ChromaDB kosong: {entities}"
            )
 
        # -- Lapis 2: Similarity search (1 Embedding API call) ----------------
        # Regex gagal bukan berarti query tidak punya entitas.
        # Coba pahami query secara semantik sebelum asumsi follow-up.
        self.logger.info("[Search:L2] Regex kosong/gagal -> similarity search.")
        contexts = self._similarity_search(query, k=3)
 
        if contexts:
            self._save_context_history(user_id, contexts)
            # Tidak save entity karena kita tidak tahu persis soal mana yang relevan
            return contexts
 
        # -- Lapis 3: Redis entity cache -> re-fetch (0 API call) -------------
        # Similarity juga kosong. Baru sekarang kita asumsikan follow-up:
        # siswa bertanya lanjutan dari soal yang dibahas sebelumnya.
        self.logger.info("[Search:L3] Similarity kosong -> cek Redis entity cache.")
        cached_entity = self._get_entity_history(user_id)
 
        if cached_entity:
            contexts = self._exact_search(
                query=query,
                id_soal=cached_entity.get("id_soal") or "",
                subject=cached_entity.get("subject") or "",
            )
            if contexts:
                self._save_context_history(user_id, contexts)
                return contexts
 
        # -- Lapis 4: LLM entity extractor (1 Generation API call) ------------
        # Truly last resort. Mahal — hanya jika tiga lapis sebelumnya semua gagal.
        self.logger.info("[Search:L4] Last resort -> LLM entity extractor.")
        llm_entities = self._entities_parser_llm(query)
 
        if llm_entities:
            for entity in llm_entities:
                docs = self._exact_search(
                    query=query,
                    id_soal=entity.get("id_soal") or "",
                    subject=entity.get("subject") or "",
                )
                contexts.extend(docs)
 
            if contexts:
                self._save_entity_history(user_id, llm_entities[-1])
                self._save_context_history(user_id, contexts)
                return contexts
 
        # -- Fallback final: Redis context cache ------------------------------
        # Semua path gagal. Kembalikan dokumen terakhir yang pernah disimpan
        # agar LLM tidak menjawab tanpa konteks apapun.
        self.logger.warning(
            "[Search] Semua lapis gagal -> load cached context dari Redis."
        )
        return self._get_context_history(user_id)

# =============================================================================
# Test Runner  (python rag_service.py [regex|full])
# =============================================================================
if __name__ == "__main__":
    import time
    import statistics
    import argparse
    from dataclasses import dataclass
 
    PASS   = "\033[92mPASS\033[0m"
    FAIL   = "\033[91mFAIL\033[0m"
    WARN   = "\033[93mWARN\033[0m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
 
    @dataclass
    class CaseResult:
        index: int
        query_short: str
        ok: bool
        latency_ms: float
        result: list[dict]
        expected: list[dict]
        error: str = ""
 
    retrieve = RetrieveService()
 
    test_cases: list[tuple[str, list[dict]]] = [
        (
            "Di soal nomor 12 penalaran umum, kenapa jawabannya B, bukan C?",
            [{"id_soal": "12", "subject": "Penalaran Umum"}],
        ),
        (
            "soal nomor 12 bilang pakai konsep limit, bisa dijelaskan ulang?",
            [{"id_soal": "12", "subject": None}],
        ),
        (
            "Yang nomor 12 penalaran umum itu mirip sama nomor 8 di penalaran matematika?",
            [{"id_soal": "12", "subject": "Penalaran Umum"},
             {"id_soal": "8",  "subject": "Penalaran Matematika"}],
        ),
        (
            "Yang ada grafik naik turun di penalaran matematika itu kenapa jawabannya D?",
            [{"id_soal": None, "subject": "Penalaran Matematika"}],
        ),
        (
            "Di soal no 1 literasi bahasa inggris tryout 1, kenapa jawabannya A bukan D?",
            [{"id_soal": "1", "subject": "Literasi Bahasa Inggris"}],
        ),
        (
            "nomor lima literasi bahasa indonesia itu tentang apa?",
            [{"id_soal": "5", "subject": "Literasi Bahasa Indonesia"}],
        ),
        (
            "kak jelaskan nomor dua belas penalaran umum dong",
            [{"id_soal": "12", "subject": "Penalaran Umum"}],
        ),
        (
            "soal ke-empat literasi bahasa inggris maksudnya apa?",
            [{"id_soal": "4", "subject": "Literasi Bahasa Inggris"}],
        ),
        (
            "Jelaskan soal kesebelas penalaran umum kak.",
            [{"id_soal": "11", "subject": "Penalaran Umum"}],
        ),
        (
            "kenapa soal nomor 12 dan tiga belas pada penalaran matematika dan penalaran umum jawabannya begitu?",
            [{"id_soal": "12", "subject": "Penalaran Matematika"},
             {"id_soal": "13", "subject": "Penalaran Umum"}],
        ),
    ]
 
    def _latency_bar(ms: float, max_ms: float, width: int = 20) -> str:
        filled = min(int((ms / max_ms) * width) if max_ms > 0 else 0, width)
        color  = GREEN if ms < max_ms * 0.33 else (YELLOW if ms < max_ms * 0.66 else RED)
        return f"{color}{'█' * filled}{'░' * (width - filled)}{RESET}"
 
    def _run_parser(name, fn, cases, rate_limit_sleep=0.0):
        results = []
        print(f"\n{BOLD}{'='*76}{RESET}")
        print(f"{BOLD}  [{name}] — Entity Extraction Test{RESET}")
        print(f"{BOLD}{'='*76}{RESET}")
        if rate_limit_sleep:
            print(f"  {CYAN}⚠  API call aktif — delay {rate_limit_sleep}s/query.{RESET}")
 
        for i, (query, expected) in enumerate(cases, 1):
            error_msg = ""
            t0 = time.perf_counter()
            try:
                result = fn(query)
            except Exception as e:
                result, error_msg = [], str(e)
            latency_ms = (time.perf_counter() - t0) * 1000
            ok = result == expected
            cr = CaseResult(
                index=i,
                query_short=(query[:52] + "…") if len(query) > 52 else query,
                ok=ok, latency_ms=latency_ms, result=result,
                expected=expected, error=error_msg,
            )
            results.append(cr)
            status = PASS if ok else (WARN if error_msg else FAIL)
            print(f"\n  [{status}] Case {i:02d} — {latency_ms:>7.2f} ms")
            print(f"    query   : {cr.query_short!r}")
            print(f"    got     : {json.dumps(result, ensure_ascii=False)}")
            if not ok:
                print(f"    expected: {json.dumps(expected, ensure_ascii=False)}")
            if error_msg:
                print(f"    error   : {RED}{error_msg}{RESET}")
            if rate_limit_sleep:
                time.sleep(rate_limit_sleep)
 
        passed = sum(1 for r in results if r.ok)
        print(f"\n  {BOLD}{name} Result: {passed}/{len(cases)} passed{RESET}")
        return results
 
    def _print_summary(r_res, l_res):
        all_lat = [r.latency_ms for r in r_res + l_res]
        max_ms  = max(all_lat) if all_lat else 1.0
        n       = len(r_res)
 
        print(f"\n{BOLD}{'='*76}{RESET}")
        print(f"{BOLD}  LATENCY MATRIX{RESET}")
        print(f"{BOLD}{'='*76}{RESET}\n")
        print(f"{BOLD}  {'#':>3}  {'Query':<54}  {'Regex':>9}  {'LLM':>9}  {'Delta':>12}{RESET}")
        print(f"  {'-'*3}  {'-'*54}  {'-'*9}  {'-'*9}  {'-'*12}")
 
        for rx, lx in zip(r_res, l_res):
            d = lx.latency_ms - rx.latency_ms
            print(
                f"  {rx.index:>3}  {rx.query_short:<54}  "
                f"{rx.latency_ms:>7,.1f}ms "
                f"{GREEN if rx.ok else RED}{'✓' if rx.ok else '✗'}{RESET}  "
                f"{lx.latency_ms:>7,.1f}ms "
                f"{GREEN if lx.ok else RED}{'✓' if lx.ok else '✗'}{RESET}  "
                f"{'+' if d >= 0 else ''}{d:,.1f}ms"
            )
 
        def _s(vals):
            return {"mean": statistics.mean(vals), "total": sum(vals)}
 
        rs = _s([r.latency_ms for r in r_res])
        ls = _s([r.latency_ms for r in l_res])
        sp = ls["mean"] / rs["mean"] if rs["mean"] > 0 else float("inf")
 
        r_pass = sum(1 for r in r_res if r.ok)
        l_pass = sum(1 for r in l_res if r.ok)
 
        print(f"\n  {'Metric':<16} {'Regex':>12} {'LLM':>12}")
        print(f"  {'-'*42}")
        for label, rv, lv in [
            ("Accuracy",   f"{r_pass/n*100:.1f}%",  f"{l_pass/n*100:.1f}%"),
            ("Mean (ms)",  f"{rs['mean']:,.1f}",     f"{ls['mean']:,.1f}"),
            ("Total (ms)", f"{rs['total']:,.1f}",    f"{ls['total']:,.1f}"),
        ]:
            print(f"  {label:<16} {rv:>12} {lv:>12}")
        print(f"\n  {BOLD}LLM rata-rata {sp:,.1f}x lebih lambat.{RESET}")
        print(f"{BOLD}{'='*76}{RESET}\n")
 
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["regex", "llm", "full"], nargs="?", default="regex")
    args = ap.parse_args()
 
    r_results = _run_parser("REGEX PARSER", retrieve._entities_parser_regex, test_cases)
 
    if args.mode == "llm":
        l_results = _run_parser(
            "LLM PARSER", retrieve._entities_parser_llm, test_cases, rate_limit_sleep=0.5
        )
        _print_summary(r_results, l_results)
    if args.mode == "full":
        print("="*120)
        user_id = str(input("Enter your name: "))
        question = ""
        while question!="stop":
            question = str(input("Enter your question: "))
            print('='*100)
            context_results = retrieve.search(user_id=user_id, query=question)
            print('-'*100)
            print("CONTEXTS:")
            for i in context_results:
                print(f"  - Metadata: {i.metadata}")
            print('='*100)
        else:
            print("="*120)
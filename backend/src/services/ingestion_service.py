"""
Ingestion Service — Orchestrator pipeline PDF → ChromaDB.

Stage 5 changes:
  1. PDF parsing logic dipindahkan ke pdf_parser.py (PDFParser class)
  2. INCREMENTAL INGESTION: tidak lagi reset ChromaDB total.
     Sebelumnya: shutil.rmtree(db_dir) di __init__ → semua data hilang
     Sekarang  : delete chunk lama per source filename sebelum insert baru

API baru: run(filenames=None)
  - filenames=None      → backward compat: process semua soal_*.pdf di dir
  - filenames=[list]    → hanya process file yang disebutkan (incremental)

Implikasi positif:
  - Saat upload + ingest file BARU, file LAMA tidak hilang
  - Bisa re-ingest 1 file tanpa mempengaruhi lain
  - Konsisten dengan endpoint DELETE yang juga hapus per source
"""

import os
import time
import yaml
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.core.config import settings
from src.core.logger import get_logger
from src.services.pdf_parser import PDFParser


# ── Konstanta ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "UTBK_TUTOR_KNOWLEDGE"  # Stage 1 rebrand
BATCH_SIZE        = 20                      # Chunks per ChromaDB insert batch
MAX_RETRIES       = 3                       # Retry per batch on rate limit


class IngestionService:
    """
    Orchestrator ingestion: parse PDF → structure to Docs → save to ChromaDB.

    Delegasi:
      PDF parsing & captioning  → PDFParser
      Embedding & ChromaDB save → langchain_chroma + Google embeddings
    """

    def __init__(self):
        self.logger = get_logger("IngestionService")

        self.pdf_dir   = settings.DATA_DIR / "raw_docs"
        self.db_dir    = settings.CHROMA_PERSIST_DIR
        self.debug_dir = settings.DATA_DIR / "debug"

        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(self.db_dir,    exist_ok=True)

        # ── LLM untuk image caption + JSON structuring ────────────────────────
        # temperature rendah agar ekstraksi tidak halusinasi.
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
        )

        # ── Embedding model untuk ChromaDB ────────────────────────────────────
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_document",
            google_api_key=settings.GOOGLE_API_KEY,
        )

        # ── ChromaDB (TIDAK lagi reset di Stage 5) ────────────────────────────
        self.vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=str(self.db_dir),
        )

        # ── Prompts dari config/prompts.yaml ──────────────────────────────────
        img_caption_prompt      = "Jelaskan detail gambar/grafik ini."
        json_structuring_prompt = "Format teks ke JSON array."
        try:
            with open(settings.PROMPT_DIR, "r", encoding="utf-8") as f:
                prompts = yaml.safe_load(f)
                img_caption_prompt      = prompts.get("image_captioning_prompt",  img_caption_prompt)
                json_structuring_prompt = prompts.get("json_structuring_prompt",  json_structuring_prompt)
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}. Pakai default.")

        # ── PDF Parser (delegate) ─────────────────────────────────────────────
        self.parser = PDFParser(
            llm                     = self.llm,
            img_caption_prompt      = img_caption_prompt,
            json_structuring_prompt = json_structuring_prompt,
            debug_dir               = self.debug_dir,
        )

        self.logger.info("IngestionService initialized (incremental mode).")

    # ══════════════════════════════════════════════════════════════════════════
    # CHROMA OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════

    def _delete_existing_chunks(self, filename: str) -> int:
        """
        Hapus semua chunk di ChromaDB yang punya metadata source == filename.

        Dipakai sebelum re-ingest file yang sama supaya tidak ada duplikasi.

        Returns:
            Jumlah chunk yang dihapus (0 jika tidak ada).
        """
        try:
            result  = self.vector_store._collection.get(
                where   = {"source": {"$eq": filename}},
                include = [],
            )
            doc_ids = result.get("ids", [])
            if doc_ids:
                self.vector_store._collection.delete(ids=doc_ids)
                self.logger.info(
                    f"[Chroma] {len(doc_ids)} chunk lama untuk '{filename}' dihapus."
                )
            return len(doc_ids)
        except Exception as e:
            self.logger.warning(f"[Chroma] Gagal hapus chunk lama '{filename}': {e}")
            return 0

    def save_to_chroma(self, chunks: List[Document]) -> int:
        """
        Simpan chunks ke ChromaDB dengan batching + retry on rate limit.

        Returns:
            Jumlah chunk yang berhasil disimpan.
        """
        if not chunks:
            self.logger.warning("Chunks kosong, tidak ada yang disimpan.")
            return 0

        total      = len(chunks)
        successful = 0
        self.logger.info(
            f"Menyimpan {total} chunks. Strategy: batch {BATCH_SIZE} dengan retry."
        )

        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            self.logger.info(f"Processing batch {i} → {i + len(batch)}...")

            for attempt in range(MAX_RETRIES):
                try:
                    self.vector_store.add_documents(documents=batch)
                    successful += len(batch)
                    time.sleep(2)  # rate limit avoidance
                    break
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg and attempt < MAX_RETRIES - 1:
                        wait = (attempt + 1) * 20
                        self.logger.warning(
                            f"Hit rate limit (429). Retry in {wait}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(wait)
                    elif attempt < MAX_RETRIES - 1:
                        wait = (attempt + 1) * 5
                        self.logger.warning(
                            f"Error: {e}. Retry in {wait}s "
                            f"(attempt {attempt + 1}/{MAX_RETRIES})"
                        )
                        time.sleep(wait)
                    else:
                        self.logger.error(f"Critical error on batch {i}: {e}")
                        raise

        self.logger.info(f"Berhasil simpan {successful}/{total} chunks ke ChromaDB.")
        return successful

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════════════════════

    def _process_single_file(self, soal_filename: str) -> List[Document]:
        """
        Process satu file soal: parse + structure + return docs.
        Tidak save ke ChromaDB di sini — itu dilakukan secara batch di run().
        """
        file_path_soal = self.pdf_dir / soal_filename
        if not file_path_soal.exists():
            self.logger.warning(f"File tidak ditemukan: {file_path_soal}")
            return []

        # Cari pasangan jawaban (soal_xxx.pdf → jawaban_xxx.pdf)
        filename_jawaban    = soal_filename.replace("soal_", "jawaban_", 1)
        file_path_jawaban   = self.pdf_dir / filename_jawaban

        # 1. Ekstrak kunci jawaban (regex, 0 LLM call)
        answer_keys = self.parser.parse_answer_key(str(file_path_jawaban))

        # 2. Parse PDF + caption gambar (multimodal, beberapa LLM call)
        raw_text = self.parser.parse_pdf_multimodal(str(file_path_soal))

        # 3. LLM structuring ke JSON + suntik kunci jawaban (1 LLM call)
        return self.parser.structure_text_to_documents(raw_text, soal_filename, answer_keys)

    def run(self, filenames: Optional[List[str]] = None) -> int:
        """
        Jalankan pipeline ingestion.

        Args:
            filenames: List filename PDF (relatif ke raw_docs/) yang ingin
                       diingest. Boleh campuran soal_*.pdf dan jawaban_*.pdf;
                       hanya yang prefix 'soal_' yang diproses sebagai unit
                       utama (jawaban auto-found via name pattern).
                       Jika None: process semua soal_*.pdf di raw_docs (legacy).

        Returns:
            Total chunk yang berhasil disimpan ke ChromaDB.
        """
        self.logger.info("=== START INGESTION ===")

        if not self.pdf_dir.exists():
            self.logger.error(f"Folder tidak ditemukan: {self.pdf_dir}")
            return 0

        # ── Tentukan file 'soal' yang akan diproses ──────────────────────────
        if filenames is None:
            target_soal = sorted(
                f.name for f in self.pdf_dir.iterdir()
                if f.name.startswith("soal_") and f.name.endswith(".pdf")
            )
            self.logger.info(f"Mode: full scan. {len(target_soal)} file soal ditemukan.")
        else:
            # Filter: hanya yang prefix soal_ + ada di disk
            target_soal = sorted({
                f for f in filenames
                if f.startswith("soal_")
                and f.endswith(".pdf")
                and (self.pdf_dir / f).exists()
            })
            self.logger.info(f"Mode: incremental. Process {len(target_soal)} file soal.")

        if not target_soal:
            self.logger.warning("Tidak ada file soal yang valid untuk diproses.")
            return 0

        # ── Process per-file, delete-then-insert untuk incremental ──────────
        total_chunks_saved = 0
        for soal_filename in target_soal:
            self.logger.info(f"--- Processing {soal_filename} ---")

            # Hapus chunk lama untuk file ini (kalau ada) supaya tidak duplikat
            self._delete_existing_chunks(soal_filename)

            # Parse + structure
            docs = self._process_single_file(soal_filename)
            if not docs:
                self.logger.warning(f"Tidak ada dokumen valid dari {soal_filename}, skip.")
                continue

            # Insert ke ChromaDB
            saved              = self.save_to_chroma(docs)
            total_chunks_saved += saved

        self.logger.info(
            f"=== FINISH INGESTION === Total {total_chunks_saved} chunks saved."
        )
        return total_chunks_saved


if __name__ == "__main__":
    # CLI mode untuk testing manual: process semua soal di raw_docs
    ingestor = IngestionService()
    ingestor.run()
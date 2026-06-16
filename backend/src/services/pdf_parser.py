"""
PDF Parser — Ekstraksi teks dan kunci jawaban dari PDF.

Diekstrak dari IngestionService di Stage 5 supaya:
  - Logic parsing terpisah dari orchestration & ChromaDB
  - Bisa di-test dengan PDF dummy tanpa butuh LLM / ChromaDB
  - Mudah diganti library PDF (pymupdf -> pypdf, dll) di masa depan

Tiga tahap parsing:
  1. parse_answer_key(filepath)         → dict {nomor_soal: kunci}
  2. parse_pdf_multimodal(filepath)     → text + image captions
  3. structure_text_to_documents(...)   → LangChain Document list
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.core.logger import get_logger


class PDFParser:
    """
    Parsing PDF + image captioning + LLM-based JSON structuring.

    Dependency:
      llm                    : LangChain LLM (ChatGoogleGenerativeAI atau setara)
      img_caption_prompt     : prompt untuk caption gambar
      json_structuring_prompt: prompt untuk strukturisasi JSON
      debug_dir              : path untuk simpan output debug JSON (Optional)
    """

    def __init__(
        self,
        llm,
        img_caption_prompt: str,
        json_structuring_prompt: str,
        debug_dir: Optional[Path] = None,
    ):
        self.llm                     = llm
        self.img_caption_prompt      = img_caption_prompt
        self.json_structuring_prompt = json_structuring_prompt
        self.debug_dir               = debug_dir
        self.logger                  = get_logger("PDFParser")

        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ANSWER KEY (Regex-based, no LLM)
    # ══════════════════════════════════════════════════════════════════════════

    def parse_answer_key(self, filepath: str) -> Dict[str, str]:
        """
        Ekstrak kunci jawaban (PG maupun Essay) menggunakan Regex.

        Pattern r'(\\d+)\\.\\s*(.*)' menangkap "<nomor>. <jawaban>":
          - (\\d+)  : nomor urut (capturing group)
          - \\.     : titik literal
          - \\s*    : whitespace (nol atau lebih)
          - (.*)    : sisa baris sebagai jawaban (capturing group)

        Returns:
            Dict {nomor_soal: jawaban}. Kosong jika file tidak ada.
        """
        if not os.path.exists(filepath):
            self.logger.warning(f"File kunci jawaban tidak ditemukan: {filepath}")
            return {}

        doc  = pymupdf.open(filepath)
        text = ""
        try:
            for page in doc:
                text += page.get_text()
        finally:
            doc.close()

        matches     = re.findall(r"(\d+)\.\s*(.*)", text)
        answer_dict = {num: ans.strip() for num, ans in matches}

        self.logger.info(
            f"Berhasil mengekstrak {len(answer_dict)} kunci jawaban "
            f"dari {os.path.basename(filepath)}"
        )
        return answer_dict

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MULTIMODAL PDF PARSING (text + image captioning)
    # ══════════════════════════════════════════════════════════════════════════

    def generate_image_caption(self, image_bytes: bytes) -> str:
        """
        Kirim byte gambar ke LLM (Gemini Flash) untuk mendapatkan deskripsi teks.
        Cost: ~1 LLM call per gambar (multimodal request).
        """
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        message = HumanMessage(
            content=[
                {"type": "text", "text": self.img_caption_prompt},
                {
                    "type":      "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                },
            ]
        )

        try:
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            self.logger.error(f"Gagal generate image caption: {e}")
            return "Gambar tidak dapat dideskripsikan."

    def parse_pdf_multimodal(self, file_path: str) -> str:
        """
        Ekstrak teks + gambar dari PDF, lalu inject deskripsi gambar.

        PDF diharapkan punya placeholder "[GAMBAR]" yang akan diganti dengan
        caption hasil LLM dalam urutan yang sama.

        Returns:
            String berisi teks PDF + caption (siap untuk LLM structuring).
        """
        self.logger.info(f"Membaca PDF: {os.path.basename(file_path)}")

        doc            = pymupdf.open(file_path)
        full_text      = ""
        captions_queue = []

        try:
            for page_num, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if block["type"] == 0:  # block = teks
                        for line in block["lines"]:
                            for span in line["spans"]:
                                full_text += span["text"] + " "
                        full_text += "\n"

                    elif block["type"] == 1:  # block = gambar
                        self.logger.info(
                            f"Gambar terdeteksi di halaman {page_num + 1}. "
                            "Melakukan captioning..."
                        )
                        image_bytes = block["image"]
                        caption     = self.generate_image_caption(image_bytes)
                        captions_queue.append(caption)
        finally:
            doc.close()

        # Ganti tag [GAMBAR] dengan caption AI, urut sesuai antrian
        for caption in captions_queue:
            inject_text = f"\n\n--- KONTEKS GAMBAR: {caption} ---\n\n"
            full_text   = full_text.replace("[GAMBAR]", inject_text, 1)

        return full_text

    # ══════════════════════════════════════════════════════════════════════════
    # 3. JSON STRUCTURING (text -> List[Document])
    # ══════════════════════════════════════════════════════════════════════════

    def structure_text_to_documents(
        self,
        raw_text:    str,
        filename:    str,
        answer_keys: Dict[str, str],
    ) -> List[Document]:
        """
        Strukturisasi teks PDF menjadi list of LangChain Document.

        Pipeline:
          1. LLM convert raw_text -> JSON array of soal
          2. Suntikkan kunci jawaban (matching by nomor_soal)
          3. Bangun page_content + metadata untuk tiap soal
          4. (Opsional) Simpan debug JSON ke disk

        Returns:
            List[Document] siap di-add ke ChromaDB. Empty list jika LLM gagal.
        """
        self.logger.info("Menyusun ulang teks menjadi JSON berstruktur...")

        prompt   = f"{self.json_structuring_prompt}\n\nTeks Ujian Mentah:\n{raw_text}"
        response = None

        try:
            response     = self.llm.invoke(prompt)
            cleaned_json = response.content.replace("```json", "").replace("```", "").strip()
            soal_list    = json.loads(cleaned_json)
        except Exception as e:
            content_preview = response.content[:200] if response else "No response"
            self.logger.error(
                f"Gagal melakukan structuring JSON: {e}\n"
                f"Response LLM preview: {content_preview}"
            )
            return []

        # Validasi jumlah soal vs jumlah kunci
        if len(soal_list) != len(answer_keys):
            self.logger.warning(
                f"⚠️  Jumlah soal ({len(soal_list)}) BEDA dengan jumlah kunci "
                f"jawaban ({len(answer_keys)}) pada file {filename}!"
            )

        documents: List[Document] = []
        for i, soal in enumerate(soal_list):
            nomor_soal = str(i + 1)

            # Suntik kunci jawaban
            if nomor_soal in answer_keys:
                soal["kunci_jawaban"] = answer_keys[nomor_soal]
            else:
                self.logger.warning(
                    f"⚠️  Kunci jawaban untuk soal nomor {nomor_soal} "
                    f"tidak ditemukan di {filename}"
                )

            # Susun content gabungan (untuk embedding)
            content = (
                f"Nomor Soal: {soal.get('id_soal', '')}\n"
                f"Subject: {soal.get('subject', '')}\n"
                f"Topik: {soal.get('topik', '')}\n"
                f"Konteks Bacaan: {soal.get('konteks_bacaan', 'Tidak ada bacaan khusus')}\n"
                f"Soal: {soal.get('pertanyaan', '')}\n"
                f"Pilihan: {soal.get('opsi', 'Tidak ada opsi (Soal Isian/Essay)')}\n"
                f"Konteks Gambar: {soal.get('konteks_gambar', 'Tidak ada gambar')}\n"
                f"Kunci Jawaban: {soal.get('kunci_jawaban', 'Tidak ada kunci jawaban')}\n"
                f"Pembahasan: {soal.get('pembahasan', 'Belum ada pembahasan')}"
            )

            meta = {
                "source":      filename,
                "subject":     soal.get("subject",     "Umum"),
                "jenis_ujian": soal.get("jenis_ujian", "Tryout"),
                "id_soal":     soal.get("id_soal", ""),
            }

            documents.append(Document(page_content=content, metadata=meta))

        # Simpan debug JSON kalau folder dikonfigurasi
        if self.debug_dir is not None and soal_list:
            debug_path = self.debug_dir / f"debug_{filename.replace('.pdf', '')}.json"
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump(soal_list, f, indent=4, ensure_ascii=False)
                self.logger.info(f"Debug JSON disimpan di: {debug_path}")
            except Exception as e:
                self.logger.warning(f"Gagal simpan debug JSON: {e}")

        return documents
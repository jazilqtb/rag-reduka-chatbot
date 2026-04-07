import os
import time
import yaml
import json
import base64
import re
import shutil
import pymupdf

from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma

from src.core.config import settings 
from src.core.logger import get_logger

class IngestionService():
    def __init__(self):
        self.logger = get_logger("IngestionService")

        self.pdf_dir = settings.DATA_DIR / "raw_docs"
        self.db_dir = settings.CHROMA_PERSIST_DIR
        self.debug_dir = settings.DATA_DIR / "debug"

        # Buat folder debug jika belum ada
        os.makedirs(self.debug_dir, exist_ok=True)

        # Gunakan Gemini 1.5 Flash karena murah, cepat, dan pintar baca gambar
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.1 # Suhu rendah agar ekstraksi tidak halusinasi
        )

        emb_model = getattr(settings, 'EMBEDDING_MODEL', "models/embedding-001")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=emb_model,
            task_type="retrieval_document",
            google_api_key=settings.GOOGLE_API_KEY
        )

        self.vector_store = Chroma(
            collection_name="RAG_REDUKA_DOC_KNOWLEDGE",
            embedding_function=self.embeddings,
            persist_directory=str(self.db_dir)   
        )

        # Prompt configuration
        try:
            with open(settings.PROMPT_DIR, 'r') as file:
                prompts = yaml.safe_load(file)
                self.img_caption_prompt = prompts.get("image_captioning_prompt", "Jelaskan gambar ini dengan detail.")
                self.json_structuring_prompt = prompts.get("json_structuring_prompt", "Ubah ke JSON.")
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}")
            self.img_caption_prompt = "Jelaskan detail gambar/grafik ini."
            self.json_structuring_prompt = "Format teks ke JSON array."
    
    def generate_image_caption(self, image_bytes: bytes) -> str:
        """Mengirim byte gambar ke Gemini Flash untuk mendapatkan deskripsi teks"""
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": self.img_caption_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
            ]
        )
        
        try:
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            self.logger.error(f"Gagal generate image caption: {e}")
            return "Gambar tidak dapat dideskripsikan."
        
    def parse_answer_key(self, filepath: str) -> dict:
        """Mengekstrak kunci jawaban (Pilihan Ganda maupun Essay) menggunakan Regex"""
        # Check file jawaban
        if not os.path.exists(filepath):
            self.logger.warning(f"File kunci jawaban tidak ditemukan: {filepath}")
            return {}
        
        # Open and ekstrak all text in every page
        doc = pymupdf.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        """
            Pernyataan r'(\d+)\.\s*(.*)' digunakan untuk ekstrak daftar bernomor dari teks

            (\d)
                - \d = digit (0-9)
                - + = satu atau lebih
                - () = capturing group
                Artinya: Mengambl angka (nomor urutan)
                Contoh: 1, 23, 456
            
            \.
                - Titik (.) adalah karakter spesial di regex (artinya “apa saja”),
                jadi harus di-escape dengan \.
                Artinya: mencocokkan titik literal
                Contoh: dari 1. → ini bagian titiknya

            \s artinya whitespace
                - * = nol atau lebih
        """
        matches = re.findall(r'(\d+)\.\s*(.*)', text) # matches berisi list [('nomor_soal', 'kunci jawban')]
        
        # Bersihkan spasi kosong (strip) pada hasil tangkapan
        answer_dict = {num: ans.strip() for num, ans in matches} # dijadikan dict
        
        self.logger.info(f"Berhasil mengekstrak {len(answer_dict)} kunci jawaban dari {os.path.basename(filepath)}")
        return answer_dict
        
    def parse_pdf_multimodal(self, file_path: str) -> str:
        """Mengekstrak teks dan gambar dari PDF, lalu menggabungkannya"""
        self.logger.info(f"Membaca PDF: {os.path.basename(file_path)}")

        # open pdf file
        doc = pymupdf.open(file_path)
        
        full_text = "" # isi full text
        captions_queue = [] # Antrean untuk menyimpan deskripsi gambar

        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"] # "dict" means get text and images
            
            for block in blocks:
                if block['type'] == 0:  # Jika block nya adalah teks
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text += span["text"] + " "
                    full_text += "\n"
                    
                elif block['type'] == 1:  # Jika blocknya adalah gambar
                    self.logger.info(f"Gambar terdeteksi di halaman {page_num + 1}. Melakukan captioning...")
                    image_bytes = block["image"]
                    
                    # 1. Panggil AI untuk mendeskripsikan gambar
                    caption = self.generate_image_caption(image_bytes)
                    captions_queue.append(caption)

        doc.close()

        # 2. Gantikan tag [GAMBAR] yang diketik admin dengan hasil deskripsi AI
        for caption in captions_queue:
            # Gunakan replace dengan argumen count=1 agar mengganti tag secara berurutan
            inject_text = f"\n\n--- KONTEKS GAMBAR: {caption} ---\n\n"
            full_text = full_text.replace("[GAMBAR]", inject_text, 1)

        return full_text # type(str) berisi full text pdf + image caption (raw text)
    
    def structure_text_to_documents(self, raw_text: str, filename: str, answer_keys: dict) -> List[Document]:
        """Mengubah teks utuh menjadi JSON, menyuntikkan kunci jawaban, dan menyusun Vector"""
        self.logger.info("Menyusun ulang teks menjadi JSON berstruktur...")
        
        prompt = f"{self.json_structuring_prompt}\n\nTeks Ujian Mentah:\n{raw_text}"
        
        try:
            response = self.llm.invoke(prompt)
            cleaned_json = response.content.replace('```json', '').replace('```', '').strip()
            
            soal_list = json.loads(cleaned_json) 
            documents = []

            # VALIDASI: Pencocokan Jumlah Soal vs Jumlah Kunci Jawaban
            if len(soal_list) != len(answer_keys):
                self.logger.warning(f"⚠️ PERINGATAN: Jumlah soal ({len(soal_list)}) BEDA dengan jumlah kunci jawaban ({len(answer_keys)}) pada file {filename}!")
            
            # 1. Suntikkan kunci jawaban & Buat dokumen Vector
            for i, soal in enumerate(soal_list):
                nomor_soal = str(i + 1)
                
                # Cek dan masukkan kunci jawaban
                if nomor_soal in answer_keys:
                    soal['kunci_jawaban'] = answer_keys[nomor_soal]
                else:
                    self.logger.warning(f"⚠️ Kunci jawaban untuk soal nomor {nomor_soal} tidak ditemukan!")
                
                # Susun teks konten untuk di-embedding (digabung agar bisa dicari vector DB)
                content = (
                    f"Nomor Soal: {soal.get('id_soal', '')}\n"
                    f"Subject: {soal.get('subject', '')}\n"
                    f"Topik: {soal.get('topik', '')}\n"
                    f"Konteks Bacaan: {soal.get('konteks_bacaan', 'Tidak ada bacaan khusus')}\n" # Tambahan atribut bacaan
                    f"Soal: {soal.get('pertanyaan', '')}\n"
                    f"Pilihan: {soal.get('opsi', 'Tidak ada opsi (Soal Isian/Essay)')}\n"
                    f"Konteks Gambar: {soal.get('konteks_gambar', 'Tidak ada gambar')}\n"
                    f"Kunci Jawaban: {soal.get('kunci_jawaban', 'Tidak ada kunci jawaban')}\n"
                    f"Pembahasan: {soal.get('pembahasan', 'Belum ada pembahasan')}"
                )
                
                meta = {
                    "source": filename,
                    "subject": soal.get("subject", "Umum"),
                    "jenis_ujian": soal.get("jenis_ujian", "Tryout"),
                    "id_soal": soal.get('id_soal', ''),
                    "subject": soal.get('subject', '')
                }
                
                doc = Document(page_content=content, metadata=meta)
                documents.append(doc)

                """
                document = Document(
                    page_content="Hello, world!", metadata={"source": "https://example.com"}
                )
                print(f"type(document): {type(document)} = {document}")

                output:
                type(document): <class 'langchain_core.documents.base.Document'> = page_content='Hello, world!' metadata={'source': 'https://example.com'}
                """

            # 2. Simpan hasil JSON ke file lokal untuk proses review / debugging Anda
            debug_file_path = os.path.join(self.debug_dir, f"debug_{filename.replace('.pdf', '')}.json")
            with open(debug_file_path, "w", encoding="utf-8") as f:
                json.dump(soal_list, f, indent=4, ensure_ascii=False)
            self.logger.info(f"File evaluasi JSON disimpan di: {debug_file_path}")
                
            return documents # documents = [('page_content':(teksss), 'metadata':{source:tekss, subject:bla bla})]
        except Exception as e:
            self.logger.error(f"Gagal melakukan structuring JSON: {e}\nResponse LLM: {response.content if 'response' in locals() else 'No Response'}")
            return []
        
    def save_to_chroma(self, chunks: List[Document]):
        """Menyimpan chunks (documents) ke ChromaDB dengan sistem Batch dan Retry Logic"""
        if not chunks:
            self.logger.warning("Chunks is empty.")
            return
        
        # Reset DB lama agar vector tidak duplikat saat dijalankan ulang
        if os.path.exists(self.db_dir):
            try:
                shutil.rmtree(self.db_dir)
                self.logger.info("The old database has been successfully deleted (Reset).")
            except Exception as e:
                self.logger.error(f"Failed to delete the old DB: {str(e)}")
        
        try:
            BATCH_SIZE = 20
            total_chunks = len(chunks) # setara jumlah soal
            
            self.logger.info(f"Total chunks: {total_chunks}. Strategy: Batch {BATCH_SIZE} with Retry Logic.")

            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE] # batch berisi teks soal dari 1 - akhir dalam satu file
                
                self.logger.info(f"Processing batch {i} to {i + len(batch)}...")
                max_retries = 3

                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(documents=batch)
                        time.sleep(2) 
                        break
                    except Exception as e:
                        error_msg = str(e)
                        if "429" in error_msg:
                            wait_time = (attempt + 1) * 20
                            self.logger.warning(f"Hit Rate Limit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                        else:
                            self.logger.error(f"Critical Error on batch {i}: {e}")
                            raise e
                            
            self.logger.info(f"Successfully saved all {total_chunks} vectors to {str(self.db_dir)}")
        except Exception as e:
            self.logger.error(f"Failed saved vector: {e}")

    def run(self):
        self.logger.info("=== START INGESTION ===")
        
        all_documents = []
        if not os.path.exists(self.pdf_dir):
            self.logger.error(f"Folder tidak ditemukan: {self.pdf_dir}")
            return

        # Hanya memproses file yang berawalan "soal_"
        for filename in os.listdir(self.pdf_dir):
            if filename.startswith("soal_") and filename.endswith(".pdf"):
                file_path_soal = os.path.join(self.pdf_dir, filename)
                
                # Cari file jawaban yang sesuai (soal_xxx.pdf -> jawaban_xxx.pdf)
                filename_jawaban = filename.replace("soal_", "jawaban_")
                file_path_jawaban = os.path.join(self.pdf_dir, filename_jawaban)
                
                # 1. Ekstrak Kunci Jawaban
                answer_keys = self.parse_answer_key(file_path_jawaban)
                
                # 2. Parse PDF Soal dan buat deskripsi gambar (Multimodal)
                raw_text_with_captions = self.parse_pdf_multimodal(file_path_soal)
                
                # 3. Minta LLM memecahnya jadi array JSON & suntik kunci jawaban
                docs = self.structure_text_to_documents(raw_text_with_captions, filename, answer_keys)
                all_documents.extend(docs)
                
        # 4. Simpan ke Vector Store
        if all_documents:
            self.save_to_chroma(all_documents)
        else:
            self.logger.warning("Tidak ada dokumen soal yang valid untuk diproses.")

        self.logger.info("=== FINISH INGESTION ===")

if __name__ == "__main__":
    ingestor = IngestionService()
    ingestor.run()
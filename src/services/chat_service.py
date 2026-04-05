import yaml
from typing import Dict
from typing import List
import os
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.core.config import settings
from src.core.logger import get_logger
# from src.services.rag_service import RAGService
from src.domain.schemas import ChatResponse

from gliner import GLiNER
from rapidfuzz import process, fuzz


"""
main function
    - get query input from user (main.py)
    - retrieve context from database
    - create prompt -> send to llm -> return the answer
"""

class ChatService():
    def __init__(self):
        # initialize logger
        self.logger = get_logger("ChatService")

        self.chunks_dir = settings.DATA_DIR / "debug"

        # initialize GLiNER
        self.gliner = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
        self.entity_labels = ["nomor_soal", "mata_pelajaran", "topik"]

        # Load service and model
        # self.rag = RAGService()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )

        # initialize embedding_model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            task_type="retrieval_query",
            google_api_key=settings.GOOGLE_API_KEY
        )

        # initialize vector_store
        self.vector_store = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=str(settings.CHROMA_PERSIST_DIR)
        )


        # Load prompt
        try:
            with open(settings.PROMPT_DIR, 'r') as file:
                prompt_data = yaml.safe_load(file)
                self.chat_prompt = prompt_data.get("rag_chat_prompt")
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}")
            self.chat_prompt = "Kamu adalah Kamu adalah asisten AI yang membantu menjawab pertanyaan"
        
        # Type hint (Dict[Key type, Val type])
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.logger.debug(f"type of self.store: {type(self.store)}")

    # def search(self, query: str, k: int=3) -> List[Document]:
    #     self.logger.info(f"Searching for query: '{query}'")
    #     try:
    #         self.logger.debug("Mulai jalankan pencarian context")
    #         doc_result = self.vector_store.similarity_search(query, k=k)
    #         self.logger.debug("Selesai menjalankan pencarian contect")
    #         self.logger.info(f"Found {len(doc_result)} document results")
    #         return doc_result
    #     except Exception as e:
    #         self.logger.error(f"Error during search: {str(e)}")
    #         return []
        
    def get_chunk_from_entities(self, entity: str, labels: List):
        match_result = process.extractOne(
            entity,
            labels,
            scorer = fuzz.token_set_ratio
        )
        return match_result
        
        
    # search using GLiNER
    def search(self, query: str) -> Dict:
        entities_raw = self.gliner.predict_entities(query, self.entity_labels)
        extracted = {"mata_pelajaran": None, "nomor_soal": None, "topik": None}
        
        for ent in entities_raw:
            label = ent["label"].lower()
            if label in extracted:
                extracted[label] = ent["text"]

        mata_pelajaran = extracted["mata_pelajaran"]
        nomor_soal = extracted["nomor_soal"]
        topik = extracted["topik"]

        # Validasi: Jika GLiNER tidak mendeteksi entitas sama sekali
        if not mata_pelajaran and not nomor_soal and not topik:
            return {
                "status": "failed", 
                "pesan": "Tidak ada entitas yang terdeteksi oleh GLiNER. Memicu fallback.",
                "data": []
            }

        hasil_konteks = []


        # 3. Iterasi file JSON di direktori
        for filename in os.listdir(self.chunks_dir):
            if filename.startswith("debug") and filename.endswith(".json"):
                file_path_chunk = os.path.join(self.chunks_dir, filename)

                with open(file_path_chunk, 'r', encoding='utf-8') as file:
                    chunk = json.load(file)
                
                # Ekstrak unique subject dari file ini untuk dicek
                daftar_subject = list(set([soal.get("subject") for soal in chunk if "subject" in soal]))
                
                # LOGIKA 1: JIKA MATA PELAJARAN DITEMUKAN OLEH GLiNER
                if mata_pelajaran:
                    match_subject = self.get_chunk_from_entities(mata_pelajaran, daftar_subject)
                    
                    # Jika ada kecocokan subject (threshold skor > 70)
                    if match_subject and match_subject[1] > 70:
                        subject_terpilih = match_subject[0]
                        # Filter soal yang subject-nya cocok
                        soal_subject_terfilter = [soal for soal in chunk if soal.get("subject") == subject_terpilih]
                        
                        # LOGIKA 1.A: JIKA ADA NOMOR SOAL
                        if nomor_soal:
                            # Pencarian id_soal menggunakan Exact Match (bukan fuzzy) karena ini angka pasti
                            for soal in soal_subject_terfilter:
                                if str(soal.get("id_soal")) == str(nomor_soal):
                                    hasil_konteks.append(soal)
                                    break
                                    
                        # LOGIKA 1.B: JIKA TIDAK ADA NOMOR SOAL, TAPI ADA TOPIK
                        elif topik:
                            daftar_topik = list(set([soal.get("topik") for soal in soal_subject_terfilter if "topik" in soal]))
                            match_topik = self.get_chunk_from_entities(topik, daftar_topik)
                            
                            # Threshold topik sedikit lebih tinggi (75) agar lebih presisi
                            if match_topik and match_topik[1] > 75:
                                topik_terpilih = match_topik[0]
                                hasil_konteks.extend([soal for soal in soal_subject_terfilter if soal.get("topik") == topik_terpilih])
                        
                        # LOGIKA 1.C: JIKA HANYA ADA MATA PELAJARAN SAJA
                        else:
                            hasil_konteks.extend(soal_subject_terfilter)
                            
                        # Karena 1 file JSON biasanya mewakili 1 subject (berdasarkan nama file debug Anda), 
                        # kita bisa melakukan 'break' dari loop file untuk menghemat waktu (latency rendah).
                        break

        # Evaluasi akhir: Apakah ada data yang berhasil dikumpulkan?
        if not hasil_konteks:
            return {
                "status": "failed", 
                "pesan": "Entitas dikenali, tetapi tidak ada kecocokan di database JSON.",
                "data": []
            }

        return {
            "status": "success",
            "pesan": "Konteks berhasil diambil menggunakan Exact/Fuzzy matching JSON.",
            "data": hasil_konteks
        }
        
    def format_docs(self, docs: List[Document]) -> str:
        self.logger.info("Formatting docs....")
        return "\n\n".join([doc.page_content for doc in docs])    
    
    def get_session_history(self, user_id: str) -> BaseChatMessageHistory:
        if user_id not in self.store:
            self.store[user_id] = ChatMessageHistory()
        return self.store[user_id]
    
    def generate_response(self, query: str, user_id: str) -> ChatResponse:
        self.logger.info(f"Processing chat for session: {user_id}")
        # self.logger.debug(f"chat_prompt: {self.chat_prompt}")

        # RAG Retrieval
        # docs = self.search(query=query, k=3)
        # context_text = self.format_docs(docs)

        # RAG Retrieval (Memanggil fungsi GLiNER)
        docs = self.search(query=query)

        if docs["status"] == "success":
            # Ubah struktur array JSON menjadi format string yang rapi untuk dibaca LLM
            context_text = json.dumps(docs["data"], indent=2, ensure_ascii=False)
            sources_list = docs["data"]
        else:
            # TODO: Di sini nanti Anda bisa memasukkan logika fallback ke ChromaDB (Vector Search)
            context_text = "Tidak ada konteks spesifik yang ditemukan."
            sources_list = []

        # get source
        # sources_list = list([doc for doc in docs])

        # Create prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.chat_prompt + "\n\nKonteks Referensi:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # self.logger.debug(f"prompt: {prompt}")

        # Build Chain
        # Chain: Prompt -> LLM -> Ubah ke String
        chain = prompt | self.llm | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        # Execute
        try:
            # invoke chain dengan user_id
            self.logger.debug("Starting generate answer")
            answer = chain_with_history.invoke(
                {"input": query, "context": context_text},
                config={"configurable": {"session_id": user_id}}
            )

            self.logger.info(f"Response generated for session: {user_id} successfully.")

            # Return Standard Schema
            return ChatResponse(
                answer=answer,
                sources=[sources_list]
            )
        except Exception as e:
            self.logger.error(f"Error generating response for session: {user_id} - {e}")
            return ChatResponse(
                answer="Maaf, terjadi kesalahan saat memproses permintaan Anda.",
                sources=""
            )
        
if __name__ == "__main__":
    import time
    import json  # Tambahkan import json untuk merapikan output sources
    from src.domain.schemas import ChatRequest
    
    # Inisialisasi Service
    chat_service = ChatService()

    # --- KONFIGURASI HARGA (Berdasarkan Gemini 1.5 Flash) ---
    PRICE_INPUT_1M_USD = 0.075
    PRICE_OUTPUT_1M_USD = 0.30
    KURS_RUPIAH = 16000

    def hitung_metrik_biaya(query_text: str, response_text: str, sources_list: list, history_text: str) -> dict:
        """
        Fungsi helper untuk mengestimasi penggunaan token dan biaya.
        (1 token secara kasar sama dengan ~4 karakter teks)
        """
        # Konversi sources ke string untuk dihitung panjangnya
        context_text = str(sources_list)
        
        # Hitung estimasi token (Riwayat obrolan ikut dihitung)
        total_karakter_input = len(query_text) + len(context_text) + len(history_text)
        input_tokens = total_karakter_input / 4
        output_tokens = len(response_text) / 4

        # Kalkulasi biaya dalam USD dan IDR
        cost_usd = (input_tokens / 1_000_000 * PRICE_INPUT_1M_USD) + (output_tokens / 1_000_000 * PRICE_OUTPUT_1M_USD)
        cost_idr = cost_usd * KURS_RUPIAH

        return {
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_usd": cost_usd,
            "cost_idr": cost_idr
        }

    print("✅ ChatService siap! Ketik 'Close' untuk keluar.\n")

    while True:
        query = input("🧑 Tanya sesuatu: ")
        if query.lower() == "close":
            break
            
        user_id = "0001"
        req = ChatRequest(query=query, user_id=user_id)

        # Ambil history obrolan SEBELUM proses RAG (Hanya untuk kalkulasi estimasi token input)
        riwayat_obj_sebelum = chat_service.get_session_history(user_id).messages
        history_text = " ".join([msg.content for msg in riwayat_obj_sebelum])

        # 1. Mulai Timer Latency
        start_time = time.perf_counter()

        # Eksekusi RAG
        response = chat_service.generate_response(query=req.query, user_id=req.user_id)
        
        # 2. Hentikan Timer Latency
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # 3. Hitung Estimasi Biaya
        metrik = hitung_metrik_biaya(query, response.answer, response.sources, history_text)

        # Ambil history obrolan SESUDAH proses RAG (Agar jawaban AI terbaru juga ikut tercetak)
        riwayat_obj_sesudah = chat_service.get_session_history(user_id).messages

        # ==========================================
        # CETAK HASIL DENGAN FORMAT DEBUGGING LENGKAP
        # ==========================================
        print("\n" + "="*80)
        
        # --- CETAK 1: CONTEXT / SOURCES ---
        print("🔍 [1] RAG CONTEXT (JSON SOURCES):")
        if response.sources and response.sources != "" and response.sources != []:
            try:
                # Format output JSON agar rapi dan mudah dibaca di terminal
                print(json.dumps(response.sources, indent=2, ensure_ascii=False))
            except Exception:
                print(response.sources)
        else:
            print("Tidak ada konteks spesifik yang diambil dari file JSON.")
        print("-" * 80)
        
        # --- CETAK 2: HISTORY PERCAKAPAN ---
        print("🕰️ [2] CHAT HISTORY (MEMORY):")
        if not riwayat_obj_sesudah:
            print("Belum ada riwayat percakapan.")
        else:
            for msg in riwayat_obj_sesudah:
                role = "🧑 User" if msg.type == "human" else "🤖 AI"
                # Jika pesan terlalu panjang, kita batasi tampilannya di console (opsional)
                # print(f"{role}: {msg.content[:150]}..." if len(msg.content) > 150 else f"{role}: {msg.content}")
                print(f"{role}: {msg.content}")
        print("-" * 80)

        # --- CETAK 3: JAWABAN AI ---
        print("💬 [3] RAG RESPONSE:")
        print(response.answer)
        print("-" * 80)
        
        # --- CETAK 4: METRIK BIAYA ---
        print("📊 [4] METRIK PERFORMA & BIAYA:")
        print(f"⏱️ Latency       : {elapsed_time:.2f} detik")
        print(f"🪙 Est. Token    : {metrik['input_tokens']} (Input) | {metrik['output_tokens']} (Output)")
        print(f"💵 Est. Biaya    : ${metrik['cost_usd']:.6f}  (~ Rp {metrik['cost_idr']:.4f})")
        print(f"📚 Hist. Length  : {len(riwayat_obj_sesudah)} pesan di memori saat ini.")
        print("="*80 + "\n")
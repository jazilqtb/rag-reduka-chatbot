from pydantic import BaseModel, Field
from typing import List, Optional

# ==========================================
# MODEL UNTUK SUMBER DOKUMEN (Opsional tapi penting)
# ==========================================
class SourceItem(BaseModel):
    subject: str = Field(..., description="Mata pelajaran/subtes dari soal")
    jenis_ujian: str = Field(..., description="Jenis ujian, misal: Tryout 1")
    topik: str = Field(..., description="Topik soal yang dibahas")
    # konten: str  <-- (Kita tidak perlu mengirim balik seluruh teks soal ke FE agar response tidak berat)

# ==========================================
# REQUEST: Payload yang dikirim Backend Golang ke RAG (Python)
# ==========================================
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="ID unik dari tabel User di MySQL. Digunakan untuk memisahkan ingatan obrolan (Redis).")
    query: str = Field(..., description="Pertanyaan atau keluhan siswa terkait soal UTBK.")

# ==========================================
# RESPONSE: Balasan dari RAG (Python) ke Backend Golang
# ==========================================
class ChatResponse(BaseModel):
    answer: str = Field(..., description="Teks jawaban dari Tutor AI yang sudah dirakit oleh LLM.")
    sources: Optional[List] = Field(default=[], description="Daftar sumber dokumen yang dibaca AI (bisa digunakan FE untuk menampilkan label 'Referensi: Penalaran Matematika').")

if __name__ == "__main__":
    print("mulai")
    req = ChatRequest(
        user_id = "0001", 
        query = "Query test"
    )

    print(req)
    print(req.query)
    print(req.user_id)
    
    req.query = "test query new"
    req.user_id = "session id test new"
    print(req)
    print(req.query)
    print(req.user_id)
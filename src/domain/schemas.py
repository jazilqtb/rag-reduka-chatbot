from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pertanyaan user terkait BPJS")
    session_id: str = Field(..., description="ID unik session untuk memori percakapan")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="Jawaban dari AI")
    sources: str = Field(..., description="Daftar dokumen referensi yang digunakan")

if __name__ == "__main__":
    print("mulai")
    req = ChatRequest(
        query="test query", 
        session_id="session id test"
    )

    print(req)
    print(req.query)
    print(req.session_id)
    
    req.query = "test query new"
    req.session_id = "session id test new"
    print(req)
    print(req.query)
    print(req.session_id)
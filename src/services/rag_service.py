from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from src.core.config import settings
from src.core.logger import get_logger

class RAGService():
    def __init__(self):
        # initialize logger
        self.logger = get_logger("RAGService")

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
    
    def search(self, query: str, k: int=3) -> List[Document]:
        self.logger.info(f"Searching for query: '{query}'")
        try:
            self.logger.debug("Mulai jalankan pencarian context")
            doc_result = self.vector_store.similarity_search(query, k=k)
            self.logger.debug("Selesai menjalankan pencarian contect")
            self.logger.info(f"Found {len(doc_result)} document results")
            return doc_result
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []
    
    def format_docs(self, docs: List[Document]) -> str:
        self.logger.info("Formatting docs....")
        return "\n\n".join([doc.page_content for doc in docs])

if __name__ == "__main__":
    # Test Block
    rag = RAGService()
    
    # Pastikan query relevan dengan PDF yang Anda ingest sebelumnya
    test_query = "saya usia 30 tahun dan ingin berobat tapi tidak punya uang, apa yang harus saya lakukan?" 

    results = rag.search(test_query, k=5)
    
    if results:
        final_text = rag.format_docs(results)
        print(f"\n====== HASIL PENCARIAN UNTUK: '{test_query}' ======")
        print(final_text + "...")
    else:
        print("Tidak ada dokumen ditemukan.")



import re

from typing import List


from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


from src.core.config import settings
from src.core.logger import get_logger
from src.services.regex_entities_extractor import RegexEntityExtractor

class RAGService():
    def __init__(self):
        # initialize logger
        self.logger = get_logger("RAGService")
        
        self.regex_entity_ext = RegexEntityExtractor()
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

    def _entities_parser_llm(self, query: str) -> Dict:
        pass

    def _similarity_search(self, query: str, k: int=3) -> List[Document]:
        self.logger.info(f"Start: Similarity search for : '{query}")
        try:
            source = self.vector_store.similarity_search(query=query, k=k)
            return source
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []
    
    def _exact_search(self, query: str, id_soal: str="", subject: str="") -> List[Document]:
        pass

    def search():
        pass
    
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



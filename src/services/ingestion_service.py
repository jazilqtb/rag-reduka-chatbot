import os
import shutil
import time

from typing import List

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

from src.core.config import settings 
from src.core.logger import get_logger

class IngestionService():
    def __init__(self):
        self.logger = get_logger("IngestionService")

        self.pdf_dir = settings.DATA_DIR / "raw_docs"
        self.db_dir = settings.CHROMA_PERSIST_DIR

        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def load_pdfs(self) -> List[Document]:
        documents = []

        if not os.path.exists(self.pdf_dir):
            self.logger.info("File/Folder not Exist")
            return []
        
        files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]
        self.logger.info(f"Ditemukan {len(files)} file PDF.")

        for file_name in files:
            file_path = self.pdf_dir/file_name
            try:
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
                documents.extend(docs)
                self.logger.info(f"Loaded: {file_name} ({len(docs)} pages)")
            except Exception as e:
                self.logger.error(f"Failed to load {file_name}: {str(e)}")
        return documents
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        if not documents:
            self.logger.info("The documents for splitted is empty")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False)
        
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks.")
        # self.logger.info(f"type of variable chunks is: {type(chunks)}")
        try:
            chunks_dir = os.path.join(settings.DATA_DIR,"chunks")
            if os.path.exists(chunks_dir):
                try:
                    shutil.rmtree(chunks_dir)
                    self.logger.info("The old chunks has been successfully deleted (Reset).")
                except Exception as e:
                    self.logger.error(f"Failed to delete the old DB: {str(e)}")
            
            if not os.path.exists(chunks_dir):
                os.makedirs(chunks_dir)

            chunks_path = os.path.join(chunks_dir, "chunks.txt")


            with open(chunks_path, 'w') as file:
                for item in chunks:
                    file.write(f"{item}\\n")
            self.logger.info(f"Successfully store chunks in {chunks_path}")
        except Exception as e:
            self.logger.info(f"Failed to store chunks: {str(e)}")

        return chunks
    
    def save_to_chroma(self, chunks: List[Document]):
        if not chunks:
            self.logger.warning("Chunks is empty.")
            return
        
        if self.db_dir.exists():
            try:
                shutil.rmtree(self.db_dir)
                self.logger.info("The old database has been successfully deleted (Reset).")
            except Exception as e:
                self.logger.error(f"Failed to delete the old DB: {str(e)}")
        
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                task_type="retrieval_document",
                google_api_key=settings.GOOGLE_API_KEY)
            
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(self.db_dir)
            )

            BATCH_SIZE = 20
            total_chunks = len(chunks)
            
            self.logger.info(f"Total chunks: {total_chunks}. Strategy: Batch {BATCH_SIZE} with Retry Logic.")

            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                
                self.logger.info(f"Processing batch {i} to {i + len(batch)}...")
                
                max_retries = 3

                for attempt in range(max_retries):
                    try:
                        vector_store.add_documents(documents=batch)
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
        
        raw_docs = self.load_pdfs()

        if raw_docs:
            chunks=self.split_documents(raw_docs)
            self.save_to_chroma(chunks)
        else:
            self.logger.warning("Ingestion cancelled due to lack of documents")

        self.logger.info("=== FINISH INGESTION ===")

if __name__ == "__main__":
    print("Ingestion Service Starting...!")
    ingestor = IngestionService()
    ingestor.run()


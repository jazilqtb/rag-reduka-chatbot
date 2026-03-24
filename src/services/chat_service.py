import yaml
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from src.core.config import settings
from src.core.logger import get_logger
from src.services.rag_service import RAGService
from src.domain.schemas import ChatResponse


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

        # Load service and model
        self.rag = RAGService()
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GENAI_MODEL,
            api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )

        # Load prompt
        try:
            with open(settings.PROMPT_DIR, 'r') as file:
                prompt_data = yaml.safe_load(file)
                self.system_instruction = prompt_data.get("prompt")
        except Exception as e:
            self.logger.error(f"Gagal load prompt: {e}")
            self.system_instruction = "Kamu adalah Kamu adalah asisten AI yang membantu menjawab pertanyaan BPJS."
        
        # Type hint (Dict[Key type, Val type])
        self.store: Dict[str, BaseChatMessageHistory] = {}
        self.logger.debug(f"type of self.store: {type(self.store)}")

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def generate_response(self, query: str, session_id: str) -> ChatResponse:
        self.logger.info(f"Processing chat for session: {session_id}")
        # self.logger.debug(f"system_instruction: {self.system_instruction}")

        # RAG Retrieval
        docs = self.rag.search(query=query, k=3)
        context_text = self.rag.format_docs(docs)

        # get source
        sources_list = list([doc for doc in docs])

        # Create prompt Template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction + "\n\nKonteks Referensi:\n{context}"),
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
            # invoke chain dengan session_id
            self.logger.debug("Starting generate answer")
            answer = chain_with_history.invoke(
                {"input": query, "context": context_text},
                config={"configurable": {"session_id": session_id}}
            )

            self.logger.info(f"Response generated for session: {session_id} successfully.")

            # Return Standard Schema
            return ChatResponse(
                answer=answer,
                sources=f"{sources_list}"
            )
        except Exception as e:
            self.logger.error(f"Error generating response for session: {session_id} - {e}")
            return ChatResponse(
                answer="Maaf, terjadi kesalahan saat memproses permintaan Anda.",
                sources=""
            )
        
if __name__ == "__main__":
    # Test Block
    from src.domain.schemas import ChatRequest
    chat_service = ChatService()

    while True:
        query = input("type your question:")
        if query=="Close":
            break
        session_id = "0001"

        req = ChatRequest(query=query, session_id=session_id)

        response = chat_service.generate_response(query=req.query, session_id=req.session_id)

        # print(f"response ({type(response)}): {response}")
        print(f"response.answer ({type(response.answer)}): {response.answer}")
        # print(f"response.sources ({type(response.sources)}): {response.sources}")
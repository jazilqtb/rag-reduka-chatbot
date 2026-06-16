from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent.parent / ".env",  # ← TAMBAH .parent
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # ── Path Project ──────────────────────────────────────────────────────────
    BASE_DIR:   Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR:   Path = BASE_DIR / "data"
    PROMPT_DIR: Path = BASE_DIR / "config" / "prompts.yaml"

    # ── Google Gemini ─────────────────────────────────────────────────────────
    GOOGLE_API_KEY:  str
    GENAI_MODEL:     str = "models/gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: Path = DATA_DIR / "vector_store"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST:         str 
    REDIS_PORT:         int
    REDIS_ENTITY_TTL:   int = 1800   # 30 menit  — entity cache (RetrieveService)
    REDIS_CONTEXT_TTL:  int = 1800   # 30 menit  — context cache (RetrieveService)
    REDIS_CHAT_TTL:     int = 86400  # 24 jam    — chat history (ChatService)
    REDIS_RATELIMIT_TTL:int = 60     # 1 menit   — sliding window rate limit

    # ── PostgreSQL ─────────────────────────────────────────────────────────
    POSTGRES_HOST:     str 
    POSTGRES_PORT:     int 
    POSTGRES_USER:     str
    POSTGRES_PASSWORD: str
    POSTGRES_DB:       str
 
    # ── API Security ──────────────────────────────────────────────────────────
    API_KEY: str  # Wajib di .env
 
    # ── Upload ────────────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
 
    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_CHAT_MAX: int = 30   # maks request /chat per user per menit


settings = Settings()

if __name__ == "__main__":
    print(f"BASE_DIR:       {settings.BASE_DIR}")
    print(f"DATA_DIR:       {settings.DATA_DIR}")
    print(f"PROMPT_DIR:     {settings.PROMPT_DIR}")
    print(f"POSTGRES_HOST:  {settings.POSTGRES_HOST}")
    print(f"POSTGRES_DB:    {settings.POSTGRES_DB}")
    print(f"REDIS_HOST:     {settings.REDIS_HOST}")


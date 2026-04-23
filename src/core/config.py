from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8"
    )

    # Path Project
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPT_DIR: Path = BASE_DIR / "config" / "prompts.yaml"

    GOOGLE_API_KEY: str
    GENAI_MODEL: str = "models/gemini-2.5-flash" 
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    CHROMA_PERSIST_DIR: Path = DATA_DIR / "vector_store"

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_HOST:         str = "localhost"
    REDIS_PORT:         int = 6379
    REDIS_ENTITY_TTL:   int = 1800   # 30 menit  — entity cache (RetrieveService)
    REDIS_CONTEXT_TTL:  int = 1800   # 30 menit  — context cache (RetrieveService)
    REDIS_CHAT_TTL:     int = 86400  # 24 jam    — chat history (ChatService)
    REDIS_RATELIMIT_TTL:int = 60     # 1 menit   — sliding window rate limit
 
    # ── API Security ──────────────────────────────────────────────────────────
    API_KEY: str  # Wajib di .env — digunakan BE untuk autentikasi ke RAG service
 
    # ── Upload ────────────────────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 50
 
    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_CHAT_MAX: int = 30   # maks request /chat per user per menit

settings = Settings()

if __name__=="__main__":
    import yaml

    print(f"type(settings.model_config): {type(settings.model_config)}")
    print(f"settings.model_config: {settings.model_config}")
    print(f"Path(__file__): {Path(__file__)}")
    print(f"Path(__file__).resolve(): {Path(__file__).resolve()}")
    print(f"Path(__file__).resolve().parent: {Path(__file__).resolve().parent}")
    print(f"Path(__file__).resolve().parent.parent.parent: {Path(__file__).resolve().parent.parent.parent}")
    print(f"Path(__file__).resolve().parent.parent.parent / data: {Path(__file__).resolve().parent.parent.parent / 'data'}")
    print(settings.PROMPT_DIR)

    with open(settings.PROMPT_DIR, 'r') as file:
        data = yaml.safe_load(file)
        print(f"prompts({type(data['prompts'])}): {data['prompts']}")


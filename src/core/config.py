from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        env_file_encoding="utf-8")

    # Path Project
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPT_DIR: Path = BASE_DIR / "config" / "prompts.yaml"

    GOOGLE_API_KEY: str

    GENAI_MODEL: str = "models/gemini-2.5-flash" 
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    CHROMA_PERSIST_DIR: Path = DATA_DIR / "vector_store"

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


from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    cache_dir: Path = Path(".cache/llm")
    samples_dir: Path = PROJECT_ROOT / "data" / "synthetic"


@lru_cache
def get_settings() -> Settings:
    return Settings()

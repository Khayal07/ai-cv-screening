from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSP_KEYWORDS = {"none", "self"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: Literal["development", "production"] = "development"
    frame_ancestors: str = "'none'"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    cache_dir: Path = Path(".cache/llm")
    samples_dir: Path = PROJECT_ROOT / "data" / "synthetic"
    rate_limit_per_minute: int = 5
    rate_limit_per_day: int = 30
    trusted_proxy_hops: int = 0
    daily_model_call_limit: int = 100

    @field_validator("frame_ancestors")
    @classmethod
    def quote_csp_keywords(cls, value: str) -> str:
        """.env files strip quotes, so accept bare none/self and quote them as CSP needs."""
        tokens = value.split() or ["'none'"]
        return " ".join(f"'{t.lower()}'" if t.lower() in CSP_KEYWORDS else t for t in tokens)


@lru_cache
def get_settings() -> Settings:
    return Settings()

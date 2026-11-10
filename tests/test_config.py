from pathlib import Path

import pytest

from app.config import Settings


def test_settings_read_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("CACHE_DIR", "some/dir")
    settings = Settings(_env_file=None)
    assert settings.llm_model == "test-model"
    assert settings.cache_dir == Path("some/dir")

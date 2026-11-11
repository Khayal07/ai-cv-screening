from pathlib import Path

import pytest

from app.llm.cache import CachedLLMClient
from app.llm.client import LLMResponseError
from tests.conftest import FakeLLMClient


def test_cache_miss_calls_inner_and_stores(tmp_path: Path) -> None:
    fake = FakeLLMClient([{"a": 1}])
    client = CachedLLMClient(fake, tmp_path)
    assert client.complete_json("sys", "user") == {"a": 1}
    assert len(fake.calls) == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    fake = FakeLLMClient([{"a": 1}])
    client = CachedLLMClient(fake, tmp_path)
    client.complete_json("sys", "user")
    assert client.complete_json("sys", "user") == {"a": 1}
    assert len(fake.calls) == 1


def test_cache_key_is_stable(tmp_path: Path) -> None:
    client = CachedLLMClient(FakeLLMClient(), tmp_path, key_salt="v1")
    assert client.cache_key("s", "u") == client.cache_key("s", "u")
    assert client.cache_key("s", "u") != client.cache_key("s", "u2")


def test_key_salt_changes_key(tmp_path: Path) -> None:
    v1 = CachedLLMClient(FakeLLMClient(), tmp_path, key_salt="v1")
    v2 = CachedLLMClient(FakeLLMClient(), tmp_path, key_salt="v2")
    assert v1.cache_key("s", "u") != v2.cache_key("s", "u")


def test_errors_are_not_cached(tmp_path: Path) -> None:
    fake = FakeLLMClient([LLMResponseError("bad"), {"ok": True}])
    client = CachedLLMClient(fake, tmp_path)
    with pytest.raises(LLMResponseError):
        client.complete_json("sys", "user")
    assert list(tmp_path.glob("*.json")) == []
    assert client.complete_json("sys", "user") == {"ok": True}


def test_has_reports_cached_prompts_only(tmp_path: Path) -> None:
    client = CachedLLMClient(FakeLLMClient([{"a": 1}]), tmp_path)
    assert not client.has("sys", "user")
    client.complete_json("sys", "user")
    assert client.has("sys", "user")
    assert not client.has("sys", "other")

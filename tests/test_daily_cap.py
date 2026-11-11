from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.extraction import prompt_version
from app.llm.cache import CachedLLMClient
from app.llm.daily_cap import DailyCapLLMClient, DailyLimitReached, seconds_until_next_day
from app.main import DAILY_LIMIT_MESSAGE, _build_client, app, get_llm_client
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used

NOON = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOON) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def capped(tmp_path: Path, limit: int, responses: int = 10, clock: Clock | None = None) -> DailyCapLLMClient:
    inner = FakeLLMClient([{"ok": True}] * responses)
    return DailyCapLLMClient(inner, limit, usage_dir=tmp_path / "usage", now=clock or Clock())


def test_calls_up_to_the_limit_then_raises(tmp_path: Path) -> None:
    client = capped(tmp_path, limit=2)
    client.complete_json("s", "a")
    client.complete_json("s", "b")
    with pytest.raises(DailyLimitReached) as reached:
        client.complete_json("s", "c")
    assert reached.value.retry_after == 12 * 3600
    assert client.used_today() == 2


def test_blocked_call_never_reaches_the_provider(tmp_path: Path) -> None:
    inner = FakeLLMClient([{"ok": True}])
    client = DailyCapLLMClient(inner, 1, usage_dir=tmp_path, now=Clock())
    client.complete_json("s", "a")
    with pytest.raises(DailyLimitReached):
        client.complete_json("s", "b")
    assert len(inner.calls) == 1


def test_failed_calls_count(tmp_path: Path) -> None:
    inner = FakeLLMClient([RuntimeError("provider down"), {"ok": True}])
    client = DailyCapLLMClient(inner, 5, usage_dir=tmp_path, now=Clock())
    with pytest.raises(RuntimeError):
        client.complete_json("s", "a")
    client.complete_json("s", "a")
    assert client.used_today() == 2


def test_count_resets_on_the_next_utc_day(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 11, 11, 23, 59, tzinfo=UTC))
    client = capped(tmp_path, limit=1, clock=clock)
    client.complete_json("s", "a")
    with pytest.raises(DailyLimitReached):
        client.complete_json("s", "b")
    clock.now += timedelta(minutes=2)
    client.complete_json("s", "b")
    assert sorted(path.name for path in (tmp_path / "usage").iterdir()) == [
        "2026-11-11.json",
        "2026-11-12.json",
    ]


def test_day_follows_utc_not_local_time(tmp_path: Path) -> None:
    east = datetime(2026, 11, 12, 1, 0, tzinfo=UTC).astimezone()  # same instant, local zone
    client = capped(tmp_path, limit=5, clock=Clock(east))
    client.complete_json("s", "a")
    assert (tmp_path / "usage" / "2026-11-12.json").exists()


def test_count_survives_a_new_client(tmp_path: Path) -> None:
    capped(tmp_path, limit=2).complete_json("s", "a")
    restarted = capped(tmp_path, limit=2)
    restarted.complete_json("s", "b")
    with pytest.raises(DailyLimitReached):
        restarted.complete_json("s", "c")


@pytest.mark.parametrize("content", ["not json", '{"calls": -1}', '{"other": 1}', '{"calls": "3"}'])
def test_unreadable_usage_file_fails_closed(tmp_path: Path, content: str) -> None:
    usage = tmp_path / "usage"
    usage.mkdir()
    (usage / "2026-11-11.json").write_text(content, encoding="utf-8")
    with pytest.raises(DailyLimitReached):
        capped(tmp_path, limit=100).complete_json("s", "a")


def test_zero_limit_blocks_every_call(tmp_path: Path) -> None:
    with pytest.raises(DailyLimitReached):
        capped(tmp_path, limit=0).complete_json("s", "a")


def test_cache_hits_do_not_count(tmp_path: Path) -> None:
    cap = capped(tmp_path, limit=1)
    client = CachedLLMClient(cap, tmp_path / "cache")
    for _ in range(5):
        assert client.complete_json("s", "a") == {"ok": True}
    assert cap.used_today() == 1


def test_model_name_passes_through_so_cache_keys_are_unchanged(tmp_path: Path) -> None:
    inner = FakeLLMClient()
    plain = CachedLLMClient(inner, tmp_path, key_salt=prompt_version)
    wrapped = CachedLLMClient(capped(tmp_path, limit=1), tmp_path, key_salt=prompt_version)
    assert wrapped.model == inner.model
    assert wrapped.cache_key("s", "u") == plain.cache_key("s", "u")


def test_seconds_until_next_day() -> None:
    assert seconds_until_next_day(NOON) == 12 * 3600
    assert seconds_until_next_day(datetime(2026, 11, 11, 23, 59, 59, tzinfo=UTC)) == 1


def test_app_client_puts_the_cap_inside_the_cache(tmp_path: Path) -> None:
    client = _build_client("test-key", "test-model", tmp_path, 7)
    assert isinstance(client, CachedLLMClient)
    assert isinstance(client._inner, DailyCapLLMClient)


def test_default_limit_is_100() -> None:
    assert Settings(_env_file=None).daily_model_call_limit == 100


# ---------- API ----------

CV_JSON = {"skills": used("Python"), "roles": [role(5, "Python")], "education_level": "bachelor"}
JOB_JSON = {"required_skills": ["Python"], "min_years": 3}


@pytest.fixture
def api(tmp_path: Path) -> Iterator[TestClient]:
    cap = DailyCapLLMClient(
        FakeLLMClient([CV_JSON, JOB_JSON] * 3), 2, usage_dir=tmp_path / "usage", now=Clock()
    )
    client = CachedLLMClient(cap, tmp_path / "cache")
    app.dependency_overrides[get_llm_client] = lambda: client
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_returns_503_when_the_daily_cap_is_reached(api: TestClient) -> None:
    body = {"cv_text": VALID_CV_TEXT, "job_text": "Python developer"}
    assert api.post("/match", json=body).status_code == 200  # 2 real calls
    assert api.post("/match", json=body).status_code == 200  # cache hits, free
    other = dict(body, job_text="Senior Python developer")
    response = api.post("/match", json=other)
    assert response.status_code == 503
    assert response.json() == {"detail": DAILY_LIMIT_MESSAGE}
    assert response.headers["retry-after"] == str(12 * 3600)

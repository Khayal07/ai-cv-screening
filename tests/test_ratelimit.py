from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app, get_llm_client, get_rate_limiter
from app.ratelimit import (
    DAY,
    MINUTE,
    RateLimiter,
    RateLimitExceeded,
    Window,
    client_ip,
    limit_message,
    wait_text,
)
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def limiter(per_minute: int = 5, per_day: int = 30, clock: FakeClock | None = None) -> RateLimiter:
    windows = [Window(MINUTE, per_minute), Window(DAY, per_day)]
    return RateLimiter(windows, clock=clock or FakeClock())


def hit(rate: RateLimiter, key: str, times: int) -> None:
    for _ in range(times):
        rate.check(key)


# ---------- limiter ----------


def test_allows_up_to_the_minute_limit_then_blocks() -> None:
    clock = FakeClock()
    rate = limiter(clock=clock)
    hit(rate, "a", 5)
    with pytest.raises(RateLimitExceeded) as blocked:
        rate.check("a")
    assert blocked.value.retry_after == 60


def test_minute_window_slides() -> None:
    clock = FakeClock()
    rate = limiter(clock=clock)
    hit(rate, "a", 4)
    clock.now += 30
    rate.check("a")
    clock.now += 31
    hit(rate, "a", 4)  # the first 4 have expired, the 5th (at +30 s) still counts
    with pytest.raises(RateLimitExceeded) as blocked:
        rate.check("a")
    assert blocked.value.retry_after == 29


def test_daily_limit_blocks_even_when_minutes_are_spread() -> None:
    clock = FakeClock()
    rate = limiter(per_minute=5, per_day=3, clock=clock)
    for _ in range(3):
        rate.check("a")
        clock.now += 120
    with pytest.raises(RateLimitExceeded) as blocked:
        rate.check("a")
    assert blocked.value.retry_after == DAY - 360
    clock.now += DAY
    rate.check("a")


def test_blocked_requests_are_not_counted() -> None:
    clock = FakeClock()
    rate = limiter(per_minute=1, clock=clock)
    rate.check("a")
    for _ in range(10):
        with pytest.raises(RateLimitExceeded):
            rate.check("a")
    clock.now += 61
    rate.check("a")


def test_clients_have_separate_counts() -> None:
    rate = limiter(per_minute=1)
    rate.check("a")
    rate.check("b")
    with pytest.raises(RateLimitExceeded):
        rate.check("a")


def test_zero_disables_a_window() -> None:
    rate = limiter(per_minute=0, per_day=0)
    hit(rate, "a", 100)


def test_tracked_clients_stay_bounded() -> None:
    clock = FakeClock()
    rate = RateLimiter([Window(MINUTE, 5)], clock=clock, max_clients=3)
    for key in "abcdef":
        rate.check(key)
    assert rate.tracked_clients() == 3
    clock.now += 61
    rate.check("g")
    assert rate.tracked_clients() == 1


# ---------- client ip ----------


@pytest.mark.parametrize(
    ("forwarded", "hops", "expected"),
    [
        (None, 1, "10.0.0.9"),
        ("203.0.113.7", 0, "10.0.0.9"),
        ("203.0.113.7", 1, "203.0.113.7"),
        ("198.51.100.1, 203.0.113.7", 1, "203.0.113.7"),  # left entry is client-supplied
        ("1.1.1.1, 198.51.100.1, 203.0.113.7", 2, "198.51.100.1"),
        ("203.0.113.7", 2, "10.0.0.9"),  # fewer entries than trusted hops
        ("not-an-ip", 1, "10.0.0.9"),
        ("203.0.113.7, ", 1, "10.0.0.9"),
        ("2001:db8::1", 1, "2001:db8::1"),
    ],
)
def test_client_ip(forwarded: str | None, hops: int, expected: str) -> None:
    assert client_ip("10.0.0.9", forwarded, hops) == expected


def test_client_ip_without_peer() -> None:
    assert client_ip(None, None, 0) == "unknown"


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(1, "1 second"), (40, "40 seconds"), (89, "89 seconds"), (600, "10 minutes"), (36000, "10 hours")],
)
def test_wait_text(seconds: int, text: str) -> None:
    assert wait_text(seconds) == text


# ---------- API ----------

CV_JSON = {"skills": used("Python"), "roles": [role(5, "Python")], "education_level": "bachelor"}
JOB_JSON = {"required_skills": ["Python"], "min_years": 3}
GOOD = {"cv_text": VALID_CV_TEXT, "job_text": "Python developer"}
EMPTY = {"cv_text": "", "job_text": ""}


@pytest.fixture
def api() -> Iterator[TestClient]:
    shared = limiter(per_minute=3, per_day=30)
    app.dependency_overrides[get_rate_limiter] = lambda: shared
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient([CV_JSON, JOB_JSON] * 3)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_match_returns_429_with_retry_after(api: TestClient) -> None:
    for _ in range(3):
        assert api.post("/match", json=EMPTY).status_code == 422
    response = api.post("/match", json=GOOD)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
    assert response.json() == {"detail": "Too many requests. Try again in 60 seconds."}


def test_successful_matches_count_too(api: TestClient) -> None:
    for _ in range(3):
        assert api.post("/match", json=GOOD).status_code == 200
    assert api.post("/match", json=GOOD).status_code == 429


def test_pdf_endpoint_shares_the_limit(api: TestClient) -> None:
    for _ in range(3):
        api.post("/match", json=EMPTY)
    files = {"cv_file": ("cv.txt", b"plain", "text/plain")}
    response = api.post("/match/pdf", files=files, data={"job_text": "Python"})
    assert response.status_code == 429


def test_other_routes_are_not_limited(api: TestClient) -> None:
    for _ in range(10):
        assert api.get("/health").status_code == 200


def test_forwarded_clients_are_limited_separately(api: TestClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, trusted_proxy_hops=1)
    for _ in range(3):
        api.post("/match", json=EMPTY, headers={"X-Forwarded-For": "203.0.113.7"})
    blocked = api.post("/match", json=EMPTY, headers={"X-Forwarded-For": "203.0.113.7"})
    other = api.post("/match", json=EMPTY, headers={"X-Forwarded-For": "198.51.100.4"})
    spoofed = api.post("/match", json=EMPTY, headers={"X-Forwarded-For": "9.9.9.9, 203.0.113.7"})
    assert (blocked.status_code, other.status_code, spoofed.status_code) == (429, 422, 429)


def test_forwarded_header_is_ignored_without_trusted_hops(api: TestClient) -> None:
    for index in range(3):
        api.post("/match", json=EMPTY, headers={"X-Forwarded-For": f"203.0.113.{index}"})
    response = api.post("/match", json=EMPTY, headers={"X-Forwarded-For": "198.51.100.4"})
    assert response.status_code == 429


def test_default_limits_come_from_settings() -> None:
    settings = Settings(_env_file=None)
    assert (settings.rate_limit_per_minute, settings.rate_limit_per_day) == (5, 30)
    assert settings.trusted_proxy_hops == 0


def test_limit_message_is_friendly() -> None:
    assert limit_message(7200) == "Too many requests. Try again in 2 hours."

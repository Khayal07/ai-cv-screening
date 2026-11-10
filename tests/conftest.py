import socket
from typing import Any

import pytest

VALID_CV_TEXT = """Jordan Blake
Backend Engineer

Experience
Backend Engineer, Example Systems (fictional), 2020 - present
- Built REST APIs with Python and FastAPI.
- Maintained PostgreSQL databases and Docker-based deployments.

Junior Developer, Sample Labs (fictional), 2018 - 2020
- Wrote internal tools in Python.

Education
BSc Computer Science, Example University (fictional), 2018

Skills
Python, FastAPI, PostgreSQL, Docker
"""


def used(*names: str) -> list[dict[str, Any]]:
    """CV skills shown in use (evidence)."""
    return [{"name": name, "evidence": True} for name in names]


def listed(*names: str) -> list[dict[str, Any]]:
    """CV skills that only appear in a skills list."""
    return [{"name": name, "evidence": False} for name in names]


def role(years: float, *skills: str, end: str = "2024-01") -> dict[str, Any]:
    """A role lasting `years` that ends at `end` (YYYY-MM) and used `skills`."""
    end_year, end_month = (int(part) for part in end.split("-"))
    start = end_year * 12 + (end_month - 1) - round(years * 12)
    return {"start": f"{start // 12}-{start % 12 + 1:02d}", "end": end, "skills": list(skills)}


class FakeLLMClient:
    """Returns queued responses in order; never touches the network."""

    def __init__(self, responses: list[dict[str, Any] | Exception] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture(autouse=True)
def block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow loopback (used by the event loop), refuse any external host."""
    original_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if host not in LOOPBACK_HOSTS:
            raise RuntimeError(f"network access is disabled in tests: {host}")
        return original_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)

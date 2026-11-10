import socket
from typing import Any

import pytest


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

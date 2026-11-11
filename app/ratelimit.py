"""Per-client request limits for the match endpoints.

State lives in memory, so limits hold per process: run the server with one worker.
"""

import ipaddress
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

MINUTE = 60.0
DAY = 24 * 60 * 60.0
MAX_TRACKED_CLIENTS = 10_000


@dataclass(frozen=True)
class Window:
    seconds: float
    limit: int  # 0 or less disables the window


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__(f"rate limit exceeded, retry after {retry_after}s")
        self.retry_after = retry_after


class RateLimiter:
    """Sliding-window limits per key. A blocked request is not recorded."""

    def __init__(
        self,
        windows: list[Window],
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = MAX_TRACKED_CLIENTS,
    ) -> None:
        self._windows = [window for window in windows if window.limit > 0]
        self._clock = clock
        self._max_clients = max_clients
        self._longest = max((window.seconds for window in self._windows), default=0.0)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        if not self._windows:
            return
        with self._lock:
            now = self._clock()
            hits = self._hits_for(key, now)
            wait = max(self._wait(hits, window, now) for window in self._windows)
            if wait > 0:
                raise RateLimitExceeded(max(1, math.ceil(wait)))
            hits.append(now)

    def tracked_clients(self) -> int:
        return len(self._hits)

    def _hits_for(self, key: str, now: float) -> deque[float]:
        if key not in self._hits and len(self._hits) >= self._max_clients:
            self._make_room(now)
        hits = self._hits.setdefault(key, deque())
        _expire(hits, now - self._longest)
        return hits

    def _make_room(self, now: float) -> None:
        for key in list(self._hits):
            _expire(self._hits[key], now - self._longest)
            if not self._hits[key]:
                del self._hits[key]
        while len(self._hits) >= self._max_clients:
            del self._hits[next(iter(self._hits))]

    @staticmethod
    def _wait(hits: deque[float], window: Window, now: float) -> float:
        recent = [hit for hit in hits if hit > now - window.seconds]
        if len(recent) < window.limit:
            return 0.0
        return recent[len(recent) - window.limit] + window.seconds - now


def _expire(hits: deque[float], cutoff: float) -> None:
    while hits and hits[0] <= cutoff:
        hits.popleft()


def client_ip(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """The client address: the socket peer, or the entry the trusted proxies added.

    Each trusted proxy appends the address it saw, so the client is the trusted_hops-th entry
    from the right. Entries further left are client-supplied and ignored.
    """
    fallback = peer or "unknown"
    if trusted_hops <= 0 or not forwarded_for:
        return fallback
    entries = [entry.strip() for entry in forwarded_for.split(",")]
    if len(entries) < trusted_hops:
        return fallback
    candidate = entries[-trusted_hops]
    return candidate if _is_ip(candidate) else fallback


def _is_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def wait_text(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 90 * 60:
        return f"{math.ceil(seconds / 60)} minutes"
    return f"{math.ceil(seconds / 3600)} hours"


def limit_message(retry_after: int) -> str:
    return f"Too many requests. Try again in {wait_text(retry_after)}."

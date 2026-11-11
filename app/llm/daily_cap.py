"""A hard limit on real model calls per UTC day, across all clients.

Wrap the provider client and put the cache outside it, Cached(DailyCap(OpenAI)), so cache hits
never count. Every attempt counts, including failed calls and retries. The count is stored in a
small file per day, so a process restart does not reset it.
"""

import json
import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.llm.client import LLMClient

EXHAUSTED = 2**31  # count used for an unreadable usage file


class DailyLimitReached(Exception):
    def __init__(self, retry_after: int) -> None:
        super().__init__("daily model call limit reached")
        self.retry_after = retry_after


def utc_now() -> datetime:
    return datetime.now(UTC)


class DailyCapLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        limit: int,
        usage_dir: Path,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        self._inner = inner
        self._limit = limit
        self._usage_dir = usage_dir
        self._now = now
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return self._inner.model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self._reserve()
        return self._inner.complete_json(system, user)

    def used_today(self) -> int:
        return _read_count(self._usage_path(self._now()))

    def _reserve(self) -> None:
        with self._lock:
            now = self._now()
            path = self._usage_path(now)
            used = _read_count(path)
            if used >= self._limit:
                raise DailyLimitReached(seconds_until_next_day(now))
            _write_count(path, used + 1)

    def _usage_path(self, now: datetime) -> Path:
        return self._usage_dir / f"{now.astimezone(UTC).date().isoformat()}.json"


def seconds_until_next_day(now: datetime) -> int:
    utc = now.astimezone(UTC)
    midnight = datetime.combine(utc.date() + timedelta(days=1), datetime.min.time(), UTC)
    return max(1, int((midnight - utc).total_seconds()))


def _read_count(path: Path) -> int:
    """The stored count; an unreadable file counts as exhausted, so the cap fails closed."""
    if not path.exists():
        return 0
    try:
        count = json.loads(path.read_text(encoding="utf-8"))["calls"]
    except (OSError, ValueError, KeyError, TypeError):
        return EXHAUSTED
    return count if isinstance(count, int) and count >= 0 else EXHAUSTED


def _write_count(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"calls": count}), encoding="utf-8")
    os.replace(temporary, path)

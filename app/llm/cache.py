import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.llm.client import LLMClient

KeySalt = str | Callable[[str], str]


class CachedLLMClient:
    """Wraps an LLMClient and stores successful responses on disk.

    `key_salt` is a fixed string, or a function of the system prompt (a per-prompt version),
    so bumping one prompt's version invalidates only that prompt's entries.
    """

    def __init__(self, inner: LLMClient, cache_dir: Path, key_salt: KeySalt = "") -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._key_salt = key_salt

    @property
    def model(self) -> str:
        return self._inner.model

    def salt_for(self, system: str) -> str:
        return self._key_salt(system) if callable(self._key_salt) else self._key_salt

    def cache_key(self, system: str, user: str) -> str:
        payload = json.dumps([self.model, self.salt_for(system), system, user])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def has(self, system: str, user: str) -> bool:
        """Whether this prompt is already cached (no model call would be made)."""
        return self._path(system, user).exists()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        path = self._path(system, user)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = self._inner.complete_json(system, user)
        self._write(path, result)
        return result

    def _path(self, system: str, user: str) -> Path:
        return self._cache_dir / f"{self.cache_key(system, user)}.json"

    def _write(self, path: Path, result: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")

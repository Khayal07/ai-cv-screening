import hashlib
import json
from pathlib import Path
from typing import Any

from app.llm.client import LLMClient


class CachedLLMClient:
    """Wraps an LLMClient and stores successful responses on disk."""

    def __init__(self, inner: LLMClient, cache_dir: Path, key_salt: str = "") -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self._key_salt = key_salt

    @property
    def model(self) -> str:
        return self._inner.model

    def cache_key(self, system: str, user: str) -> str:
        payload = json.dumps([self.model, self._key_salt, system, user])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        path = self._cache_dir / f"{self.cache_key(system, user)}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        result = self._inner.complete_json(system, user)
        self._write(path, result)
        return result

    def _write(self, path: Path, result: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")

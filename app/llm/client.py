from typing import Any, Protocol


class LLMResponseError(Exception):
    """Raised when the model returns output that is not a JSON object."""


class LLMClient(Protocol):
    """Every model call in the app goes through this interface."""

    @property
    def model(self) -> str: ...

    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...

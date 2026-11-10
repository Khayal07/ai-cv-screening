import json
from typing import Any

from openai import OpenAI

from app.llm.client import LLMResponseError


class OpenAIClient:
    def __init__(self, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return parse_json_object(response.choices[0].message.content or "")


def parse_json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMResponseError("expected a JSON object")
    return data

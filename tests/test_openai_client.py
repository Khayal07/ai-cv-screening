import pytest

from app.llm.client import LLMResponseError
from app.llm.openai_client import OpenAIClient, parse_json_object


def test_parse_valid_object() -> None:
    assert parse_json_object('{"x": 1}') == {"x": 1}


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", ""])
def test_parse_rejects_invalid(raw: str) -> None:
    with pytest.raises(LLMResponseError):
        parse_json_object(raw)


def test_client_exposes_model() -> None:
    client = OpenAIClient(api_key="test-key", model="some-mini")
    assert client.model == "some-mini"


def test_external_network_is_blocked() -> None:
    import socket

    with socket.socket() as sock, pytest.raises(RuntimeError, match="disabled"):
        sock.connect(("api.openai.com", 443))

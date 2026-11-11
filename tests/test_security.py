import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, create_app, get_llm_client
from app.security import (
    MAX_REQUEST_BYTES,
    BodySizeLimitMiddleware,
    content_security_policy,
)
from tests.conftest import FakeLLMClient

STATIC_DIR = Path(__file__).resolve().parent.parent / "app" / "static"


def settings(**values: object) -> Settings:
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def production() -> TestClient:
    return TestClient(create_app(settings(environment="production")))


# ---------- headers ----------


@pytest.mark.parametrize("path", ["/", "/health", "/static/app.js", "/static/theme.js", "/missing"])
def test_security_headers_on_every_response(client: TestClient, path: str) -> None:
    headers = client.get(path).headers
    assert headers["content-security-policy"] == content_security_policy("'none'")
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert "camera=()" in headers["permissions-policy"]


def test_csp_allows_no_inline_script_or_style() -> None:
    policy = content_security_policy("'none'")
    assert "'unsafe-inline'" not in policy and "'unsafe-eval'" not in policy
    assert "script-src 'self'" in policy and "style-src 'self'" in policy
    assert "img-src 'self' data:" in policy  # the favicon is a data: URI
    assert policy.endswith("frame-ancestors 'none'")


def test_frame_ancestors_is_configurable() -> None:
    client = TestClient(create_app(settings(frame_ancestors="https://huggingface.co")))
    policy = client.get("/health").headers["content-security-policy"]
    assert policy.endswith("frame-ancestors https://huggingface.co")


def test_hsts_only_in_production(client: TestClient, production: TestClient) -> None:
    assert "strict-transport-security" not in client.get("/health").headers
    assert production.get("/health").headers["strict-transport-security"] == "max-age=31536000"


def test_error_responses_carry_headers_too(client: TestClient) -> None:
    response = client.post("/match", json={"cv_text": "", "job_text": ""})
    assert response.status_code == 422
    assert response.headers["x-content-type-options"] == "nosniff"


# ---------- production mode ----------


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_are_disabled_in_production(production: TestClient, path: str) -> None:
    assert production.get(path).status_code == 404


def test_app_still_works_in_production(production: TestClient) -> None:
    assert production.get("/health").json() == {"status": "ok"}
    assert production.get("/").status_code == 200


def test_docs_are_served_in_development_without_csp(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers
    assert client.get("/openapi.json").status_code == 200


def test_environment_defaults_to_development() -> None:
    assert settings().environment == "development"
    with pytest.raises(ValueError):
        settings(environment="staging")


# ---------- request size ----------


def test_declared_oversized_body_gets_413(client: TestClient) -> None:
    body = b"x" * (MAX_REQUEST_BYTES + 1)
    response = client.post("/match", content=body, headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert response.json() == {"detail": "request body is too large"}
    assert response.headers["x-content-type-options"] == "nosniff"


@pytest.fixture
def small_limit() -> TestClient:
    tiny = FastAPI()

    @tiny.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    tiny.add_middleware(BodySizeLimitMiddleware, max_bytes=100)
    return TestClient(tiny)


def chunks(count: int, size: int = 30) -> Iterator[bytes]:
    for _ in range(count):
        yield b"x" * size


def test_streamed_body_over_the_limit_gets_413(small_limit: TestClient) -> None:
    response = small_limit.post("/echo", content=chunks(5))  # chunked, no Content-Length
    assert response.status_code == 413
    assert response.json() == {"detail": "request body is too large"}


def test_bodies_within_the_limit_pass(small_limit: TestClient) -> None:
    assert small_limit.post("/echo", content=chunks(3)).json() == {"size": 90}
    assert small_limit.post("/echo", content=b"x" * 100).json() == {"size": 100}


def test_streamed_oversized_body_gets_413_from_the_app(client: TestClient) -> None:
    megabyte = 1024 * 1024
    response = client.post(
        "/match", content=chunks(7, megabyte), headers={"content-type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "request body is too large"}


def test_large_pdf_still_gets_the_specific_message(client: TestClient) -> None:
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    try:
        pdf = b"%PDF-1.4\n" + b"0" * (5 * 1024 * 1024)
        files = {"cv_file": ("cv.pdf", pdf, "application/pdf")}
        response = client.post("/match/pdf", files=files, data={"job_text": "Python"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 413
    assert response.json() == {"detail": "PDF is larger than 5 MB"}


# ---------- no stack traces ----------


def test_unhandled_error_is_a_plain_500(caplog: pytest.LogCaptureFixture) -> None:
    broken = create_app(settings())

    def boom() -> None:
        raise RuntimeError("Jordan Blake, 41, secret cv text")

    broken.add_api_route("/boom", boom)
    with caplog.at_level(logging.ERROR, logger="app.errors"):
        response = TestClient(broken).get("/boom")
    assert response.status_code == 500
    assert response.json() == {"detail": "internal error"}
    assert "Traceback" not in response.text and "secret" not in response.text
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "RuntimeError" in caplog.text and "secret" not in caplog.text


# ---------- page markup works under the CSP ----------

HTML = (STATIC_DIR / "index.html").read_text(encoding="utf-8")


def test_page_has_no_inline_script() -> None:
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", HTML, flags=re.DOTALL | re.IGNORECASE)
    assert scripts and all(body.strip() == "" for body in scripts)
    assert all(" src=" in tag for tag in re.findall(r"<script\b[^>]*>", HTML, flags=re.IGNORECASE))


@pytest.mark.parametrize(
    "pattern",
    [r"\sstyle\s*=", r"<style\b", r"\son[a-z]+\s*=", r"javascript:"],
    ids=["style-attribute", "style-element", "inline-handler", "javascript-url"],
)
def test_page_has_nothing_the_csp_would_block(pattern: str) -> None:
    assert not re.search(pattern, HTML, flags=re.IGNORECASE)


def test_theme_script_loads_before_styles_without_defer() -> None:
    head = HTML[: HTML.index("</head>")]
    tag = re.search(r'<script src="/static/theme.js"[^>]*>', head)
    assert tag and "defer" not in tag.group() and "async" not in tag.group()
    assert head.index("theme.js") < head.index("style.css")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("'none'", "'none'"),
        ("none", "'none'"),
        ("self https://huggingface.co", "'self' https://huggingface.co"),
        ("https://huggingface.co", "https://huggingface.co"),
        ("", "'none'"),
    ],
)
def test_frame_ancestors_keywords_are_quoted(raw: str, expected: str) -> None:
    assert settings(frame_ancestors=raw).frame_ancestors == expected


def test_env_example_gives_a_valid_policy() -> None:
    example = Path(__file__).resolve().parent.parent / ".env.example"
    loaded = Settings(_env_file=example)  # type: ignore[call-arg]
    assert loaded.frame_ancestors == "'none'"

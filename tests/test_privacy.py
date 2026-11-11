"""CV text must never be logged, stored or echoed back; the page must say what happens to it."""

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.llm.cache import CachedLLMClient
from app.main import app, get_llm_client
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used

APP_DIR = Path(__file__).resolve().parent.parent / "app"
MARKER = "Zorvexian quarterly ledger reconciliation"
CV_WITH_MARKER = VALID_CV_TEXT + f"\n- {MARKER} for the finance team."
CV_JSON = {"skills": used("Python"), "roles": [role(5, "Python")], "education_level": "bachelor"}
JOB_JSON = {"required_skills": ["Python"], "min_years": 3}
LOG_CALL = re.compile(
    r"\bprint\(|\blogging\b|\blogger\b|\blog\.\w+\(|\bwarnings\.warn\(|sys\.std(out|err)"
)
# The only log line in the app: the 500 shield records the exception type, never its message.
ALLOWED_LOG_LINES = {
    "security.py": {
        "import logging",
        'log = logging.getLogger("app.errors")',
        'log.error("unhandled error: %s", type(exc).__name__)',
    },
}


def python_sources() -> list[Path]:
    return sorted(APP_DIR.rglob("*.py"))


@pytest.mark.parametrize("path", python_sources(), ids=lambda path: path.name)
def test_app_code_does_not_print_or_log(path: Path) -> None:
    allowed = ALLOWED_LOG_LINES.get(path.name, set())
    lines = path.read_text(encoding="utf-8").splitlines()
    offending = [line for line in lines if LOG_CALL.search(line) and line.strip() not in allowed]
    assert offending == [], path


def test_allowed_log_lines_still_exist() -> None:
    for name, allowed in ALLOWED_LOG_LINES.items():
        lines = {line.strip() for line in (APP_DIR / name).read_text(encoding="utf-8").splitlines()}
        assert allowed <= lines, name


@pytest.fixture
def client_with(tmp_path: Path) -> Iterator[object]:
    def build(responses: list[object]) -> TestClient:
        cached = CachedLLMClient(FakeLLMClient(responses), tmp_path)  # type: ignore[arg-type]
        app.dependency_overrides[get_llm_client] = lambda: cached
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()


def test_cache_stores_model_output_only(client_with: object, tmp_path: Path) -> None:
    client = client_with([CV_JSON, JOB_JSON])  # type: ignore[operator]
    response = client.post("/match", json={"cv_text": CV_WITH_MARKER, "job_text": "Python developer"})
    assert response.status_code == 200
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2
    for path in files:
        assert re.fullmatch(r"[0-9a-f]{64}\.json", path.name)
        content = path.read_text(encoding="utf-8")
        assert MARKER not in content and "Example Systems" not in content


def test_model_error_response_does_not_echo_the_cv(client_with: object) -> None:
    client = client_with([{"skills": 5}, {"skills": 5}])  # type: ignore[operator]
    response = client.post("/match", json={"cv_text": CV_WITH_MARKER, "job_text": "Python developer"})
    assert response.status_code == 502
    assert MARKER not in response.text


def test_validation_error_does_not_echo_the_input(client_with: object) -> None:
    client = client_with([])  # type: ignore[operator]
    too_long = MARKER + " x" * 30_000
    response = client.post("/match", json={"cv_text": too_long, "job_text": "Python developer"})
    assert response.status_code == 422
    assert MARKER not in response.text
    error = response.json()["detail"][0]
    assert set(error) == {"loc", "msg", "type"}
    assert error["loc"] == ["body", "cv_text"]


def test_not_a_cv_response_does_not_echo_the_text(client_with: object) -> None:
    client = client_with([])  # type: ignore[operator]
    response = client.post("/match", json={"cv_text": f"{MARKER} only.", "job_text": "Python"})
    assert response.status_code == 422
    assert MARKER not in response.text


def test_page_shows_the_privacy_notice() -> None:
    html = (APP_DIR / "static" / "index.html").read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", html)
    for phrase in ("Demo only.", "Please don't upload real CVs.",
                   "sent to OpenAI after personal details are removed",
                   "nothing is stored except the anonymized extraction"):
        assert phrase in text, phrase

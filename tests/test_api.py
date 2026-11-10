from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.llm.client import LLMResponseError
from app.main import MAX_PDF_BYTES, app, get_llm_client
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used
from tests.pdf_factory import image_only_pdf, text_pdf

CV_JSON: dict[str, Any] = {
    "skills": used("Python", "FastAPI"),
    "roles": [role(4, "Python", "FastAPI")],
    "education_level": "bachelor",
    "languages": ["English"],
}
JOB_JSON: dict[str, Any] = {
    "required_skills": ["Python", "Kubernetes"],
    "nice_to_have_skills": [],
    "min_years": 4,
    "education_level": "bachelor",
    "languages": ["English"],
}
CV_TEXT = VALID_CV_TEXT
JOB_TEXT = "Backend role. Python and Kubernetes required."


@pytest.fixture
def fake() -> FakeLLMClient:
    return FakeLLMClient([CV_JSON, JOB_JSON])


@pytest.fixture
def client(fake: FakeLLMClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_match_returns_explainable_score(client: TestClient) -> None:
    response = client.post("/match", json={"cv_text": CV_TEXT, "job_text": JOB_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == 75  # 25 + 15 + 20 + 10 + 5
    assert body["missing_skills"] == ["Kubernetes"]
    assert body["reasons"]
    assert "hiring decision" in body["disclaimer"]


def test_match_sends_anonymized_cv_to_model(client: TestClient, fake: FakeLLMClient) -> None:
    client.post("/match", json={"cv_text": CV_TEXT, "job_text": JOB_TEXT})
    cv_prompt = fake.calls[0][1]
    assert "Jordan" not in cv_prompt
    assert "Backend Engineer" in cv_prompt


def test_match_rejects_empty_text(client: TestClient) -> None:
    response = client.post("/match", json={"cv_text": "   ", "job_text": JOB_TEXT})
    assert response.status_code == 422


def test_invalid_model_output_maps_to_502(client: TestClient, fake: FakeLLMClient) -> None:
    fake.responses = [LLMResponseError("bad"), LLMResponseError("bad again")]
    response = client.post("/match", json={"cv_text": CV_TEXT, "job_text": JOB_TEXT})
    assert response.status_code == 502
    assert len(fake.calls) == 2


def test_match_pdf_success(client: TestClient) -> None:
    pdf = text_pdf(VALID_CV_TEXT.splitlines())
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.pdf", pdf, "application/pdf")},
        data={"job_text": JOB_TEXT},
    )
    assert response.status_code == 200
    assert response.json()["score"] == 75


def test_match_pdf_without_text_returns_422(client: TestClient) -> None:
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("scan.pdf", image_only_pdf(), "application/pdf")},
        data={"job_text": JOB_TEXT},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "no extractable text"


def test_match_pdf_invalid_file_returns_422(client: TestClient) -> None:
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.pdf", b"not a pdf", "application/pdf")},
        data={"job_text": JOB_TEXT},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid PDF file"


def test_match_pdf_rejects_non_pdf(client: TestClient) -> None:
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.txt", b"plain text", "text/plain")},
        data={"job_text": JOB_TEXT},
    )
    assert response.status_code == 415


def test_match_pdf_rejects_large_file(client: TestClient) -> None:
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.pdf", b"0" * (MAX_PDF_BYTES + 1), "application/pdf")},
        data={"job_text": JOB_TEXT},
    )
    assert response.status_code == 413


def test_missing_api_key_returns_503() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None, openai_api_key="")
    try:
        response = TestClient(app).post("/match", json={"cv_text": CV_TEXT, "job_text": JOB_TEXT})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 503

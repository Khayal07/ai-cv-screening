from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_client
from app.parsing.cv_check import MIN_WORDS, find_posting_phrases, looks_like_cv
from tests.conftest import VALID_CV_TEXT, FakeLLMClient
from tests.pdf_factory import text_pdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
CV_FILES = sorted(DATA_DIR.glob("cv_*.txt"))
JOB_FILES = sorted(DATA_DIR.glob("jd_*.txt"))

FILLER = " ".join(["word"] * MIN_WORDS)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", CV_FILES, ids=lambda p: p.name)
def test_synthetic_cvs_pass(path: Path) -> None:
    check = looks_like_cv(read(path))
    assert check.is_cv, check.reasons


@pytest.mark.parametrize("path", JOB_FILES, ids=lambda p: p.name)
def test_job_descriptions_are_rejected_as_cv(path: Path) -> None:
    check = looks_like_cv(read(path))
    assert not check.is_cv
    assert any("job posting" in reason or "job-posting" in reason for reason in check.reasons)


def test_shared_test_cv_passes() -> None:
    assert looks_like_cv(VALID_CV_TEXT).is_cv


def test_too_short_is_rejected() -> None:
    check = looks_like_cv("Backend Engineer, 2019 - 2022, BSc Computer Science")
    assert not check.is_cv
    assert "too short" in check.reasons[0]


def test_keyword_list_without_dates_or_education_is_rejected() -> None:
    check = looks_like_cv("Skills: " + ", ".join(["Python", "FastAPI", "Docker"] * 15))
    assert not check.is_cv
    assert any("No dated roles" in reason for reason in check.reasons)


def test_two_posting_phrases_without_dates_are_rejected() -> None:
    text = f"We are looking for an engineer. Nice to have: Redis. BSc preferred. {FILLER}"
    check = looks_like_cv(text)
    assert not check.is_cv
    assert any("reads like a job posting" in reason for reason in check.reasons)


def test_two_posting_phrases_with_dated_roles_pass() -> None:
    text = f"{VALID_CV_TEXT}\nResponsibilities\n- Owned releases you will see in the changelog."
    check = looks_like_cv(text)
    assert check.posting_phrases == ["you will", "responsibilities"]
    assert check.is_cv


def test_three_posting_phrases_are_rejected_even_with_dates() -> None:
    text = f"{VALID_CV_TEXT}\nWe are looking for you. Nice to have: Go. Apply now."
    check = looks_like_cv(text)
    assert not check.is_cv
    assert any("several job-posting phrases" in reason for reason in check.reasons)


@pytest.mark.parametrize(
    "text",
    [
        f"Developer, Jan 2020 - Present. {FILLER}",
        f"Developer, 03/2019 - 05/2021. {FILLER}",
        f"Developer, 2019–2022. {FILLER}",
        f"Developer, Sept 2018 to Dec 2020. {FILLER}",
        f"Developer, 2021 - now. {FILLER}",
    ],
)
def test_date_range_formats_are_recognised(text: str) -> None:
    check = looks_like_cv(text)
    assert check.date_ranges == 1
    assert check.is_cv


def test_birth_date_is_not_a_date_range() -> None:
    assert looks_like_cv("Date of birth: 1989-04-12").date_ranges == 0


def test_education_alone_is_enough_signal() -> None:
    assert looks_like_cv(f"Recent graduate. BSc Physics, Example University. {FILLER}").is_cv


def test_bare_apply_word_is_not_a_posting_phrase() -> None:
    assert find_posting_phrases("Applied machine learning to apply discounts.") == []


def test_heading_lines_count_as_posting_phrases() -> None:
    assert find_posting_phrases("Intro\nRequirements:\n- Python\nResponsibilities\n") == [
        "requirements",
        "responsibilities",
    ]


# ---------- API ----------


@pytest.fixture
def fake() -> FakeLLMClient:
    return FakeLLMClient([])


@pytest.fixture
def client(fake: FakeLLMClient) -> Iterator[TestClient]:
    app.dependency_overrides[get_llm_client] = lambda: fake
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_rejects_job_text_as_cv(client: TestClient, fake: FakeLLMClient) -> None:
    job = read(DATA_DIR / "jd_backend_engineer.txt")
    response = client.post("/match", json={"cv_text": job, "job_text": job})
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "does not look like a CV"
    assert body["reasons"]
    assert fake.calls == []


def test_api_rejects_job_pdf_as_cv(client: TestClient, fake: FakeLLMClient) -> None:
    job = read(DATA_DIR / "jd_backend_engineer.txt")
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.pdf", text_pdf(job.splitlines()), "application/pdf")},
        data={"job_text": job},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "does not look like a CV"
    assert fake.calls == []

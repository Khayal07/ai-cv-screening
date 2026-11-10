"""End-to-end checks over the synthetic CVs, using canned model output (no network)."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_client
from app.parsing.anonymize import strip_personal_data
from app.service import screen
from tests.conftest import FakeLLMClient, used
from tests.pdf_factory import text_pdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"

PERSONAL_MARKERS: dict[str, list[str]] = {
    "cv_backend_senior.txt": ["Riley", "Thornquist", "1989", "Female", "Photo", "@", "555"],
    "cv_frontend_mid.txt": ["Casey", "Marlowe", "Age", "29", "Male", "@", "caseymarlowe"],
    "cv_data_analyst_junior.txt": ["Sam", "Okonkwo", "DOB", "2001", "they/them", "7700"],
    "cv_career_changer.txt": ["Dana", "Whitfield", "Mr.", "Aged 38", "headshot"],
}

CANNED_CVS: dict[str, dict[str, Any]] = {
    "cv_backend_senior.txt": {
        "skills": used("Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Docker",
                       "Kubernetes", "AWS", "CI/CD"),
        "years_experience": 7, "education_level": "bachelor", "languages": ["English", "German"],
    },
    "cv_frontend_mid.txt": {
        "skills": used("React", "TypeScript", "JavaScript", "HTML", "CSS", "Node.js", "Git"),
        "years_experience": 4, "education_level": "bachelor", "languages": ["English"],
    },
    "cv_career_changer.txt": {
        "skills": used("JavaScript", "HTML", "CSS", "Python", "Git"),
        "years_experience": 0.5, "education_level": "master", "languages": ["English", "French"],
    },
}

CANNED_BACKEND_JOB: dict[str, Any] = {
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "Kubernetes"],
    "nice_to_have_skills": ["AWS", "Redis", "CI/CD"],
    "min_years": 4, "education_level": "bachelor", "languages": ["English"],
}


def read(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def test_synthetic_files_exist() -> None:
    assert len(list(DATA_DIR.glob("cv_*.txt"))) >= 3
    assert len(list(DATA_DIR.glob("jd_*.txt"))) >= 2


@pytest.mark.parametrize(("name", "markers"), PERSONAL_MARKERS.items())
def test_personal_data_removed_from_synthetic_cvs(name: str, markers: list[str]) -> None:
    cleaned = strip_personal_data(read(name))
    for marker in markers:
        assert marker not in cleaned, f"{marker!r} leaked from {name}"
    assert "Skills" in cleaned


def test_job_title_header_survives_anonymization() -> None:
    assert strip_personal_data(read("cv_frontend_mid.txt")).startswith("Frontend Developer")


def score_for(cv_name: str) -> int:
    fake = FakeLLMClient([CANNED_CVS[cv_name], CANNED_BACKEND_JOB])
    return screen(read(cv_name), read("jd_backend_engineer.txt"), fake).score


def test_backend_job_ranks_candidates_sensibly() -> None:
    senior = score_for("cv_backend_senior.txt")
    frontend = score_for("cv_frontend_mid.txt")
    changer = score_for("cv_career_changer.txt")
    assert senior == 100
    assert senior > frontend
    assert senior > changer


def test_pdf_upload_of_synthetic_cv() -> None:
    lines = [line for line in read("cv_backend_senior.txt").splitlines() if line.strip()]
    fake = FakeLLMClient([CANNED_CVS["cv_backend_senior.txt"], CANNED_BACKEND_JOB])
    app.dependency_overrides[get_llm_client] = lambda: fake
    try:
        response = TestClient(app).post(
            "/match/pdf",
            files={"cv_file": ("cv.pdf", text_pdf(lines), "application/pdf")},
            data={"job_text": read("jd_backend_engineer.txt")},
        )
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["score"] == 100
    assert "Thornquist" not in fake.calls[0][1]

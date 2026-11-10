"""Regression tests: pasted job text and keyword stuffing must not score like a real CV.

All model output is canned (FakeLLMClient); nothing touches the network.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app, get_llm_client
from app.parsing.cv_check import looks_like_cv
from app.service import screen
from tests.conftest import FakeLLMClient, listed, used
from tests.pdf_factory import text_pdf

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def read(name: str) -> str:
    return (DATA_DIR / name).read_text(encoding="utf-8")


def section(text: str, start: str, end: str | None = None) -> str:
    """The lines of `text` from the heading `start` up to (not including) the heading `end`."""
    begin = text.index(start)
    return text[begin : text.index(end, begin) if end else len(text)]


# Canned v3 extractions that follow what each synthetic file actually says.
BACKEND_JOB: dict[str, Any] = {
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "SQL", "Docker", "Kubernetes"],
    "nice_to_have_skills": ["AWS", "Redis", "CI/CD"],
    "min_years": 4, "education_level": "bachelor", "languages": ["English"],
}
ANALYST_JOB: dict[str, Any] = {
    "required_skills": ["SQL", "Excel", "Tableau"],
    "nice_to_have_skills": ["Python", "pandas", "Statistics"],
    "min_years": 1, "education_level": "bachelor", "languages": ["English"],
}
JOBS = {"jd_backend_engineer.txt": BACKEND_JOB, "jd_data_analyst.txt": ANALYST_JOB}

CVS: dict[str, dict[str, Any]] = {
    "cv_backend_senior.txt": {
        "skills": used("Python", "FastAPI", "Django", "PostgreSQL", "Docker", "Kubernetes", "AWS",
                       "CI/CD", "REST") + listed("Redis"),
        "roles": [
            {"start": "2020", "end": "present",
             "skills": ["Python", "FastAPI", "REST", "Kubernetes", "AWS", "PostgreSQL"]},
            {"start": "2017", "end": "2020", "skills": ["Python", "Django", "Docker", "CI/CD"]},
        ],
        "education_level": "bachelor", "languages": ["English", "German"],
    },
    "cv_frontend_mid.txt": {
        "skills": used("React", "TypeScript", "Node.js", "JavaScript", "HTML", "CSS")
        + listed("Jest", "Git"),
        "roles": [
            {"start": "2021", "end": "present", "skills": ["React", "TypeScript", "Node.js"]},
            {"start": "2020", "end": "2021", "skills": ["JavaScript", "HTML", "CSS"]},
        ],
        "education_level": "bachelor", "languages": ["English"],
    },
    "cv_data_analyst_junior.txt": {
        "skills": used("SQL", "Tableau", "Excel", "Python", "pandas") + listed("Statistics"),
        "roles": [
            {"start": "2023", "end": "2024",
             "skills": ["SQL", "Tableau", "Excel", "Python", "pandas"]},
        ],
        "education_level": "bachelor", "languages": ["English", "Spanish"],
    },
    "cv_career_changer.txt": {
        "skills": used("JavaScript", "HTML", "Python")
        + listed("CSS", "Git", "Teaching", "Communication"),
        "roles": [
            {"start": "2024", "end": "present", "skills": ["JavaScript", "HTML", "Python"]},
            {"start": "2012", "end": "2024", "skills": []},
        ],
        "education_level": "master", "languages": ["English", "French"],
    },
}


@pytest.fixture
def client() -> Iterator[TestClient]:
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_fake(fake: FakeLLMClient) -> None:
    app.dependency_overrides[get_llm_client] = lambda: fake


def score(cv_name: str, job_name: str) -> int:
    fake = FakeLLMClient([CVS[cv_name], JOBS[job_name]])
    return screen(read(cv_name), read(job_name), fake).score


# --- Job text sent as the CV is rejected before any model call ---

PASTED_JOB_TEXTS = {
    "full backend posting": read("jd_backend_engineer.txt"),
    "full analyst posting": read("jd_data_analyst.txt"),
    "requirements block only": section(read("jd_backend_engineer.txt"), "Requirements"),
    "requirements block repeated": section(read("jd_backend_engineer.txt"), "Requirements") * 3,
}


@pytest.mark.parametrize("cv_text", PASTED_JOB_TEXTS.values(), ids=PASTED_JOB_TEXTS.keys())
def test_job_text_as_cv_is_rejected_by_match(client: TestClient, cv_text: str) -> None:
    fake = FakeLLMClient()
    use_fake(fake)
    response = client.post(
        "/match", json={"cv_text": cv_text, "job_text": read("jd_backend_engineer.txt")}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "does not look like a CV"
    assert response.json()["reasons"]
    assert fake.calls == []


@pytest.mark.parametrize("job_name", JOBS)
def test_job_text_as_cv_is_rejected_by_match_pdf(client: TestClient, job_name: str) -> None:
    fake = FakeLLMClient()
    use_fake(fake)
    lines = read(job_name).splitlines()
    response = client.post(
        "/match/pdf",
        files={"cv_file": ("cv.pdf", text_pdf(lines), "application/pdf")},
        data={"job_text": read(job_name)},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "does not look like a CV"
    assert fake.calls == []


def test_bare_keyword_list_is_rejected(client: TestClient) -> None:
    keywords = " ".join(BACKEND_JOB["required_skills"] + BACKEND_JOB["nice_to_have_skills"]) * 5
    fake = FakeLLMClient()
    use_fake(fake)
    response = client.post(
        "/match", json={"cv_text": keywords, "job_text": read("jd_backend_engineer.txt")}
    )
    assert response.status_code == 422
    assert fake.calls == []


def test_requirements_pasted_into_a_real_cv_are_flagged(client: TestClient) -> None:
    job = read("jd_backend_engineer.txt")
    cv_text = read("cv_backend_senior.txt") + "\n" + section(job, "Requirements")
    use_fake(FakeLLMClient([CVS["cv_backend_senior.txt"], BACKEND_JOB]))
    response = client.post("/match", json={"cv_text": cv_text, "job_text": job})
    assert response.status_code == 200
    assert any("match the job description" in w for w in response.json()["warnings"])


# --- Keyword stuffing scores clearly lower than the same CV with evidence ---

STUFFED_CV_TEXT = """Operations Coordinator

Experience
Operations Coordinator, Harbor Freight Lines (fictional), 2018 - present
- Planned delivery schedules and handled customer escalations.

Office Assistant, Maple Dental (fictional), 2015 - 2018
- Managed appointments and supplier orders.

Skills
Python, FastAPI, PostgreSQL, SQL, Docker, Kubernetes, AWS, Redis, CI/CD

Education
BSc Business Administration, Example University (fictional), 2015

Languages
English
"""

JOB_SKILLS = BACKEND_JOB["required_skills"] + BACKEND_JOB["nice_to_have_skills"]
ROLE_DATES = [("2018", "present"), ("2015", "2018")]


def stuffed_profile(evidence: bool) -> dict[str, Any]:
    """The same CV text, extracted as keywords only or as skills used in the roles."""
    role_skills = [JOB_SKILLS, JOB_SKILLS] if evidence else [["Scheduling"], ["Scheduling"]]
    return {
        "skills": used(*JOB_SKILLS) if evidence else listed(*JOB_SKILLS),
        "roles": [
            {"start": start, "end": end, "skills": skills}
            for (start, end), skills in zip(ROLE_DATES, role_skills, strict=True)
        ],
        "education_level": "bachelor", "languages": ["English"],
    }


def stuffed_result(evidence: bool) -> Any:
    fake = FakeLLMClient([stuffed_profile(evidence), BACKEND_JOB])
    return screen(STUFFED_CV_TEXT, read("jd_backend_engineer.txt"), fake)


def test_stuffed_cv_text_passes_the_document_check() -> None:
    # The stuffing is only caught by scoring, so the text itself must look like a CV.
    assert looks_like_cv(STUFFED_CV_TEXT).is_cv


def test_keyword_only_cv_scores_clearly_lower_than_with_evidence() -> None:
    keywords, evidenced = stuffed_result(evidence=False), stuffed_result(evidence=True)
    assert evidenced.score == 100
    assert evidenced.score - keywords.score >= 15
    assert keywords.score < 70


def test_keyword_only_cv_explains_the_lower_score() -> None:
    result = stuffed_result(evidence=False)
    assert any("listed without evidence" in reason for reason in result.reasons)
    assert any(reason.startswith("Experience: 0.0 relevant years") for reason in result.reasons)


# --- The real synthetic CVs keep high scores and a sensible ranking ---

def test_senior_backend_cv_scores_high_and_ranks_first_for_backend_job() -> None:
    scores = {name: score(name, "jd_backend_engineer.txt") for name in CVS}
    assert scores["cv_backend_senior.txt"] >= 90
    assert max(scores, key=scores.__getitem__) == "cv_backend_senior.txt"
    assert min(scores, key=scores.__getitem__) == "cv_frontend_mid.txt"


def test_junior_analyst_cv_scores_high_and_ranks_first_for_analyst_job() -> None:
    scores = {name: score(name, "jd_data_analyst.txt") for name in CVS}
    assert scores["cv_data_analyst_junior.txt"] >= 90
    assert max(scores, key=scores.__getitem__) == "cv_data_analyst_junior.txt"


def test_senior_backend_reasons_show_evidence_and_relevant_years() -> None:
    fake = FakeLLMClient([CVS["cv_backend_senior.txt"], BACKEND_JOB])
    result = screen(read("cv_backend_senior.txt"), read("jd_backend_engineer.txt"), fake)
    assert result.missing_skills == []
    assert "Required skills: 6 of 6 matched (SQL via PostgreSQL)." in result.reasons
    assert any("listed without evidence: Redis" in reason for reason in result.reasons)
    experience = next(r for r in result.reasons if r.startswith("Experience:"))
    found = re.fullmatch(r"Experience: ([\d.]+) relevant years of ([\d.]+) total \(required 4\)\.", experience)
    assert found is not None
    assert float(found[1]) == float(found[2]) >= 6
    assert result.warnings == []


def test_career_changer_teacher_role_is_not_relevant_and_warns() -> None:
    fake = FakeLLMClient([CVS["cv_career_changer.txt"], BACKEND_JOB])
    result = screen(read("cv_career_changer.txt"), read("jd_backend_engineer.txt"), fake)
    assert result.warnings == [
        "1 role(s) have no listed skills and were not counted as relevant experience."
    ]

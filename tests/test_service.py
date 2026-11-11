import pytest

from app.parsing.cv_check import NotACVError
from app.service import screen
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used

CV_JSON = {"skills": used("Go"), "roles": [role(2, "Go")], "education_level": "master", "languages": []}
JOB_JSON = {"required_skills": ["Go"], "min_years": 2, "education_level": "bachelor"}


def test_screen_strips_personal_data_and_scores() -> None:
    fake = FakeLLMClient([CV_JSON, JOB_JSON])
    cv_text = "Name: Jordan Blake\nAge: 41\nGender: Male\n" + VALID_CV_TEXT
    result = screen(cv_text, "Go developer wanted", fake)
    cv_prompt = fake.calls[0][1]
    for value in ("Jordan", "41", "Male"):
        assert value not in cv_prompt
    assert result.score == 100


def test_screen_does_not_anonymize_job_text() -> None:
    fake = FakeLLMClient([CV_JSON, JOB_JSON])
    screen(VALID_CV_TEXT, "Hiring Manager Team\nGo required", fake)
    assert fake.calls[1][1] == "Hiring Manager Team\nGo required"


def test_screen_rejects_non_cv_before_any_model_call() -> None:
    fake = FakeLLMClient([CV_JSON, JOB_JSON])
    with pytest.raises(NotACVError) as error:
        screen("Skills: Python, Go, Docker, Kubernetes", "Go developer wanted", fake)
    assert error.value.reasons
    assert fake.calls == []


def test_screen_uses_the_given_date_for_present_roles() -> None:
    from datetime import date

    present = dict(CV_JSON, roles=[{"start": "2024-01", "end": "present", "skills": ["Go"]}])
    job = dict(JOB_JSON, min_years=4)
    early = screen(VALID_CV_TEXT, "Go", FakeLLMClient([present, job]), today=date(2025, 1, 1))
    late = screen(VALID_CV_TEXT, "Go", FakeLLMClient([present, job]), today=date(2028, 1, 1))
    assert early.score < late.score

from app.service import screen
from tests.conftest import FakeLLMClient

CV_JSON = {"skills": ["Go"], "years_experience": 2, "education_level": "master", "languages": []}
JOB_JSON = {"required_skills": ["Go"], "min_years": 2, "education_level": "bachelor"}


def test_screen_strips_personal_data_and_scores() -> None:
    fake = FakeLLMClient([CV_JSON, JOB_JSON])
    cv_text = "Name: Jordan Blake\nAge: 41\nGender: Male\nGo developer, 2 years"
    result = screen(cv_text, "Go developer wanted", fake)
    cv_prompt = fake.calls[0][1]
    for value in ("Jordan", "41", "Male"):
        assert value not in cv_prompt
    assert result.score == 100


def test_screen_does_not_anonymize_job_text() -> None:
    fake = FakeLLMClient([CV_JSON, JOB_JSON])
    screen("Go developer", "Hiring Manager Team\nGo required", fake)
    assert fake.calls[1][1] == "Hiring Manager Team\nGo required"

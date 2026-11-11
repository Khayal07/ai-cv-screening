from pathlib import Path
from typing import Any

import pytest

from app.extraction import (
    CV_SYSTEM_PROMPT,
    JOB_SYSTEM_PROMPT,
    MAX_INPUT_CHARS,
    PROMPT_VERSION,
    RETRY_NOTE,
    ExtractionError,
    extract_cv_profile,
    extract_job_requirements,
)
from app.llm.cache import CachedLLMClient
from app.llm.client import LLMResponseError
from app.models import EducationLevel
from tests.conftest import FakeLLMClient, role

VALID_CV: dict[str, Any] = {
    "skills": [{"name": "Python", "evidence": True}, {"name": "FastAPI", "evidence": False}],
    "roles": [role(5, "Python")],
    "education_level": "bachelor",
    "languages": ["English"],
}
VALID_JOB: dict[str, Any] = {
    "required_skills": ["Python"],
    "nice_to_have_skills": ["Docker"],
    "min_years": 3,
    "education_level": "bachelor",
    "languages": ["English"],
}


def test_extracts_cv_profile() -> None:
    profile = extract_cv_profile("cv text", FakeLLMClient([VALID_CV]))
    assert [(s.name, s.evidence) for s in profile.skills] == [("Python", True), ("FastAPI", False)]
    assert profile.education_level is EducationLevel.BACHELOR


def test_extracts_job_requirements() -> None:
    job = extract_job_requirements("job text", FakeLLMClient([VALID_JOB]))
    assert job.required_skills == ["Python"]
    assert job.min_years == 3


def test_extra_keys_are_ignored() -> None:
    profile = extract_cv_profile("cv", FakeLLMClient([{**VALID_CV, "name": "X"}]))
    assert "name" not in profile.model_dump()


def test_retry_succeeds_after_malformed_json() -> None:
    fake = FakeLLMClient([LLMResponseError("bad json"), VALID_CV])
    profile = extract_cv_profile("cv", fake)
    assert profile.roles[0].skills == ["Python"]
    assert len(fake.calls) == 2
    assert fake.calls[1][1].endswith(RETRY_NOTE)


def test_retry_succeeds_after_validation_error() -> None:
    fake = FakeLLMClient([{**VALID_CV, "roles": "not a list"}, VALID_CV])
    assert len(extract_cv_profile("cv", fake).roles) == 1
    assert len(fake.calls) == 2


def test_raises_after_retry_fails() -> None:
    fake = FakeLLMClient([LLMResponseError("bad"), {"education_level": "wizard"}])
    with pytest.raises(ExtractionError):
        extract_job_requirements("job", fake)
    assert len(fake.calls) == 2


def test_retry_uses_new_cache_key_so_bad_cached_output_is_bypassed(tmp_path: Path) -> None:
    fake = FakeLLMClient([{"skills": "not a list"}, VALID_CV])
    cached = CachedLLMClient(fake, tmp_path, key_salt=PROMPT_VERSION)
    assert [s.name for s in extract_cv_profile("cv", cached).skills] == ["Python", "FastAPI"]
    assert [s.name for s in extract_cv_profile("cv", cached).skills] == ["Python", "FastAPI"]
    assert len(fake.calls) == 2


def test_input_is_truncated() -> None:
    fake = FakeLLMClient([VALID_CV])
    extract_cv_profile("x" * (MAX_INPUT_CHARS + 500), fake)
    assert len(fake.calls[0][1]) == MAX_INPUT_CHARS


def test_cv_prompt_excludes_personal_data() -> None:
    for word in ("name", "age", "gender", "photo"):
        assert word in CV_SYSTEM_PROMPT


def test_prompt_version_is_set() -> None:
    assert PROMPT_VERSION == "v4"


@pytest.mark.parametrize("prompt", [CV_SYSTEM_PROMPT, JOB_SYSTEM_PROMPT])
def test_prompts_ask_for_single_concept_skills(prompt: str) -> None:
    assert "single-concept" in prompt
    assert "Split compound phrases" in prompt
    assert '"SQL query optimisation" -> ["SQL"]' in prompt
    assert '"REST API design" -> ["REST"]' in prompt


def test_prompt_version_changes_cache_key(tmp_path: Path) -> None:
    current = CachedLLMClient(FakeLLMClient(), tmp_path, key_salt=PROMPT_VERSION)
    previous = CachedLLMClient(FakeLLMClient(), tmp_path, key_salt="v1")
    assert current.cache_key(CV_SYSTEM_PROMPT, "cv") != previous.cache_key(CV_SYSTEM_PROMPT, "cv")


def test_cv_prompt_asks_for_skill_evidence() -> None:
    assert '"evidence"' in CV_SYSTEM_PROMPT
    assert '{"name": "FastAPI", "evidence": true}' in CV_SYSTEM_PROMPT
    assert '{"name": "Redis", "evidence": false}' in CV_SYSTEM_PROMPT
    assert '"evidence"' not in JOB_SYSTEM_PROMPT


def test_cv_prompt_says_projects_are_not_roles() -> None:
    assert "Projects (university, coursework, personal or open-source)" in CV_SYSTEM_PROMPT
    assert "never roles" in CV_SYSTEM_PROMPT


def test_cv_prompt_excludes_unfinished_degrees() -> None:
    assert "still in progress or expected in the future is not completed" in CV_SYSTEM_PROMPT


def test_cv_prompt_shows_numeric_dates_only() -> None:
    assert '"2019-03"' in CV_SYSTEM_PROMPT
    assert '"YYYY-MM"' not in CV_SYSTEM_PROMPT
    assert "Never output placeholder letters" in CV_SYSTEM_PROMPT


def test_job_prompt_keeps_alternatives_together() -> None:
    assert '"Tableau or Power BI"' in JOB_SYSTEM_PROMPT
    assert '"FastAPI (or similar framework)" -> "FastAPI"' in JOB_SYSTEM_PROMPT

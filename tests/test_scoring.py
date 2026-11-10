from typing import Any

import pytest

from app.models import DISCLAIMER, CVProfile, EducationLevel, JobRequirements
from app.scoring import WEIGHTS, normalize_skill, score_match, split_matches


def make_cv(**overrides: Any) -> CVProfile:
    data: dict[str, Any] = {
        "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
        "years_experience": 5,
        "education_level": "bachelor",
        "languages": ["English"],
    }
    return CVProfile(**{**data, **overrides})


def make_job(**overrides: Any) -> JobRequirements:
    data: dict[str, Any] = {
        "required_skills": ["Python", "FastAPI", "PostgreSQL", "Kubernetes"],
        "nice_to_have_skills": ["Docker", "Redis"],
        "min_years": 4,
        "education_level": "bachelor",
        "languages": ["English"],
    }
    return JobRequirements(**{**data, **overrides})


def component(result: Any, name: str) -> Any:
    return next(c for c in result.breakdown if c.name == name)


def test_weights_sum_to_100() -> None:
    assert sum(WEIGHTS.values()) == 100


def test_perfect_match_scores_100() -> None:
    job = make_job(required_skills=["Python"], nice_to_have_skills=["Docker"])
    assert score_match(make_cv(), job).score == 100


def test_no_match_scores_zero() -> None:
    cv = make_cv(skills=["Excel"], years_experience=0, education_level="none", languages=["French"])
    result = score_match(cv, make_job(education_level="master"))
    assert result.score == 0
    assert result.matched_skills == []


def test_partial_match_breakdown() -> None:
    result = score_match(make_cv(), make_job())
    # required 3/4*50=37.5, nice 1/2*15=7.5, experience 20, education 10, languages 5
    assert result.score == 80
    assert isinstance(result.score, int)
    assert result.missing_skills == ["Kubernetes"]
    assert result.matched_skills == ["Python", "FastAPI", "PostgreSQL", "Docker"]


def test_components_stay_float_and_round_once() -> None:
    job = make_job(required_skills=["Python", "Go", "Rust"], nice_to_have_skills=["Docker", "A", "B"])
    result = score_match(make_cv(), job)
    points = [c.points for c in result.breakdown]
    assert points[0] == pytest.approx(50 / 3)
    assert points[1] == pytest.approx(5.0)
    # 16.67 + 5 + 20 + 10 + 5 = 56.67 -> 57
    assert result.score == round(sum(points)) == 57


@pytest.mark.parametrize(
    ("cv_skill", "job_skill"),
    [("JS", "JavaScript"), ("postgres", "PostgreSQL"), ("K8s", "Kubernetes"), ("Node", "Node.js")],
)
def test_aliases_match(cv_skill: str, job_skill: str) -> None:
    matched, missing = split_matches([job_skill], [cv_skill])
    assert matched == [job_skill]
    assert missing == []


def test_normalize_collapses_case_and_spaces() -> None:
    assert normalize_skill("  Machine   Learning ") == "machine learning"


@pytest.mark.parametrize(("years", "expected_ratio"), [(2, 0.5), (4, 1.0), (10, 1.0)])
def test_experience_gap(years: float, expected_ratio: float) -> None:
    result = score_match(make_cv(years_experience=years), make_job())
    assert component(result, "experience").ratio == pytest.approx(expected_ratio)


def test_no_minimum_years_gives_full_experience() -> None:
    result = score_match(make_cv(years_experience=0), make_job(min_years=0))
    assert component(result, "experience").ratio == 1.0


def test_education_below_requirement_is_partial() -> None:
    result = score_match(make_cv(education_level="bachelor"), make_job(education_level="master"))
    education = component(result, "education")
    assert 0 < education.ratio < 1
    assert "below" in education.detail


def test_empty_requirements_give_full_points() -> None:
    job = JobRequirements(education_level=EducationLevel.NONE)
    assert score_match(make_cv(), job).score == 100


def test_reasons_explain_each_component() -> None:
    result = score_match(make_cv(), make_job())
    assert len(result.reasons) == len(WEIGHTS)
    assert any("missing: Kubernetes" in reason for reason in result.reasons)


def test_result_includes_disclaimer() -> None:
    assert score_match(make_cv(), make_job()).disclaimer == DISCLAIMER

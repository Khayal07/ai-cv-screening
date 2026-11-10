from typing import Any

import pytest

from app.models import DISCLAIMER, CVProfile, EducationLevel, JobRequirements
from app.scoring import (
    IMPLIES,
    SKILL_ALIASES,
    WEIGHTS,
    SkillMatch,
    implies,
    match_skills,
    normalize_skill,
    score_match,
    skill_tokens,
    skills_match,
    split_matches,
)


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


# ---------- Token matching ----------

NO_MATCH_PAIRS = [
    ("Java", "JavaScript"),
    ("Go", "MongoDB"),
    ("C", "C++"),
    ("C", "C#"),
    ("C++", "C#"),
    ("C", "Objective-C"),
    ("R", "React"),
    ("Script", "JavaScript"),
    ("PostgreSQL", "SQL"),
    ("Scala", "Scalability"),
    ("Rust", "Trust and safety"),
]


@pytest.mark.parametrize(("left", "right"), NO_MATCH_PAIRS)
def test_no_false_matches_in_either_direction(left: str, right: str) -> None:
    assert not skills_match(left, right)
    assert not skills_match(right, left)
    assert split_matches([left], [right]) == ([], [left])


MATCH_PAIRS = [  # (job skill, CV skill)
    ("SQL", "SQL query optimisation"),
    ("REST", "REST API design"),
    ("Data analysis", "Data analysis and reporting"),
    ("Kubernetes", "K8s"),
    ("JavaScript", "JS"),
    ("Node.js", "Node"),
    ("C++", "C++ 17"),
    ("Machine Learning", "ML"),
    ("CI/CD", "CI CD"),
    ("Python", "  python "),
]


@pytest.mark.parametrize(("wanted", "have"), MATCH_PAIRS)
def test_token_sequence_matches(wanted: str, have: str) -> None:
    assert skills_match(wanted, have)
    assert split_matches([wanted], [have]) == ([wanted], [])


SHORT_CV_SKILL_PAIRS = [  # (longer job skill, shorter CV skill): never a match
    ("Data analysis", "Data"),
    ("SQL query optimisation", "SQL"),
    ("REST API design", "REST"),
    ("SQL Server", "SQL"),
    ("Machine learning engineering", "Machine learning"),
    ("C++ 17", "C++"),
]


@pytest.mark.parametrize(("wanted", "have"), SHORT_CV_SKILL_PAIRS)
def test_short_cv_skill_does_not_satisfy_longer_job_skill(wanted: str, have: str) -> None:
    assert not skills_match(wanted, have)
    assert split_matches([wanted], [have]) == ([], [wanted])
    # The opposite direction is the one that matches.
    assert skills_match(have, wanted)


def test_tokens_must_be_contiguous() -> None:
    assert not skills_match("SQL design", "SQL query design")


def test_tokens_keep_symbols_and_hyphens() -> None:
    assert skill_tokens("C++ / C# and Objective-C") == ("c++", "c#", "and", "objective-c")


def test_postgresql_matches_sql_only_through_an_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not skills_match("SQL", "PostgreSQL")
    monkeypatch.setitem(SKILL_ALIASES, "postgresql", "sql")
    assert skills_match("SQL", "PostgreSQL")


def test_exact_match_preferred_over_token_match() -> None:
    matches, _ = match_skills(["SQL"], ["SQL query optimisation", "SQL"])
    assert matches == [SkillMatch(wanted="SQL", found="SQL")]


def test_reason_names_the_cv_skill_behind_each_loose_match() -> None:
    cv = make_cv(skills=["Python", "SQL query optimisation", "K8s"])
    job = make_job(required_skills=["Python", "SQL", "Kubernetes", "Rust"])
    reason = score_match(cv, job).reasons[0]
    assert reason == (
        "Required skills: 3 of 4 matched "
        "(SQL via SQL query optimisation; Kubernetes via K8s; missing: Rust)."
    )


def test_exact_matches_need_no_via_note() -> None:
    reason = score_match(make_cv(), make_job(required_skills=["python", "FastAPI"])).reasons[0]
    assert reason == "Required skills: 2 of 2 matched."


def test_loose_match_raises_score() -> None:
    job = make_job(required_skills=["Python", "SQL"], nice_to_have_skills=[])
    strict = score_match(make_cv(skills=["Python"]), job).score
    loose_cv = make_cv(skills=["Python", "SQL query tuning"])
    assert score_match(loose_cv, job).score > strict
    assert score_match(loose_cv, job).missing_skills == []


# ---------- One-way implications ----------

IMPLIED_PAIRS = [  # (job skill, CV skill that implies it)
    ("SQL", "PostgreSQL"),
    ("SQL", "Postgres"),
    ("SQL", "MySQL"),
    ("SQL", "SQLite"),
    ("SQL", "SQL Server"),
    ("SQL", "MSSQL"),
    ("Python", "FastAPI"),
    ("Python", "Django"),
    ("Python", "Flask"),
    ("JavaScript", "React"),
    ("JavaScript", "React.js"),
    ("JavaScript", "Vue.js"),
    ("JavaScript", "Angular"),
    ("JavaScript", "TypeScript"),
    ("React", "Next.js"),
    ("Node.js", "Express"),
]


@pytest.mark.parametrize(("job_skill", "cv_skill"), IMPLIED_PAIRS)
def test_cv_skill_implies_job_skill(job_skill: str, cv_skill: str) -> None:
    matches, missing = match_skills([job_skill], [cv_skill])
    assert matches == [SkillMatch(wanted=job_skill, found=cv_skill)]
    assert missing == []


@pytest.mark.parametrize(("job_skill", "cv_skill"), IMPLIED_PAIRS)
def test_implication_is_one_way(job_skill: str, cv_skill: str) -> None:
    # Swap roles: the general CV skill never satisfies the specific job skill.
    matches, missing = match_skills([cv_skill], [job_skill])
    assert matches == []
    assert missing == [cv_skill]


@pytest.mark.parametrize(
    ("job_skill", "cv_skill"),
    [
        ("JavaScript", "Next.js"),  # next.js -> react -> javascript would be two hops
        ("Python", "SQL"),
        ("SQL", "Python"),
        ("JavaScript", "Java"),
    ],
)
def test_no_transitive_or_unrelated_implications(job_skill: str, cv_skill: str) -> None:
    assert split_matches([job_skill], [cv_skill]) == ([], [job_skill])


def test_implication_values_are_never_expanded() -> None:
    # Every implied skill is checked against the CV skill's own entry only.
    for cv_skill, implied in IMPLIES.items():
        for job_skill in implied:
            assert implies(cv_skill, job_skill)
            for second_hop in IMPLIES.get(job_skill, frozenset()):
                if second_hop not in implied:
                    assert not implies(cv_skill, second_hop)


def test_reason_shows_implied_match() -> None:
    cv = make_cv(skills=["PostgreSQL", "FastAPI"])
    job = make_job(required_skills=["SQL", "Python", "PostgreSQL"])
    assert score_match(cv, job).reasons[0] == (
        "Required skills: 3 of 3 matched (SQL via PostgreSQL; Python via FastAPI)."
    )


def test_exact_match_preferred_over_implication() -> None:
    matches, _ = match_skills(["Python"], ["FastAPI", "Python"])
    assert matches == [SkillMatch(wanted="Python", found="Python")]

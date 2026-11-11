from datetime import date
from typing import Any

import pytest

from app.models import DISCLAIMER, CVProfile, CVSkill, EducationLevel, JobRequirements
from tests.conftest import listed, role, used
from app.scoring import (
    IMPLIES,
    SKILL_ALIASES,
    WEIGHTS,
    SkillMatch,
    implies,
    match_cv_skills,
    match_skills,
    normalize_skill,
    score_match,
    skill_tokens,
    skills_match,
    split_matches,
)


def make_cv(**overrides: Any) -> CVProfile:
    """Plain skill names in tests mean skills shown in use (evidence=True).

    `years_experience=N` is shorthand for one role of N years that used the CV's skills.
    """
    if "skills" in overrides:
        overrides["skills"] = [
            {"name": s, "evidence": True} if isinstance(s, str) else s for s in overrides["skills"]
        ]
    data: dict[str, Any] = {
        "skills": used("Python", "FastAPI", "Docker", "PostgreSQL"),
        "roles": [role(5, "Python", "FastAPI", "Docker", "PostgreSQL")],
        "education_level": "bachelor",
        "languages": ["English"],
    }
    data.update(overrides)
    if "years_experience" in data:
        years = data.pop("years_experience")
        names = [skill["name"] for skill in data["skills"]]
        data["roles"] = [role(years, *names)] if years else []
    return CVProfile(**data)


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


# ---------- Evidence weighting ----------


def test_listed_only_skill_gets_half_credit() -> None:
    job = make_job(required_skills=["Python", "Docker"], nice_to_have_skills=[])
    cv = make_cv(skills=used("Python") + listed("Docker"))
    required = score_match(cv, job).breakdown[0]
    assert required.ratio == pytest.approx(0.75)
    assert required.points == pytest.approx(37.5)


def test_all_listed_scores_half_of_all_used() -> None:
    job = make_job(required_skills=["Python", "Docker"], nice_to_have_skills=["Redis"])
    full = score_match(make_cv(skills=used("Python", "Docker", "Redis")), job).breakdown
    half = score_match(make_cv(skills=listed("Python", "Docker", "Redis")), job).breakdown
    assert half[0].points == pytest.approx(full[0].points / 2)
    assert half[1].points == pytest.approx(full[1].points / 2)


def test_evidenced_implied_skill_beats_listed_exact_skill() -> None:
    cv_skills = [CVSkill(name="Python"), CVSkill(name="FastAPI", evidence=True)]
    matches, missing = match_cv_skills(["Python"], cv_skills)
    assert matches == [SkillMatch(wanted="Python", found="FastAPI", evidence=True)]
    assert missing == []


def test_listed_exact_skill_used_when_nothing_has_evidence() -> None:
    matches, _ = match_cv_skills(["Python"], [CVSkill(name="Python"), CVSkill(name="Go")])
    assert matches == [SkillMatch(wanted="Python", found="Python", evidence=False)]


def test_reason_names_skills_listed_without_evidence() -> None:
    cv = make_cv(skills=used("PostgreSQL", "Python") + listed("Redis"))
    job = make_job(required_skills=["SQL", "Python", "Redis", "Rust"])
    assert score_match(cv, job).reasons[0] == (
        "Required skills: 3 of 4 matched "
        "(SQL via PostgreSQL; listed without evidence: Redis; missing: Rust)."
    )


def test_listed_skills_still_count_as_matched_not_missing() -> None:
    result = score_match(make_cv(skills=listed("Python")), make_job(required_skills=["Python"]))
    assert result.matched_skills[0] == "Python"
    assert "Python" not in result.missing_skills


def test_plain_skill_names_from_model_have_no_evidence() -> None:
    profile = CVProfile(skills=["Python", {"name": "Go", "evidence": True}])
    assert [(s.name, s.evidence) for s in profile.skills] == [("Python", False), ("Go", True)]


# --- Experience from dated roles ---

TODAY = date(2026, 11, 10)


def experience_detail(cv: CVProfile, job: JobRequirements) -> str:
    return component(score_match(cv, job, today=TODAY), "experience").detail


def test_experience_counts_only_roles_with_job_skills() -> None:
    cv = make_cv(roles=[role(3.5, "Python", "FastAPI"), role(1.5, "Teaching", end="2020-07")])
    result = score_match(cv, make_job(min_years=4), today=TODAY)
    experience = component(result, "experience")
    assert experience.detail == "Experience: 3.5 relevant years of 5.0 total (required 4)."
    assert experience.ratio == pytest.approx(3.5 / 4)


def test_teacher_role_is_not_relevant_for_backend_job() -> None:
    cv = make_cv(roles=[role(10, "Teaching", "Classroom management")])
    detail = experience_detail(cv, make_job(min_years=4))
    assert detail == "Experience: 0.0 relevant years of 10.0 total (required 4)."


def test_role_counts_through_implication() -> None:
    cv = make_cv(roles=[role(4, "Django")])
    detail = experience_detail(cv, make_job(required_skills=["Python"], nice_to_have_skills=[]))
    assert detail.startswith("Experience: 4.0 relevant years")


def test_role_counts_through_token_match() -> None:
    cv = make_cv(roles=[role(2, "SQL query tuning")])
    detail = experience_detail(cv, make_job(required_skills=["SQL"], nice_to_have_skills=[]))
    assert detail.startswith("Experience: 2.0 relevant years")


def test_role_does_not_count_through_substring() -> None:
    cv = make_cv(roles=[role(2, "JavaScript")])
    detail = experience_detail(cv, make_job(required_skills=["Java"], nice_to_have_skills=[]))
    assert detail.startswith("Experience: 0.0 relevant years of 2.0 total")


def test_parallel_relevant_roles_are_not_double_counted() -> None:
    cv = make_cv(roles=[role(3, "Python"), role(2, "FastAPI")])
    assert experience_detail(cv, make_job()).startswith("Experience: 3.0 relevant years of 3.0")


def test_no_minimum_years_reason() -> None:
    detail = experience_detail(make_cv(roles=[role(2, "Python")]), make_job(min_years=0))
    assert detail == "Experience: 2.0 relevant years of 2.0 total (no minimum stated)."


def test_role_without_skills_is_not_counted_and_warns() -> None:
    cv = make_cv(roles=[role(2, "Python"), role(3, end="2021-01")])
    result = score_match(cv, make_job(), today=TODAY)
    assert component(result, "experience").detail.startswith(
        "Experience: 2.0 relevant years of 5.0 total"
    )
    assert result.warnings == [
        "1 role(s) have no listed skills and were not counted as relevant experience."
    ]


def test_unreadable_dates_are_ignored_and_warn() -> None:
    cv = make_cv(roles=[role(2, "Python"), {"start": "a while ago", "end": "present", "skills": ["Python"]}])
    result = score_match(cv, make_job(), today=TODAY)
    assert component(result, "experience").detail.startswith("Experience: 2.0 relevant years of 2.0")
    assert result.warnings == ["1 role(s) have unreadable dates and were ignored."]


def test_present_role_runs_until_today() -> None:
    cv = make_cv(roles=[{"start": "2022-11", "end": "present", "skills": ["Python"]}])
    assert experience_detail(cv, make_job()).startswith("Experience: 4.0 relevant years")


def test_job_without_skills_counts_every_dated_role() -> None:
    cv = make_cv(roles=[role(3, "Teaching"), role(1, end="2020-01")])
    result = score_match(cv, JobRequirements(min_years=2), today=TODAY)
    assert component(result, "experience").ratio == 1.0
    assert result.warnings == []


def test_no_roles_gives_zero_experience_points() -> None:
    result = score_match(make_cv(roles=[]), make_job(min_years=3), today=TODAY)
    experience = component(result, "experience")
    assert experience.points == 0
    assert experience.detail == "Experience: 0.0 relevant years of 0.0 total (required 3)."


# ---------- spoken languages named as job skills ----------


def test_language_skill_is_matched_by_cv_languages() -> None:
    cv = make_cv(languages=["English", "Spanish"])
    job = make_job(nice_to_have_skills=["Docker", "Spanish"])
    nice = component(score_match(cv, job), "nice_to_have_skills")
    assert nice.ratio == 1.0
    assert nice.detail == "Nice-to-have skills: 2 of 2 matched."


def test_language_with_level_matches_and_reason_names_it() -> None:
    matches, missing = match_cv_skills(["Spanish"], [], ["Spanish (intermediate)"])
    assert matches == [SkillMatch(wanted="Spanish", found="Spanish (intermediate)", evidence=True)]
    assert missing == []


def test_evidenced_skill_is_preferred_over_a_language() -> None:
    cv_skills = [CVSkill(name="Spanish translation", evidence=True)]
    matches, _ = match_cv_skills(["Spanish"], cv_skills, ["Spanish"])
    assert matches[0].found == "Spanish translation"


def test_language_beats_a_listed_only_skill() -> None:
    matches, _ = match_cv_skills(["Spanish"], [CVSkill(name="Spanish")], ["Spanish"])
    assert matches[0].evidence is True


def test_languages_do_not_satisfy_technical_skills() -> None:
    matches, missing = match_cv_skills(["Go", "R"], [], ["English", "Romanian"])
    assert matches == []
    assert missing == ["Go", "R"]

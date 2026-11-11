"""The evaluation labels must be complete and consistent before any model output exists."""

import re
from typing import Any

import pytest
from pydantic import ValidationError

from app.evaluation import (
    CATEGORY_BANDS,
    CVS_DIR,
    JOBS_DIR,
    EvalCase,
    EvalLabels,
    Outcome,
    load_labels,
)

LABELS = load_labels()
EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[a-z]{2,}", re.IGNORECASE)


def scored(**overrides: Any) -> dict[str, Any]:
    case = {"cv": "a.txt", "job": "jd.txt", "category": "strong", "outcome": "scored",
            "score": [80, 100]}
    return case | overrides


def test_labels_cover_24_to_30_cv_files() -> None:
    files = {path.name for path in CVS_DIR.glob("*.txt")}
    assert 24 <= len(files) <= 30
    assert {case.cv for case in LABELS.cases} == files


def test_every_referenced_file_exists() -> None:
    for case in LABELS.cases:
        assert (CVS_DIR / case.cv).is_file(), case.cv
        assert (JOBS_DIR / case.job).is_file(), case.job


def test_both_jobs_have_every_ranked_category() -> None:
    ranked = {category for rule in LABELS.rankings for category in rule.above + rule.below}
    for job in {case.job for case in LABELS.cases}:
        present = {case.category for case in LABELS.cases if case.job == job}
        assert ranked <= present, job


def test_rejected_and_warning_cases_exist() -> None:
    assert sum(case.outcome is Outcome.REJECTED for case in LABELS.cases) >= 4
    assert any(case.warnings for case in LABELS.cases)


def test_bands_do_not_exceed_score_limits() -> None:
    for band in filter(None, CATEGORY_BANDS.values()):
        assert 0 <= band[0] <= band[1] <= 100


def test_eval_cvs_use_only_example_email_domains() -> None:
    for path in CVS_DIR.glob("*.txt"):
        for match in EMAIL.finditer(path.read_text(encoding="utf-8")):
            assert match.group(0).lower().endswith(("@example.com", "@example.org")), path.name


def test_case_id_combines_cv_and_job() -> None:
    assert EvalCase.model_validate(scored()).id == "a@jd"


@pytest.mark.parametrize(
    "overrides",
    [
        {"category": "unknown"},
        {"score": None},
        {"score": [90, 80]},
        {"score": [70, 100]},
        {"category": "pasted_job"},
        {"outcome": "rejected_422"},
        {"outcome": "rejected_422", "category": "non_cv"},
        {"warnings": ["made_up"]},
    ],
)
def test_invalid_cases_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalCase.model_validate(scored(**overrides))


def test_rejected_case_without_score_is_valid() -> None:
    case = {"cv": "x.txt", "job": "jd.txt", "category": "non_cv", "outcome": "rejected_422"}
    assert EvalCase.model_validate(case).outcome is Outcome.REJECTED


def test_own_range_categories_accept_any_valid_range() -> None:
    case = scored(category="cross_job", score=[10, 60])
    assert EvalCase.model_validate(case).score == (10, 60)


def test_duplicate_cases_are_refused() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        EvalLabels.model_validate(
            {"today": "2026-11-01", "rankings": [], "cases": [scored(), scored()]}
        )


def test_ranking_rules_need_scored_categories() -> None:
    with pytest.raises(ValidationError):
        EvalLabels.model_validate(
            {"today": "2026-11-01", "rankings": [{"above": ["strong"], "below": ["non_cv"]}],
             "cases": []}
        )

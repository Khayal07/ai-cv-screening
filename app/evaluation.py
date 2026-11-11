"""Evaluation set: labeled synthetic CVs, expected outcomes and ranking rules."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_DIR / "data" / "eval"
CVS_DIR = EVAL_DIR / "cvs"
JOBS_DIR = PROJECT_DIR / "data" / "synthetic"
LABELS_PATH = EVAL_DIR / "labels.json"


class Outcome(StrEnum):
    SCORED = "scored"
    REJECTED = "rejected_422"


class WarningKind(StrEnum):
    OVERLAP = "overlap"
    NO_SKILL_ROLES = "no_skill_roles"
    UNREADABLE_DATES = "unreadable_dates"


# Default score band per scored category. A case may narrow its band, never widen it.
# None means the case sets its own range (with a note saying why).
CATEGORY_BANDS: dict[str, tuple[int, int] | None] = {
    "strong": (80, 100),
    "partial": (50, 79),
    "weak": (0, 45),
    "junior_projects": (25, 65),
    "career_changer": (15, 55),
    "stuffed": (25, 54),
    "copied_requirements": None,
    "cross_job": None,
}
REJECTED_CATEGORIES = frozenset({"pasted_job", "very_short", "non_cv"})
CATEGORIES = frozenset(CATEGORY_BANDS) | REJECTED_CATEGORIES


class EvalCase(BaseModel):
    """One CV scored against one job, with the expected result."""

    model_config = ConfigDict(extra="forbid")

    cv: str
    job: str
    category: str
    outcome: Outcome
    score: tuple[int, int] | None = None
    warnings: list[WarningKind] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    note: str = ""

    @property
    def id(self) -> str:
        return f"{Path(self.cv).stem}@{Path(self.job).stem}"

    @model_validator(mode="after")
    def check_expectation(self) -> Self:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category: {self.category}")
        if self.outcome is Outcome.REJECTED:
            _check_rejected(self)
        else:
            _check_scored(self)
        return self


def _check_rejected(case: EvalCase) -> None:
    if case.category not in REJECTED_CATEGORIES:
        raise ValueError(f"category {case.category} cannot be rejected")
    if case.score is not None or case.warnings:
        raise ValueError("a rejected case has no score range or warnings")


def _check_scored(case: EvalCase) -> None:
    if case.category in REJECTED_CATEGORIES:
        raise ValueError(f"category {case.category} must be rejected")
    if case.score is None:
        raise ValueError("a scored case needs a score range")
    low, high = case.score
    if not 0 <= low <= high <= 100:
        raise ValueError(f"invalid score range: {case.score}")
    band = CATEGORY_BANDS[case.category]
    if band is not None and not band[0] <= low <= high <= band[1]:
        raise ValueError(f"range {case.score} is outside the {case.category} band {band}")


class RankingRule(BaseModel):
    """Per job: every case in `above` must outscore every case in `below`."""

    model_config = ConfigDict(extra="forbid")

    above: list[str]
    below: list[str]

    @model_validator(mode="after")
    def check_categories(self) -> Self:
        unknown = set(self.above + self.below) - set(CATEGORY_BANDS)
        if unknown:
            raise ValueError(f"unknown or unscored categories: {sorted(unknown)}")
        return self


class EvalLabels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    today: date
    rankings: list[RankingRule]
    cases: list[EvalCase]

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        ids = [case.id for case in self.cases]
        duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate cases: {duplicates}")
        return self


def load_labels(path: Path = LABELS_PATH) -> EvalLabels:
    return EvalLabels.model_validate_json(path.read_text(encoding="utf-8"))

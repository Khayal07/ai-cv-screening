"""Evaluation set: labeled synthetic CVs, expected outcomes and ranking rules."""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.extraction import ExtractionError
from app.llm.client import LLMClient
from app.parsing.cv_check import NotACVError
from app.service import screen

PROJECT_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_DIR / "data" / "eval"
CVS_DIR = EVAL_DIR / "cvs"
JOBS_DIR = PROJECT_DIR / "data" / "synthetic"
LABELS_PATH = EVAL_DIR / "labels.json"
EXTRACTIONS_DIR = EVAL_DIR / "extractions"


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
    "stuffed": (25, 49),
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


def read_text(path: Path) -> str:
    """Read an eval file the way the API sees it (surrounding whitespace stripped)."""
    return path.read_text(encoding="utf-8").strip()


# A fragment of each warning text produced by app/parsing/overlap.py and app/scoring.py.
WARNING_MARKERS: dict[WarningKind, str] = {
    WarningKind.OVERLAP: "match the job description",
    WarningKind.NO_SKILL_ROLES: "have no listed skills",
    WarningKind.UNREADABLE_DATES: "have unreadable dates",
}


def warning_kinds(warnings: list[str]) -> tuple[WarningKind, ...]:
    return tuple(
        kind for kind, marker in WARNING_MARKERS.items() if any(marker in text for text in warnings)
    )


@dataclass(frozen=True)
class CaseResult:
    """The actual result of one case. `outcome` is None when extraction failed."""

    case: EvalCase
    outcome: Outcome | None
    score: int | None
    warnings: tuple[WarningKind, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def check_case(
    case: EvalCase, outcome: Outcome, score: int | None, warnings: tuple[WarningKind, ...]
) -> tuple[str, ...]:
    """Every way the actual result differs from the label, as readable lines."""
    if outcome is not case.outcome:
        got = f"{outcome.value} (score {score})" if score is not None else outcome.value
        return (f"expected {case.outcome.value}, got {got}",)
    failures: list[str] = []
    if case.score is not None and score is not None and not case.score[0] <= score <= case.score[1]:
        failures.append(f"score {score} is outside {case.score[0]}-{case.score[1]}")
    failures += [f"missing warning: {kind.value}" for kind in case.warnings if kind not in warnings]
    if WarningKind.OVERLAP in warnings and WarningKind.OVERLAP not in case.warnings:
        failures.append("unexpected warning: overlap")
    return tuple(failures)


def judge(
    case: EvalCase,
    outcome: Outcome,
    score: int | None = None,
    warnings: tuple[WarningKind, ...] = (),
) -> CaseResult:
    return CaseResult(case, outcome, score, warnings, check_case(case, outcome, score, warnings))


def run_case(
    case: EvalCase,
    llm: LLMClient,
    today: date,
    cvs_dir: Path = CVS_DIR,
    jobs_dir: Path = JOBS_DIR,
) -> CaseResult:
    cv_text, job_text = read_text(cvs_dir / case.cv), read_text(jobs_dir / case.job)
    try:
        result = screen(cv_text, job_text, llm, today)
    except NotACVError:
        return judge(case, Outcome.REJECTED)
    except ExtractionError:
        return CaseResult(case, None, None, (), ("model output invalid after the retry",))
    return judge(case, Outcome.SCORED, result.score, warning_kinds(result.warnings))


@dataclass(frozen=True)
class RankingCheck:
    job: str
    rule: RankingRule
    lowest_above: int | None
    highest_below: int | None

    @property
    def passed(self) -> bool:
        if self.lowest_above is None or self.highest_below is None:
            return False
        return self.lowest_above > self.highest_below

    def describe(self) -> str:
        above, below = "+".join(self.rule.above), "+".join(self.rule.below)
        return (
            f"{Path(self.job).stem}: min({above}) {_or_dash(self.lowest_above)} > "
            f"max({below}) {_or_dash(self.highest_below)}"
        )


def _or_dash(value: int | None) -> str:
    return "-" if value is None else str(value)


def check_rankings(rules: list[RankingRule], results: list[CaseResult]) -> list[RankingCheck]:
    jobs = sorted({result.case.job for result in results})
    return [
        RankingCheck(
            job,
            rule,
            min(_scores(results, job, rule.above), default=None),
            max(_scores(results, job, rule.below), default=None),
        )
        for job in jobs
        for rule in rules
    ]


def _scores(results: list[CaseResult], job: str, categories: list[str]) -> list[int]:
    return [
        result.score
        for result in results
        if result.case.job == job and result.case.category in categories and result.score is not None
    ]


TABLE_HEADER = ("Case", "Category", "Expected", "Actual", "Result")


def expected_text(case: EvalCase) -> str:
    base = "422" if case.score is None else f"{case.score[0]}-{case.score[1]}"
    return " ".join([base, *(f"+{kind.value}" for kind in case.warnings)])


def actual_text(result: CaseResult) -> str:
    if result.outcome is None:
        return "error"
    base = "422" if result.score is None else str(result.score)
    return " ".join([base, *(f"+{kind.value}" for kind in result.warnings)])


def table_rows(results: list[CaseResult]) -> list[tuple[str, ...]]:
    return [
        (
            result.case.id,
            result.case.category,
            expected_text(result.case),
            actual_text(result),
            "PASS" if result.passed else "FAIL",
        )
        for result in results
    ]


def format_table(results: list[CaseResult], markdown: bool = False) -> str:
    rows = [TABLE_HEADER, *table_rows(results)]
    if markdown:
        lines = [f"| {' | '.join(row)} |" for row in rows]
        lines.insert(1, f"|{'|'.join('---' for _ in TABLE_HEADER)}|")
        return "\n".join(lines)
    widths = [max(len(row[i]) for row in rows) for i in range(len(TABLE_HEADER))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows
    )


def format_summary(results: list[CaseResult], rankings: list[RankingCheck]) -> str:
    passed = sum(result.passed for result in results)
    ranked = sum(check.passed for check in rankings)
    lines = [
        f"Cases: {passed}/{len(results)} passed ({passed / max(len(results), 1):.0%})",
        f"Rankings: {ranked}/{len(rankings)} passed",
        *(f"  {'PASS' if check.passed else 'FAIL'}  {check.describe()}" for check in rankings),
    ]
    failed = [result for result in results if not result.passed]
    if failed:
        lines.append("Failed labels:")
        lines += [f"  {result.case.id}: {'; '.join(result.failures)}" for result in failed]
    return "\n".join(lines)

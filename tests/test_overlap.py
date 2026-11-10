from itertools import product
from pathlib import Path

import pytest

from app.parsing.overlap import (
    OVERLAP_WARNING_THRESHOLD,
    overlap_warning,
    text_overlap,
    word_ngrams,
)
from app.service import screen
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"
PAIRS = list(product(sorted(DATA_DIR.glob("cv_*.txt")), sorted(DATA_DIR.glob("jd_*.txt"))))
MEASURED_MAX_REAL_PAIR = 0.067


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ngrams_ignore_case_and_punctuation() -> None:
    assert word_ngrams("Python, FastAPI and Docker!") == {
        ("python", "fastapi", "and"),
        ("fastapi", "and", "docker"),
    }


def test_short_text_has_no_overlap() -> None:
    assert text_overlap("two words", "two words") == 0.0


@pytest.mark.parametrize(("cv", "job"), PAIRS, ids=lambda p: p.stem)
def test_real_synthetic_pairs_stay_well_below_threshold(cv: Path, job: Path) -> None:
    overlap = text_overlap(read(cv), read(job))
    assert overlap <= MEASURED_MAX_REAL_PAIR + 0.001
    assert overlap_warning(read(cv), read(job)) is None


def test_threshold_has_margin_over_real_pairs() -> None:
    assert OVERLAP_WARNING_THRESHOLD >= 3 * MEASURED_MAX_REAL_PAIR


def test_pasted_requirements_block_is_flagged() -> None:
    job = read(DATA_DIR / "jd_backend_engineer.txt")
    cv = read(DATA_DIR / "cv_backend_senior.txt") + "\n" + job[job.index("Requirements") :]
    warning = overlap_warning(cv, job)
    assert warning is not None
    assert "match the job description" in warning


def test_identical_text_is_full_overlap() -> None:
    job = read(DATA_DIR / "jd_backend_engineer.txt")
    assert text_overlap(job, job) == 1.0


CV_JSON = {
    "skills": used("Python"),
    "roles": [role(5, "Python")],
    "education_level": "bachelor",
    "languages": ["English"],
}
JOB_JSON = {"required_skills": ["Python"], "min_years": 3, "education_level": "bachelor"}


def test_screen_adds_warning_but_still_scores() -> None:
    job = read(DATA_DIR / "jd_backend_engineer.txt")
    cv = VALID_CV_TEXT + "\n" + job[job.index("Requirements") :]
    result = screen(cv, job, FakeLLMClient([CV_JSON, JOB_JSON]))
    assert result.score == 100
    assert len(result.warnings) == 1
    assert "copied from the posting" in result.warnings[0]


def test_screen_without_overlap_has_no_warnings() -> None:
    result = screen(VALID_CV_TEXT, "Python developer wanted.", FakeLLMClient([CV_JSON, JOB_JSON]))
    assert result.warnings == []

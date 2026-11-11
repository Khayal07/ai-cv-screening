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


# ---------- skills named only in copied lines get no credit ----------

from app.models import CVProfile  # noqa: E402
from app.parsing.overlap import (  # noqa: E402
    exclude_copied_skills,
    excluded_skills_warning,
    split_copied_lines,
)
from tests.conftest import listed  # noqa: E402

BACKEND_JOB = read(DATA_DIR / "jd_backend_engineer.txt")
PASTED_CV = VALID_CV_TEXT + "\n" + BACKEND_JOB[BACKEND_JOB.index("Requirements") :]


def names(profile: CVProfile) -> list[str]:
    return [skill.name for skill in profile.skills]


def test_pasted_requirement_lines_are_copied() -> None:
    copied, own = split_copied_lines(PASTED_CV, BACKEND_JOB)
    assert "- Docker and Kubernetes in production" in copied
    assert "- AWS" in copied and "Nice to have" in copied
    assert "- Built REST APIs with Python and FastAPI." in own


def test_line_sharing_a_few_words_is_not_copied() -> None:
    job = "Requirements\n- Strong Python and FastAPI (or similar framework)"
    copied, own = split_copied_lines("Built REST APIs with Python and FastAPI at Example.", job)
    assert copied == []
    assert own == ["Built REST APIs with Python and FastAPI at Example."]


def test_skills_named_only_in_copied_lines_are_excluded() -> None:
    profile = CVProfile(skills=used("Python", "Kubernetes", "AWS") + listed("Redis"))
    kept, excluded = exclude_copied_skills(profile, PASTED_CV, BACKEND_JOB)
    assert names(kept) == ["Python"]
    assert excluded == ["Kubernetes", "AWS", "Redis"]


def test_cv_text_is_not_changed() -> None:
    before = str(PASTED_CV)
    exclude_copied_skills(CVProfile(skills=used("AWS")), PASTED_CV, BACKEND_JOB)
    assert PASTED_CV == before


def test_skill_not_named_in_the_text_is_kept() -> None:
    profile = CVProfile(skills=used("Terraform"))
    assert exclude_copied_skills(profile, PASTED_CV, BACKEND_JOB) == (profile, [])


def test_real_cv_sharing_words_with_the_job_keeps_every_skill() -> None:
    profile = CVProfile(skills=used("Python", "FastAPI", "PostgreSQL", "Docker"))
    assert exclude_copied_skills(profile, VALID_CV_TEXT, BACKEND_JOB) == (profile, [])


def test_excluded_skills_warning_names_them() -> None:
    assert excluded_skills_warning([]) is None
    assert excluded_skills_warning(["AWS", "Redis"]) == (
        "Not counted, named only in text copied from the job: AWS, Redis."
    )


def test_screen_gives_copied_only_skills_zero_credit() -> None:
    cv_json = dict(CV_JSON, skills=used("Python", "Kubernetes"))
    job_json = dict(JOB_JSON, required_skills=["Python", "Kubernetes"])
    pasted = screen(PASTED_CV, BACKEND_JOB, FakeLLMClient([cv_json, job_json]))
    assert pasted.reasons[0] == "Required skills: 1 of 2 matched (missing: Kubernetes)."
    assert pasted.warnings[1] == "Not counted, named only in text copied from the job: Kubernetes."


def test_real_cv_sharing_a_few_words_with_the_job_keeps_full_credit() -> None:
    cv_json = dict(CV_JSON, skills=used("Python", "FastAPI"), roles=[role(5, "Python", "FastAPI")])
    job_json = dict(JOB_JSON, required_skills=["Python", "FastAPI"])
    result = screen(VALID_CV_TEXT, BACKEND_JOB, FakeLLMClient([cv_json, job_json]))
    assert result.reasons[0] == "Required skills: 2 of 2 matched."
    assert result.warnings == []
    assert result.score == 100

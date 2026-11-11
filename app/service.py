from datetime import date

from app.extraction import extract_cv_profile, extract_job_requirements
from app.llm.client import LLMClient
from app.models import MatchResult
from app.parsing.anonymize import strip_personal_data
from app.parsing.cv_check import NotACVError, looks_like_cv
from app.parsing.overlap import discount_copied_skills, overlap_warning
from app.scoring import score_match


def screen(
    cv_text: str, job_text: str, llm: LLMClient, today: date | None = None
) -> MatchResult:
    """Check the document, anonymize the CV, extract structured data, then score.

    If the CV overlaps the job text, skills named only in copied lines lose their evidence.
    `today` fixes the date used for "present" roles (the evaluation set needs this).
    """
    check = looks_like_cv(cv_text)
    if not check.is_cv:
        raise NotACVError(check.reasons)
    warning = overlap_warning(cv_text, job_text)
    cv_profile = extract_cv_profile(strip_personal_data(cv_text), llm)
    if warning is not None:
        cv_profile = discount_copied_skills(cv_profile, cv_text, job_text)
    job_requirements = extract_job_requirements(job_text, llm)
    result = score_match(cv_profile, job_requirements, today)
    return _with_warning(result, warning)


def _with_warning(result: MatchResult, warning: str | None) -> MatchResult:
    if warning is None:
        return result
    return result.model_copy(update={"warnings": [warning, *result.warnings]})

from app.extraction import extract_cv_profile, extract_job_requirements
from app.llm.client import LLMClient
from app.models import MatchResult
from app.parsing.anonymize import strip_personal_data
from app.parsing.cv_check import NotACVError, looks_like_cv
from app.scoring import score_match


def screen(cv_text: str, job_text: str, llm: LLMClient) -> MatchResult:
    """Check the document, anonymize the CV, extract structured data, then score."""
    check = looks_like_cv(cv_text)
    if not check.is_cv:
        raise NotACVError(check.reasons)
    cv_profile = extract_cv_profile(strip_personal_data(cv_text), llm)
    job_requirements = extract_job_requirements(job_text, llm)
    return score_match(cv_profile, job_requirements)

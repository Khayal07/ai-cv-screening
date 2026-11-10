"""Turn free text into structured data via the model, with validation and one retry."""

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.client import LLMClient, LLMResponseError
from app.models import CVProfile, JobRequirements

PROMPT_VERSION = "v2"
MAX_ATTEMPTS = 2
MAX_INPUT_CHARS = 12_000

EDUCATION_VALUES = "none, high_school, bachelor, master, phd"

SKILL_NAME_RULES = (
    "Every skill must be a short, single-concept name (usually one or two words), "
    "such as a language, tool, framework, platform or method. "
    "Split compound phrases into separate skills and drop activity words "
    "like design, optimisation, experience or production. Examples: "
    '"SQL query optimisation" -> ["SQL"]; '
    '"REST API design" -> ["REST"]; '
    '"Docker and Kubernetes in production" -> ["Docker", "Kubernetes"]. '
)

CV_SYSTEM_PROMPT = (
    "You extract structured professional data from a CV. "
    "Return only a JSON object with these keys: "
    '"skills" (list of skill names), '
    '"years_experience" (number, total years of professional work), '
    f'"education_level" (highest completed, one of: {EDUCATION_VALUES}), '
    '"languages" (list of spoken languages). '
    f"{SKILL_NAME_RULES}"
    "Ignore and never output name, age, gender, photo, nationality or contact details."
)

JOB_SYSTEM_PROMPT = (
    "You extract structured requirements from a job description. "
    "Return only a JSON object with these keys: "
    '"required_skills" (list of skill names that are mandatory), '
    '"nice_to_have_skills" (list of skill names that are optional or preferred), '
    '"min_years" (number, minimum years of experience, 0 if not stated), '
    f'"education_level" (minimum required, one of: {EDUCATION_VALUES}), '
    '"languages" (list of required spoken languages). '
    f"{SKILL_NAME_RULES}"
    "Ignore any requirement about age, gender, appearance or nationality."
)

RETRY_NOTE = (
    "\n\nYour previous answer was invalid. Return only a JSON object "
    "with exactly the required keys and value types."
)

ModelT = TypeVar("ModelT", bound=BaseModel)


class ExtractionError(Exception):
    """Raised when the model output is still invalid after the retry."""


def extract_cv_profile(cv_text: str, llm: LLMClient) -> CVProfile:
    return _extract(llm, CV_SYSTEM_PROMPT, cv_text, CVProfile)


def extract_job_requirements(job_text: str, llm: LLMClient) -> JobRequirements:
    return _extract(llm, JOB_SYSTEM_PROMPT, job_text, JobRequirements)


def _extract(llm: LLMClient, system: str, text: str, model_cls: type[ModelT]) -> ModelT:
    user = text[:MAX_INPUT_CHARS]
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        prompt = user if attempt == 0 else user + RETRY_NOTE
        try:
            return model_cls.model_validate(llm.complete_json(system, prompt))
        except (LLMResponseError, ValidationError) as exc:
            last_error = exc
    raise ExtractionError(f"invalid model output after {MAX_ATTEMPTS} attempts") from last_error

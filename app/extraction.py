"""Turn free text into structured data via the model, with validation and one retry."""

from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.llm.client import LLMClient, LLMResponseError
from app.models import CVProfile, JobRequirements

PROMPT_VERSION = "v4"
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

SKILL_EVIDENCE_RULES = (
    'Each skill is an object {"name": string, "evidence": boolean}. '
    '"evidence" is true only when the CV shows the skill being used in a job, project or '
    "education entry, and false when it only appears in a skills list or summary. Examples: "
    '"Built REST APIs with FastAPI at Acme" -> {"name": "FastAPI", "evidence": true}; '
    '"Skills: Redis, Kafka" with no other mention -> {"name": "Redis", "evidence": false}. '
)

ROLE_RULES = (
    "Projects (university, coursework, personal or open-source), courses and education are "
    "never roles, even when they are dated; skills used in them still count as evidence. "
    'Dates are always numeric: "2019-03" or "2019", or "present" for an ongoing role. '
    'Convert other formats, for example "03/2019", "2019.03" or "Sept. 2019" -> "2019-03". '
    'Never output placeholder letters such as "YYYY" or "MM"; if a date cannot be read, '
    'use "". '
)

EDUCATION_RULES = (
    "A degree that is still in progress or expected in the future is not completed: use the "
    "highest level already finished. Example: "
    '"BSc Computer Science (expected 2027)" plus A-levels -> "high_school". '
)

CV_SYSTEM_PROMPT = (
    "You extract structured professional data from a CV. "
    "Return only a JSON object with these keys: "
    '"skills" (list of skill objects), '
    '"roles" (list of professional roles: jobs, internships, freelance work. '
    'Each role is {"start": "2019-03" or "2019", "end": "2021-07", "2021" or "present", '
    '"skills": list of skill names used in that role}. Use an empty skills list if none are '
    "named for that role), "
    f'"education_level" (highest completed, one of: {EDUCATION_VALUES}), '
    '"languages" (list of spoken languages). '
    f"{ROLE_RULES}"
    f"{EDUCATION_RULES}"
    f"{SKILL_EVIDENCE_RULES}"
    f"{SKILL_NAME_RULES}"
    "Ignore and never output name, age, gender, photo, nationality or contact details."
)

EITHER_OR_RULES = (
    "A requirement that accepts concrete alternatives is one item, with the alternatives "
    'joined by " or ". Examples: "A BI tool such as Tableau or Power BI" -> '
    '"Tableau or Power BI"; "AWS or GCP" -> "AWS or GCP". Drop vague alternatives: '
    '"FastAPI (or similar framework)" -> "FastAPI". Lists joined by "and" are still split. '
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
    f"{EITHER_OR_RULES}"
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

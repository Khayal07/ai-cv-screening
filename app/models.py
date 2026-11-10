from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EducationLevel(StrEnum):
    NONE = "none"
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


EDUCATION_RANK: dict[EducationLevel, int] = {
    level: rank for rank, level in enumerate(EducationLevel)
}


class CVSkill(BaseModel):
    """A skill plus whether the CV shows it in use (job, project, education)."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    evidence: bool = False


class Role(BaseModel):
    """A professional role: dates as "YYYY-MM", "YYYY" or "present", plus skills used in it."""

    model_config = ConfigDict(extra="ignore")

    start: str
    end: str
    skills: list[str] = Field(default_factory=list)


class CVProfile(BaseModel):
    """Professional data only. There are deliberately no personal fields."""

    model_config = ConfigDict(extra="ignore")

    skills: list[CVSkill] = Field(default_factory=list)
    roles: list[Role] = Field(default_factory=list)
    education_level: EducationLevel = EducationLevel.NONE
    languages: list[str] = Field(default_factory=list)

    @field_validator("skills", mode="before")
    @classmethod
    def plain_names_have_no_evidence(cls, value: Any) -> Any:
        """A bare skill name proves nothing, so it counts as listed only."""
        if not isinstance(value, list):
            return value
        return [{"name": item, "evidence": False} if isinstance(item, str) else item for item in value]


class JobRequirements(BaseModel):
    model_config = ConfigDict(extra="ignore")

    required_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    min_years: float = Field(default=0, ge=0, le=60)
    education_level: EducationLevel = EducationLevel.NONE
    languages: list[str] = Field(default_factory=list)


DISCLAIMER = (
    "This score is decision support only. It does not make or recommend a hiring "
    "decision. It ignores name, age, gender and photo; a human must review every candidate."
)


class ComponentScore(BaseModel):
    name: str
    weight: float
    ratio: float = Field(ge=0, le=1)
    points: float
    detail: str


class MatchResult(BaseModel):
    score: int = Field(ge=0, le=100)
    breakdown: list[ComponentScore]
    matched_skills: list[str]
    missing_skills: list[str]
    reasons: list[str]
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER

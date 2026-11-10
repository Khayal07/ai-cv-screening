from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EducationLevel(StrEnum):
    NONE = "none"
    HIGH_SCHOOL = "high_school"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"


EDUCATION_RANK: dict[EducationLevel, int] = {
    level: rank for rank, level in enumerate(EducationLevel)
}


class CVProfile(BaseModel):
    """Professional data only. There are deliberately no personal fields."""

    model_config = ConfigDict(extra="ignore")

    skills: list[str] = Field(default_factory=list)
    years_experience: float = Field(ge=0, le=60)
    education_level: EducationLevel = EducationLevel.NONE
    languages: list[str] = Field(default_factory=list)


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

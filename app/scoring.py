"""Deterministic, explainable scoring. No model calls happen here."""

import re
from dataclasses import dataclass

from app.models import (
    EDUCATION_RANK,
    ComponentScore,
    CVProfile,
    JobRequirements,
    MatchResult,
)

WEIGHTS: dict[str, float] = {
    "required_skills": 50.0,
    "nice_to_have_skills": 15.0,
    "experience": 20.0,
    "education": 10.0,
    "languages": 5.0,
}

SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "nodejs": "node.js",
    "node": "node.js",
    "reactjs": "react",
    "react.js": "react",
    "ml": "machine learning",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "ci cd": "ci/cd",
}


# A token is a run of letters/digits, optionally joined by dots or hyphens
# (node.js, objective-c, scikit-learn) and followed by + or # (c++, c#).
# Matching works on whole tokens only, never on raw substrings.
TOKEN = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*[+#]*")


@dataclass(frozen=True)
class SkillMatch:
    wanted: str
    found: str


def normalize_skill(skill: str) -> str:
    key = re.sub(r"\s+", " ", skill.strip().lower())
    return SKILL_ALIASES.get(key, key)


def skill_tokens(skill: str) -> tuple[str, ...]:
    return tuple(TOKEN.findall(normalize_skill(skill)))


def contains_sequence(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    size = len(inner)
    if size == 0 or size > len(outer):
        return False
    return any(outer[i : i + size] == inner for i in range(len(outer) - size + 1))


def skills_match(wanted: str, have: str) -> bool:
    if normalize_skill(wanted) == normalize_skill(have):
        return True
    a, b = skill_tokens(wanted), skill_tokens(have)
    return contains_sequence(a, b) or contains_sequence(b, a)


def find_match(wanted: str, have: list[str]) -> str | None:
    """Prefer an exact (normalized) match, then a token-sequence match."""
    key = normalize_skill(wanted)
    exact = next((item for item in have if normalize_skill(item) == key), None)
    return exact or next((item for item in have if skills_match(wanted, item)), None)


def match_skills(wanted: list[str], have: list[str]) -> tuple[list[SkillMatch], list[str]]:
    matches: list[SkillMatch] = []
    missing: list[str] = []
    for item in wanted:
        found = find_match(item, have)
        if found is None:
            missing.append(item)
        else:
            matches.append(SkillMatch(wanted=item, found=found))
    return matches, missing


def split_matches(wanted: list[str], have: list[str]) -> tuple[list[str], list[str]]:
    matches, missing = match_skills(wanted, have)
    return [match.wanted for match in matches], missing


def score_match(cv: CVProfile, job: JobRequirements) -> MatchResult:
    req_matched, req_missing = split_matches(job.required_skills, cv.skills)
    nice_matched, _ = split_matches(job.nice_to_have_skills, cv.skills)
    components = [
        _list_component("required_skills", "Required skills", job.required_skills, cv.skills),
        _list_component(
            "nice_to_have_skills", "Nice-to-have skills", job.nice_to_have_skills, cv.skills
        ),
        _experience_component(cv.years_experience, job.min_years),
        _education_component(cv, job),
        _list_component("languages", "Languages", job.languages, cv.languages),
    ]
    return MatchResult(
        score=_final_score(components),
        breakdown=components,
        matched_skills=req_matched + nice_matched,
        missing_skills=req_missing,
        reasons=[component.detail for component in components],
    )


def _final_score(components: list[ComponentScore]) -> int:
    total = sum(component.points for component in components)
    return max(0, min(100, round(total)))


def _component(name: str, ratio: float, detail: str) -> ComponentScore:
    weight = WEIGHTS[name]
    return ComponentScore(
        name=name, weight=weight, ratio=ratio, points=ratio * weight, detail=detail
    )


def _list_component(name: str, label: str, wanted: list[str], have: list[str]) -> ComponentScore:
    if not wanted:
        return _component(name, 1.0, f"{label}: none specified by the job.")
    matches, missing = match_skills(wanted, have)
    detail = _describe_matches(label, matches, missing, len(wanted))
    return _component(name, len(matches) / len(wanted), detail)


def _describe_matches(
    label: str, matches: list[SkillMatch], missing: list[str], total: int
) -> str:
    """Name the CV skill behind every match whose wording differs from the job's."""
    notes = [
        f"{match.wanted} via {match.found}"
        for match in matches
        if match.wanted.casefold() != match.found.casefold()
    ]
    if missing:
        notes.append(f"missing: {', '.join(missing)}")
    detail = f"{label}: {len(matches)} of {total} matched"
    return f"{detail} ({'; '.join(notes)})." if notes else f"{detail}."


def _experience_component(years: float, min_years: float) -> ComponentScore:
    if min_years <= 0:
        return _component("experience", 1.0, f"Experience: {years:g} years; no minimum stated.")
    ratio = min(years / min_years, 1.0)
    verdict = "meets" if ratio >= 1.0 else "is below"
    detail = f"Experience: {years:g} years {verdict} the {min_years:g} years required."
    return _component("experience", ratio, detail)


def _education_component(cv: CVProfile, job: JobRequirements) -> ComponentScore:
    have, need = EDUCATION_RANK[cv.education_level], EDUCATION_RANK[job.education_level]
    if need == 0 or have >= need:
        ratio, verdict = 1.0, "meets"
    else:
        ratio, verdict = have / need, "is below"
    detail = (
        f"Education: {cv.education_level.value} {verdict} "
        f"the required {job.education_level.value}."
    )
    return _component("education", ratio, detail)

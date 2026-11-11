"""Deterministic, explainable scoring. No model calls happen here."""

import re
from dataclasses import dataclass
from datetime import date

from app.experience import Interval, role_interval, total_years
from app.models import (
    EDUCATION_RANK,
    ComponentScore,
    CVProfile,
    CVSkill,
    JobRequirements,
    MatchResult,
    Role,
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
    "vue.js": "vue",
    "vuejs": "vue",
    "angularjs": "angular",
    "nextjs": "next.js",
    "mssql": "sql server",
    "microsoft sql server": "sql server",
}

# One-way implications: a CV skill (key) also satisfies these job skills.
# Keys and values are normalized names. Never applied transitively:
# "next.js" satisfies "react", but not "javascript".
IMPLIES: dict[str, frozenset[str]] = {
    "postgresql": frozenset({"sql"}),
    "mysql": frozenset({"sql"}),
    "mariadb": frozenset({"sql"}),
    "sqlite": frozenset({"sql"}),
    "sql server": frozenset({"sql"}),
    "fastapi": frozenset({"python"}),
    "django": frozenset({"python"}),
    "flask": frozenset({"python"}),
    "pandas": frozenset({"python"}),
    "react": frozenset({"javascript"}),
    "vue": frozenset({"javascript"}),
    "angular": frozenset({"javascript"}),
    "typescript": frozenset({"javascript"}),
    "next.js": frozenset({"react"}),
    "express": frozenset({"node.js"}),
    "spring boot": frozenset({"java"}),
    "ruby on rails": frozenset({"ruby"}),
    "laravel": frozenset({"php"}),
}


# A token is a run of letters/digits, optionally joined by dots or hyphens
# (node.js, objective-c, scikit-learn) and followed by + or # (c++, c#).
# Matching works on whole tokens only, never on raw substrings.
TOKEN = re.compile(r"[^\W_]+(?:[.-][^\W_]+)*[+#]*")


EVIDENCE_CREDIT = 1.0
LISTED_ONLY_CREDIT = 0.5


@dataclass(frozen=True)
class SkillMatch:
    wanted: str
    found: str
    evidence: bool = True

    @property
    def credit(self) -> float:
        return EVIDENCE_CREDIT if self.evidence else LISTED_ONLY_CREDIT


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
    """True if the job skill's tokens appear, in order and contiguous, inside the CV skill.

    One direction only: CV "SQL query tuning" satisfies job "SQL",
    but CV "Data" does not satisfy job "Data analysis".
    """
    if normalize_skill(wanted) == normalize_skill(have):
        return True
    return contains_sequence(skill_tokens(have), skill_tokens(wanted))


def implies(have: str, wanted: str) -> bool:
    """One hop only: the CV skill directly implies the job skill."""
    return normalize_skill(wanted) in IMPLIES.get(normalize_skill(have), frozenset())


def find_match(wanted: str, have: list[str]) -> str | None:
    """Prefer an exact (normalized) match, then a token match, then an implication."""
    key = normalize_skill(wanted)
    exact = next((item for item in have if normalize_skill(item) == key), None)
    token = next((item for item in have if skills_match(wanted, item)), None)
    implied = next((item for item in have if implies(item, wanted)), None)
    return exact or token or implied


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


def match_cv_skills(
    wanted: list[str], cv_skills: list[CVSkill], languages: list[str] | None = None
) -> tuple[list[SkillMatch], list[str]]:
    """Like match_skills, but prefer CV skills with evidence (full credit) over listed-only ones.

    A job skill that is a spoken language (e.g. a nice-to-have "Spanish") is also matched
    against the CV's languages; speaking it counts as evidence.
    """
    evidenced = [skill.name for skill in cv_skills if skill.evidence]
    listed = [skill.name for skill in cv_skills]
    matches: list[SkillMatch] = []
    missing: list[str] = []
    for item in wanted:
        if (found := find_match(item, evidenced)) is not None:
            matches.append(SkillMatch(wanted=item, found=found, evidence=True))
        elif (found := find_match(item, languages or [])) is not None:
            matches.append(SkillMatch(wanted=item, found=found, evidence=True))
        elif (found := find_match(item, listed)) is not None:
            matches.append(SkillMatch(wanted=item, found=found, evidence=False))
        else:
            missing.append(item)
    return matches, missing


@dataclass(frozen=True)
class ExperienceSummary:
    relevant_years: float
    total_years: float
    warnings: list[str]


def is_relevant_role(role: Role, job_skills: list[str]) -> bool:
    """A role counts if it used at least one job skill (same token and implication rules)."""
    return any(find_match(skill, role.skills) is not None for skill in job_skills)


def summarize_experience(roles: list[Role], job_skills: list[str], today: date) -> ExperienceSummary:
    """Relevant years come from roles that used a job skill; if the job names none, all count."""
    dated: list[tuple[Role, Interval]] = []
    unreadable = 0
    for role in roles:
        interval = role_interval(role.start, role.end, today)
        if interval is None:
            unreadable += 1
        else:
            dated.append((role, interval))
    if not job_skills:
        relevant = [interval for _, interval in dated]
        no_skills = 0
    else:
        relevant = [interval for role, interval in dated if is_relevant_role(role, job_skills)]
        no_skills = sum(1 for role, _ in dated if not role.skills)
    return ExperienceSummary(
        relevant_years=total_years(relevant),
        total_years=total_years([interval for _, interval in dated]),
        warnings=_experience_warnings(unreadable, no_skills),
    )


def _experience_warnings(unreadable: int, no_skills: int) -> list[str]:
    warnings: list[str] = []
    if no_skills:
        warnings.append(
            f"{no_skills} role(s) have no listed skills and were not counted as relevant experience."
        )
    if unreadable:
        warnings.append(f"{unreadable} role(s) have unreadable dates and were ignored.")
    return warnings


def score_match(cv: CVProfile, job: JobRequirements, today: date | None = None) -> MatchResult:
    job_skills = job.required_skills + job.nice_to_have_skills
    experience = summarize_experience(cv.roles, job_skills, today or date.today())
    required, req_missing = match_cv_skills(job.required_skills, cv.skills, cv.languages)
    nice, nice_missing = match_cv_skills(job.nice_to_have_skills, cv.skills, cv.languages)
    languages, lang_missing = match_skills(job.languages, cv.languages)
    components = [
        _matches_component(
            "required_skills", "Required skills", job.required_skills, required, req_missing
        ),
        _matches_component(
            "nice_to_have_skills", "Nice-to-have skills", job.nice_to_have_skills, nice,
            nice_missing,
        ),
        _experience_component(experience, job.min_years),
        _education_component(cv, job),
        _matches_component("languages", "Languages", job.languages, languages, lang_missing),
    ]
    return MatchResult(
        score=_final_score(components),
        breakdown=components,
        matched_skills=[match.wanted for match in required + nice],
        missing_skills=req_missing,
        reasons=[component.detail for component in components],
        warnings=experience.warnings,
    )


def _final_score(components: list[ComponentScore]) -> int:
    total = sum(component.points for component in components)
    return max(0, min(100, round(total)))


def _component(name: str, ratio: float, detail: str) -> ComponentScore:
    weight = WEIGHTS[name]
    return ComponentScore(
        name=name, weight=weight, ratio=ratio, points=ratio * weight, detail=detail
    )


def _matches_component(
    name: str, label: str, wanted: list[str], matches: list[SkillMatch], missing: list[str]
) -> ComponentScore:
    if not wanted:
        return _component(name, 1.0, f"{label}: none specified by the job.")
    credit = sum(match.credit for match in matches)
    detail = _describe_matches(label, matches, missing, len(wanted))
    return _component(name, credit / len(wanted), detail)


def _describe_matches(
    label: str, matches: list[SkillMatch], missing: list[str], total: int
) -> str:
    """Name the CV skill behind each reworded match, and flag skills listed without evidence."""
    notes = [
        f"{match.wanted} via {match.found}"
        for match in matches
        if match.wanted.casefold() != match.found.casefold()
    ]
    listed_only = [match.wanted for match in matches if not match.evidence]
    if listed_only:
        notes.append(f"listed without evidence: {', '.join(listed_only)}")
    if missing:
        notes.append(f"missing: {', '.join(missing)}")
    detail = f"{label}: {len(matches)} of {total} matched"
    return f"{detail} ({'; '.join(notes)})." if notes else f"{detail}."


def _experience_component(experience: ExperienceSummary, min_years: float) -> ComponentScore:
    relevant, total = experience.relevant_years, experience.total_years
    summary = f"Experience: {relevant:.1f} relevant years of {total:.1f} total"
    if min_years <= 0:
        return _component("experience", 1.0, f"{summary} (no minimum stated).")
    ratio = min(relevant / min_years, 1.0)
    return _component("experience", ratio, f"{summary} (required {min_years:g}).")


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

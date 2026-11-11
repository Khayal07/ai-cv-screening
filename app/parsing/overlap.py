"""Word n-gram overlap between a CV and a job description (copy detection)."""

import re

from app.models import CVProfile
from app.scoring import TOKEN, contains_sequence, skill_tokens

NGRAM_SIZE = 3
# Measured on the 8 synthetic CV/job pairs: max 0.067 (data analyst CV x data analyst job).
# A CV with the job's requirements block pasted in measures 0.694; a pasted full job, 1.0.
OVERLAP_WARNING_THRESHOLD = 0.25
# A CV line counts as copied when at least this share of its word 3-grams is in the job text.
COPIED_LINE_SHARE = 0.5

WORD = re.compile(r"[^\W_]+")


def word_ngrams(text: str, size: int = NGRAM_SIZE) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def text_overlap(cv_text: str, job_text: str) -> float:
    """Larger of: share of CV n-grams found in the job, share of job n-grams found in the CV."""
    cv_grams, job_grams = word_ngrams(cv_text), word_ngrams(job_text)
    if not cv_grams or not job_grams:
        return 0.0
    shared = len(cv_grams & job_grams)
    return max(shared / len(cv_grams), shared / len(job_grams))


def overlap_warning(cv_text: str, job_text: str) -> str | None:
    overlap = text_overlap(cv_text, job_text)
    if overlap < OVERLAP_WARNING_THRESHOLD:
        return None
    return (
        f"{overlap:.0%} of the word sequences match the job description. "
        "Check that the CV was not copied from the posting."
    )


def _line_words(line: str) -> tuple[str, ...]:
    return tuple(WORD.findall(line.lower()))


def is_copied_line(line: str, job_lines: set[tuple[str, ...]], job_grams: set[tuple[str, ...]]) -> bool:
    """Same words as a job line (bullets and punctuation ignored), or mostly the job's 3-grams."""
    words = _line_words(line)
    if not words:
        return False
    if words in job_lines:
        return True
    grams = word_ngrams(line)
    return bool(grams) and len(grams & job_grams) / len(grams) >= COPIED_LINE_SHARE


def split_copied_lines(cv_text: str, job_text: str) -> tuple[list[str], list[str]]:
    """(copied, own): the CV's non-empty lines, split by whether they repeat the job text."""
    job_lines = {words for line in job_text.splitlines() if (words := _line_words(line))}
    job_grams = word_ngrams(job_text)
    copied: list[str] = []
    own: list[str] = []
    for line in filter(str.strip, cv_text.splitlines()):
        (copied if is_copied_line(line, job_lines, job_grams) else own).append(line)
    return copied, own


def mentions_skill(skill: str, lines: list[str]) -> bool:
    wanted = skill_tokens(skill)
    return any(contains_sequence(tuple(TOKEN.findall(line.lower())), wanted) for line in lines)


def discount_copied_skills(profile: CVProfile, cv_text: str, job_text: str) -> CVProfile:
    """Skills named only in lines copied from the job count as listed, not as evidence.

    Lines are never removed; only the evidence flag changes. A skill that also appears in
    the CV's own lines, or that the CV text never names literally, keeps its evidence.
    """
    copied, own = split_copied_lines(cv_text, job_text)
    if not copied:
        return profile
    skills = [
        skill.model_copy(update={"evidence": False})
        if skill.evidence and mentions_skill(skill.name, copied) and not mentions_skill(skill.name, own)
        else skill
        for skill in profile.skills
    ]
    return profile.model_copy(update={"skills": skills})

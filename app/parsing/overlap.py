"""Word n-gram overlap between a CV and a job description (copy detection)."""

import re

NGRAM_SIZE = 3
# Measured on the 8 synthetic CV/job pairs: max 0.067 (data analyst CV x data analyst job).
# A CV with the job's requirements block pasted in measures 0.694; a pasted full job, 1.0.
OVERLAP_WARNING_THRESHOLD = 0.25

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

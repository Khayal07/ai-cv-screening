"""Remove personal data (name, age, gender, photo, contact details) from CV text.

Runs before any text reaches the model, so scoring can never depend on it.
"""

import re

LABELLED_LINE = re.compile(
    r"^\s*(full\s+name|name|date\s+of\s+birth|birth\s*date|d\.?o\.?b\.?|age|gender|sex|"
    r"pronouns|photo|picture|email|e-mail|phone|mobile|tel|address|marital\s+status|"
    r"nationality)\s*[:\-]",
    re.IGNORECASE,
)
FAMILY = re.compile(
    r"\b(married|divorced|widowed)\b|\b(no|one|two|three|four|five|\d+)\s+(children|kids)\b",
    re.IGNORECASE,
)
LABEL_PREFIX = re.compile(r"^\s*[\w\s]+:")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL = re.compile(r"(https?://\S+|www\.\S+|\b(linkedin|github)\.com/\S+)", re.IGNORECASE)
PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")
AGE = re.compile(r"\b(aged\s+\d{1,2}|\d{1,2}\s*(years?\s+old|y/o|yo)\b)", re.IGNORECASE)
TITLE_WITH_NAME = re.compile(r"\b(Mr|Mrs|Ms|Miss|Mx)\.?\s+(?:[A-Z][\w'-]+\s?){1,3}")
PRONOUNS = re.compile(r"\b(he/him|she/her|they/them)\b", re.IGNORECASE)
# "Born 1990", "Born in 1990", "born on 3 May 1990", "Born in Lisbon": up to the next , ; or line end.
BIRTH = re.compile(r"\bborn(?:\s+(?:in|on))?\s*:?\s+[^\s,;][^,;\n]*", re.IGNORECASE)
MIN_PHONE_DIGITS = 9

JOB_TITLE_WORDS = {
    "engineer", "developer", "manager", "analyst", "designer", "intern",
    "consultant", "scientist", "architect", "lead", "specialist", "administrator",
    "director", "officer", "technician", "programmer", "tester", "devops",
    "senior", "junior", "head", "coordinator", "assistant", "owner", "curriculum",
    "vitae", "resume", "cv", "profile", "summary",
}


def strip_personal_data(text: str) -> str:
    lines = [line for line in text.splitlines() if not LABELLED_LINE.match(line)]
    lines = _drop_name_header(lines)
    kept = [cleaned for line in lines if (cleaned := _clean_line(line)) is not None]
    return _collapse_blank_lines("\n".join(kept)).strip()


def looks_like_name(line: str) -> bool:
    words = line.split()
    if not 2 <= len(words) <= 4:
        return False
    if any(word.lower().strip(",.") in JOB_TITLE_WORDS for word in words):
        return False
    return all(_is_capitalized_word(word) for word in words)


def _is_capitalized_word(word: str) -> bool:
    letters = word.replace("-", "").replace("'", "").replace(".", "")
    return letters.isalpha() and word[0].isupper()


def _drop_name_header(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        if line.strip():
            if looks_like_name(line.strip()):
                return lines[:index] + lines[index + 1 :]
            return lines
    return lines


def _clean_line(line: str) -> str | None:
    """Return the cleaned line, or None if only residue is left after removal."""
    cleaned = line
    for pattern in (EMAIL, URL, AGE, BIRTH, TITLE_WITH_NAME, PRONOUNS, FAMILY):
        cleaned = pattern.sub("", cleaned)
    cleaned = PHONE_CANDIDATE.sub(_redact_phone, cleaned).rstrip()
    if cleaned != line.rstrip() and _is_residue(cleaned):
        return None
    return cleaned


def _is_residue(line: str) -> bool:
    """True when a line has no words left apart from an optional 'Label:' prefix."""
    return re.search(r"\w", LABEL_PREFIX.sub("", line)) is None


def _redact_phone(match: re.Match[str]) -> str:
    digits = sum(char.isdigit() for char in match.group())
    return "" if digits >= MIN_PHONE_DIGITS else match.group()


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)

"""Remove personal data (name, age, gender, photo, contact details, referees) from CV text.

Runs before any text reaches the model, so scoring can never depend on it.
"""

import re

from app.parsing.dates import DATE_RANGE

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

# "References: Jane Doe", "Referee - J. Smith", "Reference available from ...", "References available on request".
REFERENCE_LINE = re.compile(
    r"^[\s\-*•]*(references?|referees?)\b\s*(?:[:\-–]|available\b)", re.IGNORECASE
)
# A bare "References" / "Referees:" heading: the block below it names other people.
REFERENCE_HEADING = re.compile(r"^[\s\-*•]*(references|referees)\s*:?\s*$", re.IGNORECASE)
MAX_REFERENCE_BLOCK_LINES = 8
# A whole line that is only a letter sign-off; the next line may be the signer's name.
SIGN_OFF = re.compile(
    r"^\s*(?:(?:kind|best|warm|warmest)\s+regards|regards|best\s+wishes|"
    r"(?:yours\s+)?(?:sincerely|faithfully|truly))\s*[,.!]?\s*$",
    re.IGNORECASE,
)
SECTION_HEADINGS = {
    "profile", "summary", "experience", "work experience", "professional experience",
    "employment", "employment history", "education", "skills", "technical skills",
    "languages", "projects", "certifications", "courses", "interests", "publications",
    "volunteering", "awards",
}
COMPANY_WORDS = {
    "ltd", "inc", "llc", "gmbh", "corp", "corporation", "company", "co", "group", "plc",
    "ag", "bank", "university", "college", "labs", "technologies", "solutions", "systems",
    "partners", "consulting", "studio", "agency", "school", "institute", "fictional",
}

JOB_TITLE_WORDS = {
    "engineer", "developer", "manager", "analyst", "designer", "intern",
    "consultant", "scientist", "architect", "lead", "specialist", "administrator",
    "director", "officer", "technician", "programmer", "tester", "devops",
    "senior", "junior", "head", "coordinator", "assistant", "owner", "curriculum",
    "vitae", "resume", "cv", "profile", "summary",
}


def strip_personal_data(text: str) -> str:
    lines = _drop_signatures(_drop_references(text.splitlines()))
    lines = [line for line in lines if not LABELLED_LINE.match(line)]
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


def _drop_references(lines: list[str]) -> list[str]:
    """Remove reference lines and the block under a bare References heading."""
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if REFERENCE_HEADING.match(line):
            index = _end_of_reference_block(lines, index + 1)
            continue
        if not REFERENCE_LINE.match(line):
            kept.append(line)
        index += 1
    return kept


def _end_of_reference_block(lines: list[str], start: int) -> int:
    """Index after the block: it ends at a blank line, a section heading, or the line cap."""
    index = start
    while index < len(lines) and not lines[index].strip():
        index += 1
    end = min(len(lines), index + MAX_REFERENCE_BLOCK_LINES)
    while index < end and lines[index].strip() and not _is_section_heading(lines[index]):
        index += 1
    return index


def _is_section_heading(line: str) -> bool:
    return line.strip().rstrip(":").strip().lower() in SECTION_HEADINGS


def _drop_signatures(lines: list[str]) -> list[str]:
    """Remove sign-off lines, and the next non-blank line when it is a person's name."""
    kept: list[str] = []
    after_sign_off = False
    for line in lines:
        if SIGN_OFF.match(line):
            after_sign_off = True
            continue
        if after_sign_off and line.strip():
            after_sign_off = False
            if _is_signature_name(line.strip()):
                continue
        kept.append(line)
    return kept


def _is_signature_name(line: str) -> bool:
    words = {word.lower().strip(",.()") for word in line.split()}
    return looks_like_name(line) and not words & COMPANY_WORDS


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
    """Drop phone-like digit runs, but keep date ranges such as 2019-03 - 2021-07 inside them."""
    candidate, kept, last = match.group(), [], 0
    for date_range in DATE_RANGE.finditer(candidate):
        kept += [_drop_if_phone(candidate[last : date_range.start()]), date_range.group()]
        last = date_range.end()
    kept.append(_drop_if_phone(candidate[last:]))
    return "".join(kept)


def _drop_if_phone(piece: str) -> str:
    digits = sum(char.isdigit() for char in piece)
    return "" if digits >= MIN_PHONE_DIGITS else piece


def _collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text)

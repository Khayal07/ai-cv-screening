"""Rule-based check that a text is a CV, not a job posting or a bare keyword list.

Runs before any model call, so a rejected document costs nothing.
"""

import re
from dataclasses import dataclass, field

MIN_WORDS = 40
POSTING_PHRASES_WITHOUT_DATES = 2
POSTING_PHRASES_ALWAYS = 3

_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s+|\d{{1,2}}/)?(?:19|20)\d{{2}}"
_END = rf"(?:{_DATE}|present|now|today|current)"
DATE_RANGE = re.compile(rf"\b{_DATE}\s*(?:-|–|—|to)\s*{_END}\b", re.IGNORECASE)

EDUCATION = re.compile(
    r"\b(b\.?sc|m\.?sc|b\.?a|m\.?a|mba|ph\.?d|bachelor'?s?|master'?s?|university|college|"
    r"degree|diploma|bootcamp)\b",
    re.IGNORECASE,
)

POSTING_PHRASES = (
    "we are looking for",
    "nice to have",
    "the ideal candidate",
    "what we offer",
    "how to apply",
    "apply now",
    "join our team",
    "about the role",
    "you will",
    "you'll",
    "must have",
)
# Cover letter markers: a salutation line, the classic opening, and sign-offs.
LETTER_SIGNALS: dict[str, re.Pattern[str]] = {
    "Dear ...": re.compile(r"^\s*dear\s+\w", re.IGNORECASE | re.MULTILINE),
    "I am writing to apply": re.compile(r"\bi am writing to apply\b", re.IGNORECASE),
    "Yours sincerely": re.compile(r"\byours\s+(sincerely|faithfully)\b", re.IGNORECASE),
    "Kind regards": re.compile(r"\b(kind|best|warm)\s+regards\b", re.IGNORECASE),
}
LETTER_SIGNALS_TO_REJECT = 2
POSTING_HEADING = re.compile(r"^\s*(requirements|responsibilities)\s*:?\s*$", re.IGNORECASE | re.M)
WORD = re.compile(r"[^\W_]+")


class NotACVError(Exception):
    """Raised when a submitted CV looks like a job posting or lacks CV content."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("does not look like a CV")
        self.reasons = reasons


@dataclass(frozen=True)
class DocumentCheck:
    words: int
    date_ranges: int
    has_education: bool
    posting_phrases: list[str]
    letter_signals: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def is_cv(self) -> bool:
        return not self.reasons


def looks_like_cv(text: str) -> DocumentCheck:
    words = len(WORD.findall(text))
    date_ranges = len(DATE_RANGE.findall(text))
    has_education = EDUCATION.search(text) is not None
    phrases = find_posting_phrases(text)
    letter = find_letter_signals(text)
    reasons = _rejection_reasons(words, date_ranges, has_education, phrases)
    if len(letter) >= LETTER_SIGNALS_TO_REJECT:
        quoted = ", ".join(f'"{signal}"' for signal in letter)
        reasons.append(f"It reads like a cover letter ({quoted}), not a CV.")
    return DocumentCheck(words, date_ranges, has_education, phrases, letter, reasons)


def find_letter_signals(text: str) -> list[str]:
    return [name for name, pattern in LETTER_SIGNALS.items() if pattern.search(text)]


def find_posting_phrases(text: str) -> list[str]:
    lowered = re.sub(r"\s+", " ", text.lower())
    found = [phrase for phrase in POSTING_PHRASES if phrase in lowered]
    headings = {match.group(1).lower() for match in POSTING_HEADING.finditer(text)}
    return found + sorted(headings)


def _rejection_reasons(
    words: int, date_ranges: int, has_education: bool, phrases: list[str]
) -> list[str]:
    reasons: list[str] = []
    quoted = ", ".join(f'"{phrase}"' for phrase in phrases)
    if words < MIN_WORDS:
        reasons.append(f"The text is too short for a CV ({words} words; at least {MIN_WORDS}).")
    if len(phrases) >= POSTING_PHRASES_ALWAYS:
        reasons.append(f"It contains several job-posting phrases ({quoted}).")
    elif len(phrases) >= POSTING_PHRASES_WITHOUT_DATES and date_ranges == 0:
        reasons.append(f"It reads like a job posting ({quoted}) and has no dated work history.")
    if date_ranges == 0 and not has_education:
        reasons.append("No dated roles (such as 2019 - 2022) and no education details were found.")
    return reasons

"""Year-month date ranges in CV text: recognise them and normalise dotted ones.

Only whole ranges are touched ("2019.03 - 2021.07", "2024.05 - present"). A lone dotted
number such as a version ("Python 3.11", "v2019.03") is never changed.
"""

import re

_MONTH = r"(?:1[0-2]|0?[1-9])"
_YEAR = r"(?:19|20)\d{2}"
# A range side: a year, optionally followed by a month after ".", "/" or "-".
_SIDE = rf"{_YEAR}(?:[./-]{_MONTH})?"
_END_WORDS = r"present|now|current|today"
_SEPARATOR = r"\s*(?:-|–|—|\bto\b)\s*"

DATE_RANGE = re.compile(
    rf"(?<![\w.])({_SIDE})({_SEPARATOR})({_SIDE}|{_END_WORDS})(?!\w|\.\d)", re.IGNORECASE
)
DOTTED_SIDE = re.compile(rf"^({_YEAR})\.({_MONTH})$")


def contains_date_range(text: str) -> bool:
    return DATE_RANGE.search(text) is not None


def normalize_date_ranges(text: str) -> str:
    """'2023.01 - 2023.12' -> '2023-01 - 2023-12'; ranges without a dotted side are unchanged."""
    return DATE_RANGE.sub(_normalize_range, text)


def _normalize_range(match: re.Match[str]) -> str:
    start, _, end = match.groups()
    if not (DOTTED_SIDE.match(start) or DOTTED_SIDE.match(end)):
        return match.group(0)
    return f"{_dash_side(start)} - {_dash_side(end)}"


def _dash_side(side: str) -> str:
    dotted = DOTTED_SIDE.match(side)
    if dotted is None:
        return side
    year, month = dotted.groups()
    return f"{year}-{int(month):02d}"

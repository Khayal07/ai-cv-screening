"""Date math for work experience. Months are counted as year * 12 + (month - 1).

Rules:
- Accepted values: "YYYY-MM", "YYYY", or "present" / "now" / "current" / "today".
- A year without a month counts as mid-year (June), so "2017" to "2020" is 3.0 years.
- Duration is end month minus start month.
- An end date after today is capped at today; a start date after today is invalid.
- Overlapping or touching roles are merged so parallel jobs are not counted twice.
"""

import re
from datetime import date

Interval = tuple[int, int]

MID_YEAR_MONTH = 6
EARLIEST_YEAR = 1950
PRESENT_WORDS = frozenset({"present", "now", "current", "today"})
DATE_VALUE = re.compile(r"(\d{4})(?:-(\d{1,2}))?")


def month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def parse_month(value: str, today: date) -> int | None:
    text = value.strip().lower()
    if text in PRESENT_WORDS:
        return month_index(today.year, today.month)
    match = DATE_VALUE.fullmatch(text)
    if match is None:
        return None
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else MID_YEAR_MONTH
    if year < EARLIEST_YEAR or not 1 <= month <= 12:
        return None
    return month_index(year, month)


def role_interval(start: str, end: str, today: date) -> Interval | None:
    """Return the role's [start, end) in months, or None if the dates are unusable."""
    now = month_index(today.year, today.month)
    first, last = parse_month(start, today), parse_month(end, today)
    if first is None or last is None or first > now:
        return None
    last = min(last, now)
    if last < first:
        return None
    return first, last


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    merged: list[Interval] = []
    for first, last in sorted(intervals):
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def total_years(intervals: list[Interval]) -> float:
    return sum(last - first for first, last in merge_intervals(intervals)) / 12

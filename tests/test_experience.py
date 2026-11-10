from datetime import date

import pytest

from app.experience import merge_intervals, month_index, parse_month, role_interval, total_years

TODAY = date(2026, 11, 10)
NOW = month_index(2026, 11)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2020-03", month_index(2020, 3)),
        ("2020-3", month_index(2020, 3)),
        (" 2020-12 ", month_index(2020, 12)),
        ("2020", month_index(2020, 6)),
        ("present", NOW),
        ("Present", NOW),
        ("now", NOW),
        ("current", NOW),
        ("today", NOW),
    ],
)
def test_parse_month_accepts_known_formats(value: str, expected: int) -> None:
    assert parse_month(value, TODAY) == expected


@pytest.mark.parametrize(
    "value", ["", "soon", "2020-13", "2020-00", "20-03", "March 2020", "1890", "2020-03-01"]
)
def test_parse_month_rejects_unreadable_values(value: str) -> None:
    assert parse_month(value, TODAY) is None


def test_year_only_dates_count_from_mid_year() -> None:
    assert total_years([role_interval("2017", "2020", TODAY)]) == 3.0


def test_month_dates_give_exact_duration() -> None:
    assert total_years([role_interval("2019-01", "2020-07", TODAY)]) == 1.5


def test_present_ends_today() -> None:
    assert role_interval("2026-05", "present", TODAY) == (month_index(2026, 5), NOW)


def test_end_after_today_is_capped() -> None:
    assert role_interval("2025-11", "2030-01", TODAY) == (month_index(2025, 11), NOW)


def test_start_after_today_is_invalid() -> None:
    assert role_interval("2027-01", "present", TODAY) is None


def test_end_before_start_is_invalid() -> None:
    assert role_interval("2022-01", "2020-01", TODAY) is None


def test_unreadable_date_is_invalid() -> None:
    assert role_interval("sometime", "2020", TODAY) is None


def test_overlapping_roles_are_merged() -> None:
    intervals = [(0, 24), (12, 36), (48, 60)]
    assert merge_intervals(intervals) == [(0, 36), (48, 60)]
    assert total_years(intervals) == 4.0


def test_touching_and_contained_roles_are_merged() -> None:
    assert merge_intervals([(12, 24), (0, 12), (2, 6)]) == [(0, 24)]


def test_no_roles_means_zero_years() -> None:
    assert total_years([]) == 0.0

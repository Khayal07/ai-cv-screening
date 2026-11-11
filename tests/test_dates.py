import pytest

from app.parsing.dates import contains_date_range, normalize_date_ranges
from app.service import prepare_cv_text, screen
from tests.conftest import VALID_CV_TEXT, FakeLLMClient, role, used


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2023.01 - 2023.12", "2023-01 - 2023-12"),
        ("2019.03-2021.07", "2019-03 - 2021-07"),
        ("2019.3 – 2020.11", "2019-03 - 2020-11"),
        ("2024.05 - present", "2024-05 - present"),
        ("2018 - 2020.06", "2018 - 2020-06"),
        ("2019.03 to 2021.07", "2019-03 - 2021-07"),
        ("Developer, Example Ltd, 2023.01 - 2023.12.", "Developer, Example Ltd, 2023-01 - 2023-12."),
    ],
)
def test_dotted_ranges_become_year_month(text: str, expected: str) -> None:
    assert normalize_date_ranges(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Python 3.11",
        "Upgraded Python 3.11 - 3.12 across services",
        "Released v2019.03 - v2019.04",
        "Release 2019.03 shipped",
        "2019 - 2021",
        "2019-03 - 2021-07",
        "Sept. 2021 - Present",
        "2023.13 - 2024.01",
        "Scored 2019.03 points",
    ],
)
def test_other_text_is_unchanged(text: str) -> None:
    assert normalize_date_ranges(text) == text


def test_contains_date_range() -> None:
    assert contains_date_range("worked 2019-03 - 2021-07 at Example")
    assert contains_date_range("2015 - 2019")
    assert not contains_date_range("Python 3.11 - 3.12")
    assert not contains_date_range("+44 7700 900123")


def test_prepared_cv_text_has_normalised_ranges_and_no_personal_data() -> None:
    text = "Name: Sam Example\nIntern, Example Bank (fictional), 2023.01 - 2023.12\nPython 3.11"
    assert prepare_cv_text(text) == "Intern, Example Bank (fictional), 2023-01 - 2023-12\nPython 3.11"


def test_screen_sends_normalised_dates_to_the_model() -> None:
    fake = FakeLLMClient([
        {"skills": used("Python"), "roles": [role(2, "Python")], "education_level": "bachelor"},
        {"required_skills": ["Python"]},
    ])
    screen(VALID_CV_TEXT + "\nIntern, Example Labs (fictional), 2017.01 - 2017.12", "Python", fake)
    assert "2017-01 - 2017-12" in fake.calls[0][1]
    assert "2017.01" not in fake.calls[0][1]

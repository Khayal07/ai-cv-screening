import pytest

from app.parsing.anonymize import looks_like_name, strip_personal_data


def test_removes_name_header_line() -> None:
    text = "Jordan Avery Blake\nSkills: Python, SQL"
    result = strip_personal_data(text)
    assert "Jordan" not in result
    assert "Skills: Python, SQL" in result


def test_keeps_job_title_first_line() -> None:
    text = "Senior Backend Engineer\nSkills: Python, SQL"
    result = strip_personal_data(text)
    assert result.startswith("Senior Backend Engineer")


@pytest.mark.parametrize(
    "line",
    ["Data Analyst", "Product Designer", "Software Developer", "Marketing Intern", "Project Manager"],
)
def test_job_titles_are_not_names(line: str) -> None:
    assert not looks_like_name(line)


@pytest.mark.parametrize(
    "line",
    ["Jordan Blake", "Mary-Ann O'Neil", "Alex J. Morgan", "Ana Lucia De Souza"],
)
def test_names_are_detected(line: str) -> None:
    assert looks_like_name(line)


@pytest.mark.parametrize(
    "line",
    ["python developer since 2015", "Skills: Python, SQL", "Experience", "A B C D E"],
)
def test_other_first_lines_are_not_names(line: str) -> None:
    assert not looks_like_name(line)


@pytest.mark.parametrize(
    "line",
    [
        "Name: Jordan Blake",
        "Full name - Jordan Blake",
        "Date of birth: 1990-01-01",
        "DOB: 01/01/1990",
        "Age: 34",
        "Gender: Female",
        "Sex: M",
        "Pronouns: she/her",
        "Photo: attached",
    ],
)
def test_removes_labelled_personal_lines(line: str) -> None:
    text = f"Backend Engineer\n{line}\nSkills: Go"
    result = strip_personal_data(text)
    assert line not in result
    assert "Skills: Go" in result


def test_removes_inline_age() -> None:
    result = strip_personal_data("Profile\nI am 34 years old and love Python.")
    assert "34" not in result
    assert "love Python" in result


def test_removes_gendered_title_with_name() -> None:
    result = strip_personal_data("Profile\nReferences: Mrs. Jane Doe, former manager")
    assert "Jane" not in result
    assert "Mrs" not in result


def test_removes_contact_details() -> None:
    text = (
        "Profile\n"
        "Contact jordan.blake@example.com or +1 (555) 123-4567\n"
        "See https://linkedin.com/in/jordanblake"
    )
    result = strip_personal_data(text)
    assert "@" not in result
    assert "555" not in result
    assert "jordanblake" not in result


def test_keeps_date_ranges() -> None:
    result = strip_personal_data("Experience\nAcme Corp 2015 - 2019")
    assert "2015 - 2019" in result


def test_keeps_professional_content() -> None:
    text = "Experience\n5 years of Python, FastAPI and PostgreSQL.\nBSc Computer Science"
    assert strip_personal_data(text) == text

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


def test_removes_marital_and_family_details() -> None:
    result = strip_personal_data("Profile\nAged 38, married, two children.\nSkills: Go")
    assert "married" not in result
    assert "children" not in result
    assert result == "Profile\nSkills: Go"


@pytest.mark.parametrize("line", ["Marital status: single", "Nationality: Examplestan"])
def test_removes_marital_status_and_nationality_lines(line: str) -> None:
    assert strip_personal_data(f"Profile\n{line}\nSkills: Go") == "Profile\nSkills: Go"


def test_drops_label_left_empty_after_removal() -> None:
    text = "Profile\nContact: someone@example.com | www.someone.example\nSkills: Go"
    assert strip_personal_data(text) == "Profile\nSkills: Go"


def test_keeps_single_as_professional_word() -> None:
    text = "Experience\nBuilt a single sign-on service."
    assert strip_personal_data(text) == text


def test_keeps_date_ranges() -> None:
    result = strip_personal_data("Experience\nAcme Corp 2015 - 2019")
    assert "2015 - 2019" in result


def test_keeps_professional_content() -> None:
    text = "Experience\n5 years of Python, FastAPI and PostgreSQL.\nBSc Computer Science"
    assert strip_personal_data(text) == text


@pytest.mark.parametrize(
    "line",
    [
        "Born 1990",
        "Born in 1990",
        "born on 3 May 1990",
        "Born in Lisbon",
        "Born: 12/02/1990",
    ],
)
def test_removes_birth_details(line: str) -> None:
    assert strip_personal_data(f"Profile\n{line}\nPython developer") == "Profile\nPython developer"


def test_removes_birth_details_inside_a_line() -> None:
    assert strip_personal_data("Born 1990, married, two children.") == ""
    cleaned = strip_personal_data("Analyst, born in 1990; based in Example City")
    assert "born" not in cleaned and "1990" not in cleaned
    assert cleaned.startswith("Analyst") and cleaned.endswith("; based in Example City")


@pytest.mark.parametrize(
    "text",
    ["Migrated born-digital archives to S3.", "Stubborn bugs fixed in the billing service."],
)
def test_keeps_words_that_only_contain_born(text: str) -> None:
    assert strip_personal_data(text) == text


@pytest.mark.parametrize(
    "line",
    [
        "Analyst, 2019-03 - 2021-07",
        "Intern Developer, Example Bank (fictional), 2023.01 - 2023.12",
        "Developer, 2019.03-2021.07",
        "Engineer, 2020/01 - 2022/06",
        "Engineer, 2019-03 – present",
    ],
)
def test_year_month_date_ranges_are_not_phone_numbers(line: str) -> None:
    assert strip_personal_data(line) == line


def test_phone_next_to_a_date_range_is_still_removed() -> None:
    cleaned = strip_personal_data("Contact 555 010 2233 2019-03 - 2020-01")
    assert "2233" not in cleaned
    assert "2019-03 - 2020-01" in cleaned


@pytest.mark.parametrize("phone", ["+1 (555) 010-2233", "+44 7700 900123", "+45 2012 3456"])
def test_phone_numbers_are_still_removed(phone: str) -> None:
    assert strip_personal_data(f"Call {phone} today") == "Call  today"


# ---------- references and signatures ----------


@pytest.mark.parametrize(
    "line",
    [
        "References: Jane Doe, Head of Data, Example Ltd (fictional)",
        "Referee: Dr. Anna Berg",
        "Referees - Tom Price and Lea Kim",
        "Reference available from Jane Doe on request",
        "References available on request.",
        "- References: Jane Doe",
        "REFERENCES: JANE DOE",
    ],
)
def test_reference_lines_are_removed(line: str) -> None:
    text = f"Data Analyst\n{line}\nSkills: SQL"
    assert strip_personal_data(text) == "Data Analyst\nSkills: SQL"


def test_block_under_references_heading_is_removed() -> None:
    text = (
        "Skills: Python\n\nReferences\n\nJane Doe\nHead of Data, Example Ltd (fictional)\n"
        "jane.doe@example.com\n\nLanguages: English"
    )
    result = strip_personal_data(text)
    assert "Jane" not in result and "Head of Data" not in result
    assert "Skills: Python" in result and "Languages: English" in result


def test_reference_block_stops_at_a_section_heading() -> None:
    text = "Referees:\nTom Price\nEducation\nBSc Mathematics, 2015 - 2018"
    assert strip_personal_data(text) == "Education\nBSc Mathematics, 2015 - 2018"


def test_reference_block_is_capped() -> None:
    block = "\n".join(f"Line {n} of notes" for n in range(12))
    result = strip_personal_data(f"Skills: Go\nReferences\n{block}")
    assert "Line 7 of notes" not in result
    assert "Line 8 of notes" in result


@pytest.mark.parametrize(
    "line",
    [
        "Built a reference implementation in Rust for the parser.",
        "Reference data management for 40 product feeds.",
        "Referenced in the 2024 engineering report.",
        "Wrote the referee assignment tool for a sports league (fictional).",
        "Kind of work: contract, remote.",
        "Regards for safety rules were part of onboarding.",
        "Worked with Johnson & Johnson and Smith Partners as clients.",
    ],
)
def test_lines_that_only_mention_references_are_kept(line: str) -> None:
    assert strip_personal_data(f"Data Analyst\n{line}") == f"Data Analyst\n{line}"


@pytest.mark.parametrize("sign_off", ["Kind regards,", "Best regards", "Yours sincerely,", "Sincerely", "Regards,"])
def test_sign_off_and_signer_name_are_removed(sign_off: str) -> None:
    text = f"Skills: SQL, Excel\n\n{sign_off}\nJane Doe"
    assert strip_personal_data(text) == "Skills: SQL, Excel"


@pytest.mark.parametrize(
    "after",
    ["Python, SQL, Docker", "Example Bank Ltd", "Acme Labs", "Data Analyst", "Available from March 2027."],
)
def test_line_after_sign_off_is_kept_when_not_a_name(after: str) -> None:
    text = f"Skills: SQL\nKind regards,\n{after}"
    assert strip_personal_data(text) == f"Skills: SQL\n{after}"

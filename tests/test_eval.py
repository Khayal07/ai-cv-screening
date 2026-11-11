"""Replay the recorded real extractions through the full pipeline and check every label.

The fixtures in data/eval/extractions were recorded once with the real model
(scripts/run_eval.py --real --record). Nothing here touches the network.
"""

import json
import re

import pytest

from app.eval_clients import fixture_path, load_fixture, replay_client, stale_fixtures
from app.evaluation import (
    CVS_DIR,
    EXTRACTIONS_DIR,
    CaseResult,
    EvalCase,
    check_rankings,
    load_labels,
    read_text,
    run_case,
)
from app.parsing.anonymize import strip_personal_data

LABELS = load_labels()

# Real failures found by the first recorded run (2026-11-11), each with its root cause.
# Strict: a case that starts passing must have its entry removed.
KNOWN_FAILURES: dict[str, str] = {
    "be_junior_projects@jd_backend_engineer":
        "67 > 65: projects extracted as roles (2 years), unfinished degree as bachelor",
    "da_junior_projects@jd_data_analyst":
        "82 > 65: projects extracted as roles (2 years), unfinished degree as bachelor",
    "da_strong_2@jd_data_analyst":
        "78 < 80: 'Tableau or Power BI' extracted as two required skills",
    "da_copied_reqs@jd_data_analyst":
        "80 > 79: skills from the pasted requirements block earn listed-only credit",
}

FIXTURE_KEYS = {
    "skills", "roles", "education_level", "languages",
    "required_skills", "nice_to_have_skills", "min_years",
}
CONTACT = re.compile(r"[\w.+-]+@[\w.-]+|\+?\d[\d ()]{8,}\d|\b19[4-9]\d\b")
WORD = re.compile(r"[A-Z][a-z]{2,}")


def case_params() -> list[object]:
    params: list[object] = []
    for case in LABELS.cases:
        reason = KNOWN_FAILURES.get(case.id)
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(case, id=case.id, marks=marks))
    return params


def replay(case: EvalCase) -> CaseResult:
    return run_case(case, replay_client(case), LABELS.today)


@pytest.mark.parametrize("case", case_params())
def test_case_matches_its_label(case: EvalCase) -> None:
    result = replay(case)
    assert result.passed, result.failures


def test_known_failures_are_real_case_ids() -> None:
    assert set(KNOWN_FAILURES) <= {case.id for case in LABELS.cases}


def test_rankings_hold_for_both_jobs() -> None:
    checks = check_rankings(LABELS.rankings, [replay(case) for case in LABELS.cases])
    assert len(checks) == 2 * len(LABELS.rankings)
    assert all(check.passed for check in checks), [check.describe() for check in checks]


# Fixtures whose model input changed after recording; re-recorded with the next prompt version.
KNOWN_STALE = ["da_strong_2.txt"]  # "Born 1990" is now removed before extraction


def test_fixtures_match_current_prompt_and_text() -> None:
    assert stale_fixtures(LABELS) == KNOWN_STALE


def test_every_fixture_is_used() -> None:
    used = {fixture_path(name).name for case in LABELS.cases for name in (case.cv, case.job)}
    assert {path.name for path in EXTRACTIONS_DIR.glob("*.json")} <= used


@pytest.mark.parametrize("path", sorted(EXTRACTIONS_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_fixture_holds_no_personal_data(path: object) -> None:
    fixture = load_fixture(path)  # type: ignore[arg-type]
    assert fixture is not None
    assert set(fixture.output) <= FIXTURE_KEYS
    output = json.dumps(fixture.output)
    assert not CONTACT.search(output)
    source = CVS_DIR / f"{path.stem}.txt"  # type: ignore[attr-defined]
    if source.exists():
        text = read_text(source)
        removed = set(WORD.findall(text)) - set(WORD.findall(strip_personal_data(text)))
        assert not [word for word in removed if re.search(rf"\b{word}\b", output)]

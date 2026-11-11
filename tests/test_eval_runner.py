"""Evaluation runner, replay/budget/recording clients and the run_eval script. No network."""

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.eval_clients import (
    BudgetExceeded,
    BudgetLLMClient,
    RecordingLLMClient,
    ReplayError,
    ReplayLLMClient,
    case_inputs,
    fixture_path,
    input_hash,
    load_fixture,
    replay_client,
    save_recordings,
    stale_fixtures,
)
from app.evaluation import (
    CaseResult,
    EvalCase,
    EvalLabels,
    Outcome,
    RankingRule,
    WarningKind,
    check_case,
    check_rankings,
    format_summary,
    format_table,
    judge,
    load_labels,
    run_case,
    warning_kinds,
)
from app.extraction import CV_SYSTEM_PROMPT, JOB_SYSTEM_PROMPT, PROMPT_VERSION
from app.llm.cache import CachedLLMClient
from app.llm.client import LLMResponseError
from app.parsing.overlap import overlap_warning
from app.scoring import _experience_warnings
from scripts.run_eval import main
from tests.conftest import FakeLLMClient, role, used

TODAY = load_labels().today
CV_JSON: dict[str, Any] = {
    "skills": used("Python"), "roles": [role(3, "Python")], "education_level": "bachelor",
    "languages": ["English"],
}
JOB_JSON: dict[str, Any] = {"required_skills": ["Python"], "min_years": 2}


def case(**overrides: Any) -> EvalCase:
    data = {"cv": "be_strong_1.txt", "job": "jd_backend_engineer.txt", "category": "strong",
            "outcome": "scored", "score": [80, 100]}
    return EvalCase.model_validate(data | overrides)


def result(category: str, score: int, job: str = "jd_backend_engineer.txt") -> CaseResult:
    band = {"strong": [80, 100], "partial": [50, 79], "weak": [0, 45]}[category]
    return judge(case(category=category, score=band, job=job), Outcome.SCORED, score)


class CannedClient:
    """Answers by system prompt and counts calls, like a real model would be called."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self.error = error

    @property
    def model(self) -> str:
        return "fake-model"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return CV_JSON if system == CV_SYSTEM_PROMPT else JOB_JSON


# --- warnings and case checks ---

def test_warning_kinds_recognise_real_warning_texts() -> None:
    texts = [overlap_warning("a b c d e", "a b c d e") or "", *_experience_warnings(1, 1)]
    assert warning_kinds(texts) == (
        WarningKind.OVERLAP, WarningKind.NO_SKILL_ROLES, WarningKind.UNREADABLE_DATES
    )


def test_score_in_range_passes() -> None:
    assert check_case(case(), Outcome.SCORED, 80, ()) == ()


def test_score_outside_range_fails() -> None:
    assert check_case(case(), Outcome.SCORED, 79, ()) == ("score 79 is outside 80-100",)


def test_scored_when_rejection_expected_fails() -> None:
    rejected = case(category="non_cv", outcome="rejected_422", score=None)
    assert check_case(rejected, Outcome.SCORED, 61, ()) == (
        "expected rejected_422, got scored (score 61)",
    )


def test_rejected_when_score_expected_fails() -> None:
    assert check_case(case(), Outcome.REJECTED, None, ()) == ("expected scored, got rejected_422",)


def test_missing_expected_warning_fails() -> None:
    copied = case(category="copied_requirements", score=[45, 79], warnings=["overlap"])
    assert check_case(copied, Outcome.SCORED, 60, ()) == ("missing warning: overlap",)


def test_unexpected_overlap_warning_fails() -> None:
    assert check_case(case(), Outcome.SCORED, 90, (WarningKind.OVERLAP,)) == (
        "unexpected warning: overlap",
    )


def test_unlisted_experience_warning_is_allowed() -> None:
    assert check_case(case(), Outcome.SCORED, 90, (WarningKind.NO_SKILL_ROLES,)) == ()


# --- rankings ---

RULE = RankingRule(above=["strong"], below=["partial"])


def test_ranking_passes_when_every_above_beats_every_below() -> None:
    [check] = check_rankings([RULE], [result("strong", 82), result("strong", 95),
                                      result("partial", 81)])
    assert check.passed
    assert check.describe() == "jd_backend_engineer: min(strong) 82 > max(partial) 81"


def test_ranking_fails_on_a_tie_or_inversion() -> None:
    [check] = check_rankings([RULE], [result("strong", 80), result("partial", 80)])
    assert not check.passed


def test_ranking_is_checked_per_job() -> None:
    checks = check_rankings([RULE], [
        result("strong", 90), result("partial", 70),
        result("strong", 60, job="jd_data_analyst.txt"),
        result("partial", 75, job="jd_data_analyst.txt"),
    ])
    assert [check.passed for check in checks] == [True, False]


def test_ranking_with_a_missing_category_fails() -> None:
    [check] = check_rankings([RULE], [result("strong", 90)])
    assert not check.passed
    assert check.describe().endswith("max(partial) -")


# --- report ---

def test_plain_table_has_header_and_aligned_rows() -> None:
    lines = format_table([result("strong", 92), result("partial", 40)]).splitlines()
    assert lines[0].split() == ["Case", "Category", "Expected", "Actual", "Result"]
    assert lines[1].split() == ["be_strong_1@jd_backend_engineer", "strong", "80-100", "92", "PASS"]
    assert lines[2].split()[-1] == "FAIL"


def test_markdown_table_has_separator_row() -> None:
    lines = format_table([result("strong", 92)], markdown=True).splitlines()
    assert lines[0] == "| Case | Category | Expected | Actual | Result |"
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2] == "| be_strong_1@jd_backend_engineer | strong | 80-100 | 92 | PASS |"


def test_summary_reports_pass_rate_and_failed_labels() -> None:
    results = [result("strong", 92), result("partial", 40)]
    summary = format_summary(results, check_rankings([RULE], results))
    assert "Cases: 1/2 passed (50%)" in summary
    assert "Rankings: 1/1 passed" in summary
    assert "be_strong_1@jd_backend_engineer: score 40 is outside 50-79" in summary


# --- clients ---

def test_replay_answers_by_system_prompt() -> None:
    client = ReplayLLMClient({CV_SYSTEM_PROMPT: CV_JSON, JOB_SYSTEM_PROMPT: JOB_JSON})
    assert client.complete_json(CV_SYSTEM_PROMPT, "any") == CV_JSON
    assert client.complete_json(JOB_SYSTEM_PROMPT, "any") == JOB_JSON


def test_replay_returns_copies() -> None:
    client = ReplayLLMClient({CV_SYSTEM_PROMPT: CV_JSON})
    client.complete_json(CV_SYSTEM_PROMPT, "")["skills"].clear()
    assert client.complete_json(CV_SYSTEM_PROMPT, "")["skills"]


def test_replay_without_recording_raises() -> None:
    with pytest.raises(ReplayError):
        ReplayLLMClient({}).complete_json(CV_SYSTEM_PROMPT, "text")


def test_budget_refuses_the_call_after_the_limit() -> None:
    client = BudgetLLMClient(FakeLLMClient([{"a": 1}, {"b": 2}]), max_calls=1)
    client.complete_json("s", "u")
    with pytest.raises(BudgetExceeded):
        client.complete_json("s", "u2")
    assert client.calls == 1


def test_budget_counts_failed_calls() -> None:
    client = BudgetLLMClient(FakeLLMClient([LLMResponseError("bad")]), max_calls=1)
    with pytest.raises(LLMResponseError):
        client.complete_json("s", "u")
    assert client.calls == 1


def test_cache_hits_do_not_use_the_budget(tmp_path: Path) -> None:
    budget = BudgetLLMClient(FakeLLMClient([{"a": 1}]), max_calls=1)
    cache = CachedLLMClient(budget, tmp_path)
    cache.complete_json("s", "u")
    assert cache.complete_json("s", "u") == {"a": 1}
    assert budget.calls == 1


def test_recording_keeps_the_last_answer_per_prompt() -> None:
    recorder = RecordingLLMClient(FakeLLMClient([{"bad": 1}, {"good": 1}, {"job": 1}]))
    recorder.complete_json(CV_SYSTEM_PROMPT, "u")
    recorder.complete_json(CV_SYSTEM_PROMPT, "u retry")
    recorder.complete_json(JOB_SYSTEM_PROMPT, "j")
    assert recorder.take() == {CV_SYSTEM_PROMPT: {"good": 1}, JOB_SYSTEM_PROMPT: {"job": 1}}
    assert recorder.take() == {}


# --- running cases and fixtures ---

def test_rejected_case_makes_no_model_call() -> None:
    blog = case(cv="x_blog_post.txt", category="non_cv", outcome="rejected_422", score=None)
    fake = FakeLLMClient()
    outcome = run_case(blog, fake, TODAY)
    assert outcome.passed and outcome.outcome is Outcome.REJECTED
    assert fake.calls == []


def test_scored_case_uses_the_fixed_date() -> None:
    present = dict(CV_JSON, roles=[{"start": "2025-11", "end": "present", "skills": ["Python"]}])
    job = dict(JOB_JSON, min_years=3)
    weak = case(cv="be_weak_2.txt", category="weak", score=[0, 45])
    before = run_case(weak, FakeLLMClient([present, job]), TODAY)
    later = run_case(weak, FakeLLMClient([present, job]), TODAY.replace(year=2027))
    assert before.score is not None and later.score is not None
    assert later.score > before.score


def test_invalid_model_output_is_a_failed_case() -> None:
    bad = LLMResponseError("bad")
    outcome = run_case(case(), FakeLLMClient([bad, bad]), TODAY)
    assert outcome.outcome is None and not outcome.passed
    assert format_table([outcome]).splitlines()[1].split()[-2:] == ["error", "FAIL"]


def test_recordings_store_output_hash_and_prompt_version(tmp_path: Path) -> None:
    strong = case()
    written = save_recordings(
        strong, {CV_SYSTEM_PROMPT: CV_JSON, JOB_SYSTEM_PROMPT: JOB_JSON}, "fake-model", tmp_path
    )
    assert sorted(path.name for path in written) == ["be_strong_1.json", "jd_backend_engineer.json"]
    fixture = load_fixture(fixture_path("be_strong_1.txt", tmp_path))
    assert fixture is not None
    assert fixture.prompt_version == PROMPT_VERSION
    assert fixture.input_sha256 == input_hash(case_inputs(strong)[CV_SYSTEM_PROMPT][1])
    assert replay_client(strong, tmp_path).complete_json(CV_SYSTEM_PROMPT, "") == CV_JSON


def test_stale_fixtures_are_reported(tmp_path: Path) -> None:
    strong = case()
    save_recordings(strong, {CV_SYSTEM_PROMPT: CV_JSON}, "fake-model", tmp_path)
    labels = EvalLabels(today=TODAY, rankings=[], cases=[strong])
    assert stale_fixtures(labels, tmp_path) == []
    path = fixture_path("be_strong_1.txt", tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(PROMPT_VERSION, "v0"), "utf-8")
    assert stale_fixtures(labels, tmp_path) == ["be_strong_1.txt"]


# --- script ---

def settings(tmp_path: Path, key: str = "test-key") -> Settings:
    return Settings(openai_api_key=key, cache_dir=tmp_path / "cache", _env_file=None)  # type: ignore[call-arg]


def run_script(tmp_path: Path, *args: str, inner: CannedClient, key: str = "test-key") -> int:
    argv = ["--extractions", str(tmp_path / "extractions"), *args]
    return main(argv, settings(tmp_path, key), lambda _settings: inner)


def test_replay_without_fixtures_stops(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_script(tmp_path, inner=CannedClient()) == 2
    assert "no recorded extraction" in capsys.readouterr().err


def test_record_requires_real(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run_script(tmp_path, "--record", inner=CannedClient())


def test_real_without_key_stops_before_any_call(tmp_path: Path) -> None:
    inner = CannedClient()
    assert run_script(tmp_path, "--real", inner=inner, key="") == 2
    assert inner.calls == 0


def test_real_stops_when_planned_calls_exceed_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inner = CannedClient()
    assert run_script(tmp_path, "--real", "--max-calls", "3", inner=inner) == 2
    assert inner.calls == 0
    assert "planned calls exceed the budget" in capsys.readouterr().err


def test_retries_beyond_the_plan_still_hit_the_budget(tmp_path: Path) -> None:
    inner = CannedClient(error=LLMResponseError("bad"))
    assert run_script(tmp_path, "--real", "--max-calls", "30", inner=inner) == 2
    assert inner.calls == 30


def test_real_record_then_cache_then_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = CannedClient()
    run_script(tmp_path, "--real", "--record", inner=first)
    recorded = capsys.readouterr().out
    planned = int(recorded.split("cache misses): ")[1].split(";")[0])
    assert first.calls == planned > 0
    assert len(list((tmp_path / "extractions").glob("*.json"))) == planned

    second = CannedClient()
    run_script(tmp_path, "--real", inner=second)
    cached = capsys.readouterr().out
    assert second.calls == 0
    assert "cache misses): 0;" in cached

    run_script(tmp_path, inner=CannedClient())
    replayed = capsys.readouterr().out
    assert replayed in recorded and replayed in cached

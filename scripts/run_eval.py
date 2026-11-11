"""Run the evaluation set and compare every result with its label.

Default: replay recorded extractions from data/eval/extractions (no model calls).
--real:   call the configured model through the disk cache, capped by --max-calls.
--record: with --real, write each extraction to data/eval/extractions.

Exit codes: 0 all cases and rankings pass, 1 some fail, 2 the run could not start or finish.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.config import Settings  # noqa: E402
from app.eval_clients import (  # noqa: E402
    BudgetExceeded,
    BudgetLLMClient,
    RecordingLLMClient,
    ReplayError,
    count_cache_misses,
    replay_client,
    save_recordings,
    stale_fixtures,
)
from app.evaluation import (  # noqa: E402
    EXTRACTIONS_DIR,
    LABELS_PATH,
    CaseResult,
    EvalCase,
    EvalLabels,
    check_rankings,
    format_summary,
    format_table,
    load_labels,
    run_case,
)
from app.extraction import prompt_version  # noqa: E402
from app.llm.cache import CachedLLMClient  # noqa: E402
from app.llm.client import LLMClient  # noqa: E402
from app.llm.openai_client import OpenAIClient  # noqa: E402

DEFAULT_MAX_CALLS = 40
InnerFactory = Callable[[Settings], LLMClient]


class RunStopped(Exception):
    """The run cannot start or was stopped; the message says why."""


def openai_inner(settings: Settings) -> LLMClient:
    return OpenAIClient(api_key=settings.openai_api_key, model=settings.llm_model)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", action="store_true", help="call the model (through the cache)")
    parser.add_argument("--record", action="store_true", help="with --real: write fixtures")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--markdown", action="store_true", help="print the table as Markdown")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--extractions", type=Path, default=EXTRACTIONS_DIR)
    args = parser.parse_args(argv)
    if args.record and not args.real:
        parser.error("--record needs --real")
    return args


def run_replay(labels: EvalLabels, extractions: Path) -> list[CaseResult]:
    stale = stale_fixtures(labels, extractions)
    if stale:
        raise RunStopped(f"stale fixtures (re-record with --real --record): {', '.join(stale)}")
    try:
        return [run_case(case, replay_client(case, extractions), labels.today) for case in labels.cases]
    except ReplayError as exc:
        raise RunStopped(str(exc)) from exc


def cache_dir(settings: Settings) -> Path:
    path = settings.cache_dir
    return path if path.is_absolute() else PROJECT_DIR / path


def run_real(
    labels: EvalLabels, args: argparse.Namespace, settings: Settings, make_inner: InnerFactory
) -> list[CaseResult]:
    if not settings.openai_api_key:
        raise RunStopped("OPENAI_API_KEY is not configured; --real needs it")
    budget = BudgetLLMClient(make_inner(settings), args.max_calls)
    cache = CachedLLMClient(budget, cache_dir(settings), key_salt=prompt_version)
    misses = count_cache_misses(labels, cache)
    print(f"Planned model calls (cache misses): {misses}; budget: {args.max_calls}")
    if misses > args.max_calls:
        raise RunStopped("planned calls exceed the budget; nothing was called")
    recorder = RecordingLLMClient(cache)
    try:
        return [_run_and_record(case, recorder, labels, args) for case in labels.cases]
    except BudgetExceeded as exc:
        raise RunStopped(str(exc)) from exc
    finally:
        print(f"Model calls made: {budget.calls}")


def _run_and_record(
    case: EvalCase, recorder: RecordingLLMClient, labels: EvalLabels, args: argparse.Namespace
) -> CaseResult:
    result = run_case(case, recorder, labels.today)
    outputs = recorder.take()
    if args.record:
        save_recordings(case, outputs, recorder.model, args.extractions)
    return result


def report(labels: EvalLabels, results: list[CaseResult], markdown: bool) -> bool:
    rankings = check_rankings(labels.rankings, results)
    print(format_table(results, markdown))
    print()
    print(format_summary(results, rankings))
    return all(result.passed for result in results) and all(check.passed for check in rankings)


def main(
    argv: list[str] | None = None,
    settings: Settings | None = None,
    make_inner: InnerFactory = openai_inner,
) -> int:
    args = parse_args(argv)
    labels = load_labels(args.labels)
    try:
        if args.real:
            settings = settings or Settings(_env_file=PROJECT_DIR / ".env")  # type: ignore[call-arg]
            results = run_real(labels, args, settings, make_inner)
        else:
            results = run_replay(labels, args.extractions)
    except RunStopped as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2
    return 0 if report(labels, results, args.markdown) else 1


if __name__ == "__main__":
    sys.exit(main())

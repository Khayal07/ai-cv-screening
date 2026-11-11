"""Model clients for the evaluation: replay recorded output, cap real calls, record output."""

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.evaluation import CVS_DIR, EXTRACTIONS_DIR, JOBS_DIR, EvalCase, EvalLabels, read_text
from app.extraction import CV_SYSTEM_PROMPT, JOB_SYSTEM_PROMPT, MAX_INPUT_CHARS, prompt_version
from app.llm.cache import CachedLLMClient
from app.llm.client import LLMClient
from app.parsing.anonymize import strip_personal_data
from app.parsing.cv_check import looks_like_cv


class ReplayError(Exception):
    """Raised when replay has no recorded extraction for a prompt."""


class BudgetExceeded(Exception):
    """Raised instead of making a model call beyond the allowed number."""


class Fixture(BaseModel):
    """One recorded extraction. Stores only the model output and a hash of its input."""

    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    model: str
    input_sha256: str
    output: dict[str, Any]


def cv_model_input(cv_text: str) -> str:
    """The exact text the CV extraction prompt receives."""
    return strip_personal_data(cv_text)[:MAX_INPUT_CHARS]


def job_model_input(job_text: str) -> str:
    return job_text[:MAX_INPUT_CHARS]


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fixture_path(file_name: str, directory: Path = EXTRACTIONS_DIR) -> Path:
    return directory / f"{Path(file_name).stem}.json"


def load_fixture(path: Path) -> Fixture | None:
    if not path.exists():
        return None
    return Fixture.model_validate_json(path.read_text(encoding="utf-8"))


def write_fixture(path: Path, fixture: Fixture) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture.model_dump(), indent=2) + "\n", encoding="utf-8")


class ReplayLLMClient:
    """Answers each prompt with its recorded output; never calls a model."""

    def __init__(self, outputs: dict[str, dict[str, Any]]) -> None:
        self._outputs = outputs

    @property
    def model(self) -> str:
        return "replay"

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if system not in self._outputs:
            raise ReplayError("no recorded extraction; run scripts/run_eval.py --real --record")
        return copy.deepcopy(self._outputs[system])


class BudgetLLMClient:
    """Refuses to make more than `max_calls` calls to the wrapped client."""

    def __init__(self, inner: LLMClient, max_calls: int) -> None:
        self._inner = inner
        self.max_calls = max_calls
        self.calls = 0

    @property
    def model(self) -> str:
        return self._inner.model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"model call budget of {self.max_calls} reached")
        self.calls += 1
        return self._inner.complete_json(system, user)


class RecordingLLMClient:
    """Remembers the last output per system prompt (a retry overwrites the failed answer)."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self._outputs: dict[str, dict[str, Any]] = {}

    @property
    def model(self) -> str:
        return self._inner.model

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        result = self._inner.complete_json(system, user)
        self._outputs[system] = copy.deepcopy(result)
        return result

    def take(self) -> dict[str, dict[str, Any]]:
        outputs, self._outputs = self._outputs, {}
        return outputs


def case_inputs(
    case: EvalCase, cvs_dir: Path = CVS_DIR, jobs_dir: Path = JOBS_DIR
) -> dict[str, tuple[str, str]]:
    """System prompt -> (file name, exact model input) for both extractions of a case."""
    cv_text, job_text = read_text(cvs_dir / case.cv), read_text(jobs_dir / case.job)
    return {
        CV_SYSTEM_PROMPT: (case.cv, cv_model_input(cv_text)),
        JOB_SYSTEM_PROMPT: (case.job, job_model_input(job_text)),
    }


def planned_prompts(labels: EvalLabels, cvs_dir: Path = CVS_DIR) -> set[tuple[str, str]]:
    """Every (system, user) prompt a run needs. Rejected documents need none."""
    prompts: set[tuple[str, str]] = set()
    for case in labels.cases:
        if looks_like_cv(read_text(cvs_dir / case.cv)).is_cv:
            for system, (_, text) in case_inputs(case, cvs_dir).items():
                prompts.add((system, text))
    return prompts


def count_cache_misses(labels: EvalLabels, cache: CachedLLMClient, cvs_dir: Path = CVS_DIR) -> int:
    return sum(not cache.has(system, user) for system, user in planned_prompts(labels, cvs_dir))


def save_recordings(
    case: EvalCase, outputs: dict[str, dict[str, Any]], model: str, directory: Path
) -> list[Path]:
    written: list[Path] = []
    for system, (file_name, text) in case_inputs(case).items():
        if system in outputs:
            fixture = Fixture(
                prompt_version=prompt_version(system),
                model=model,
                input_sha256=input_hash(text),
                output=outputs[system],
            )
            write_fixture(path := fixture_path(file_name, directory), fixture)
            written.append(path)
    return written


def replay_client(case: EvalCase, directory: Path = EXTRACTIONS_DIR) -> ReplayLLMClient:
    outputs: dict[str, dict[str, Any]] = {}
    for system, file_name in ((CV_SYSTEM_PROMPT, case.cv), (JOB_SYSTEM_PROMPT, case.job)):
        fixture = load_fixture(fixture_path(file_name, directory))
        if fixture is not None:
            outputs[system] = fixture.output
    return ReplayLLMClient(outputs)


def stale_fixtures(labels: EvalLabels, directory: Path = EXTRACTIONS_DIR) -> list[str]:
    """Fixtures recorded with another version of their prompt, or for text changed since."""
    stale: list[str] = []
    for case in labels.cases:
        for system, (file_name, text) in case_inputs(case).items():
            fixture = load_fixture(fixture_path(file_name, directory))
            if fixture is None:
                continue
            version_changed = fixture.prompt_version != prompt_version(system)
            if version_changed or fixture.input_sha256 != input_hash(text):
                stale.append(file_name)
    return sorted(set(stale))

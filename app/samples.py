"""Read-only access to the synthetic sample CVs and job descriptions."""

from pathlib import Path

from pydantic import BaseModel

CV_PREFIX = "cv_"
JOB_PREFIX = "jd_"


class Sample(BaseModel):
    id: str
    label: str
    text: str


class SamplesResponse(BaseModel):
    cvs: list[Sample]
    jobs: list[Sample]


def load_samples(directory: Path) -> SamplesResponse:
    return SamplesResponse(
        cvs=_load_group(directory, CV_PREFIX),
        jobs=_load_group(directory, JOB_PREFIX),
    )


def _load_group(directory: Path, prefix: str) -> list[Sample]:
    if not directory.is_dir():
        return []
    files = sorted(directory.glob(f"{prefix}*.txt"))
    return [_to_sample(path, prefix) for path in files if path.is_file()]


def _to_sample(path: Path, prefix: str) -> Sample:
    return Sample(
        id=path.stem,
        label=make_label(path.stem, prefix),
        text=path.read_text(encoding="utf-8"),
    )


def make_label(stem: str, prefix: str) -> str:
    words = stem.removeprefix(prefix).replace("_", " ").strip()
    return words.capitalize() or stem

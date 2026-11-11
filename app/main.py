from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.extraction import ExtractionError, prompt_version
from app.llm.cache import CachedLLMClient
from app.llm.client import LLMClient
from app.llm.daily_cap import DailyCapLLMClient, DailyLimitReached
from app.llm.openai_client import OpenAIClient
from app.models import MatchResult
from app.parsing.cv_check import NotACVError
from app.parsing.pdf import NoExtractableTextError, PDFParseError, extract_pdf_text
from app.ratelimit import (
    DAY,
    MINUTE,
    RateLimiter,
    RateLimitExceeded,
    Window,
    client_ip,
    limit_message,
)
from app.samples import SamplesResponse, load_samples
from app.service import screen

MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_TEXT_CHARS = 50_000
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CV Screening", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MatchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    cv_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    job_text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


DAILY_LIMIT_MESSAGE = "Daily demo limit reached. Try again after 00:00 UTC."


@lru_cache
def _build_client(api_key: str, model: str, cache_dir: Path, daily_limit: int) -> LLMClient:
    """Cache outside the daily cap, so only real provider calls count toward it."""
    provider = OpenAIClient(api_key=api_key, model=model)
    capped = DailyCapLLMClient(provider, daily_limit, usage_dir=cache_dir / "usage")
    return CachedLLMClient(capped, cache_dir, key_salt=prompt_version)


def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> LLMClient:
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured")
    return _build_client(
        settings.openai_api_key,
        settings.llm_model,
        settings.cache_dir,
        settings.daily_model_call_limit,
    )


LLMDep = Annotated[LLMClient, Depends(get_llm_client)]


@lru_cache
def get_rate_limiter() -> RateLimiter:
    settings = get_settings()
    return RateLimiter(
        [Window(MINUTE, settings.rate_limit_per_minute), Window(DAY, settings.rate_limit_per_day)]
    )


def enforce_rate_limit(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    peer = request.client.host if request.client else None
    forwarded = request.headers.get("x-forwarded-for")
    limiter.check(client_ip(peer, forwarded, settings.trusted_proxy_hops))


RateLimited = [Depends(enforce_rate_limit)]


@app.exception_handler(RateLimitExceeded)
def handle_rate_limit(_request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": limit_message(exc.retry_after)},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(ExtractionError)
def handle_extraction_error(_request: Request, _exc: ExtractionError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "model returned invalid output"})


@app.exception_handler(NotACVError)
def handle_not_a_cv(_request: Request, exc: NotACVError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": "does not look like a CV", "reasons": exc.reasons}
    )


@app.exception_handler(DailyLimitReached)
def handle_daily_limit(_request: Request, exc: DailyLimitReached) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": DAILY_LIMIT_MESSAGE},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(OpenAIError)
def handle_provider_error(_request: Request, _exc: OpenAIError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "model provider error"})


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/samples")
def samples(settings: Annotated[Settings, Depends(get_settings)]) -> SamplesResponse:
    return load_samples(settings.samples_dir)


@app.post("/match", dependencies=RateLimited)
def match(body: MatchRequest, llm: LLMDep) -> MatchResult:
    return screen(body.cv_text, body.job_text, llm)


@app.post("/match/pdf", dependencies=RateLimited)
def match_pdf(
    cv_file: Annotated[UploadFile, File()],
    job_text: Annotated[str, Form(min_length=1, max_length=MAX_TEXT_CHARS)],
    llm: LLMDep,
) -> MatchResult:
    cv_text = _read_pdf_upload(cv_file)
    return screen(cv_text, job_text, llm)


def _read_pdf_upload(upload: UploadFile) -> str:
    if not _is_pdf(upload):
        raise HTTPException(status_code=415, detail="CV must be a PDF file")
    data = upload.file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than 5 MB")
    try:
        return extract_pdf_text(data)
    except NoExtractableTextError as exc:
        raise HTTPException(status_code=422, detail="no extractable text") from exc
    except PDFParseError as exc:
        raise HTTPException(status_code=422, detail="invalid PDF file") from exc


def _is_pdf(upload: UploadFile) -> bool:
    name_ok = (upload.filename or "").lower().endswith(".pdf")
    return name_ok or upload.content_type == "application/pdf"

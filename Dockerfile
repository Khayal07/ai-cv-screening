FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_DIR=/app/.cache/llm

WORKDIR /app

COPY pyproject.toml ./
COPY app ./app
COPY data/synthetic ./data/synthetic
RUN pip install . \
    && useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache/llm \
    && chown -R appuser /app/.cache

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

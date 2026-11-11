FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=production \
    CACHE_DIR=/app/.cache/llm \
    PORT=8000

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request as r; r.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# One worker: rate limits and the daily cap are counted in this process.
# --no-proxy-headers: the app reads X-Forwarded-For itself (TRUSTED_PROXY_HOPS).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --no-proxy-headers --no-server-header"]

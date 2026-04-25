# syntax=docker/dockerfile:1.6
# ----- Stage 1: builder -----
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Build deps required to compile wheels (numpy, xgboost, asyncpg, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /build/requirements.txt

# Install all Python deps into an isolated prefix we can copy into the runtime stage.
RUN pip install --upgrade pip \
    && pip install --prefix=/install -r /build/requirements.txt

# ----- Stage 2: runtime -----
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/usr/local/bin:$PATH

# curl is required by the HEALTHCHECK; libpq5 is the runtime for asyncpg/psycopg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home /app --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copy the prebuilt site-packages from the builder.
COPY --from=builder /install /usr/local

# Copy ONLY runtime artifacts. No tests, no ml/data, no .env, no frontend.
COPY --chown=appuser:appuser backend/ /app/backend/
COPY --chown=appuser:appuser ml/artifacts/ /app/ml/artifacts/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

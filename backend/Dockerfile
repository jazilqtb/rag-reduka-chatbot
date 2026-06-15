# ============================================================================
# UTBK Tutor RAG Backend - Multi-stage Dockerfile
# ----------------------------------------------------------------------------
# Stage 1 (builder) : full image with build tools, installs Python packages
# Stage 2 (runtime) : slim image with only what's needed to run
#
# This separation keeps the final image small (~400 MB instead of ~1 GB).
# ============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System build deps needed for psycopg2 (libpq-dev), pymupdf, chromadb.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer cache.
# Subsequent code changes won't trigger a pip install if requirements.txt
# is unchanged.
COPY requirements.txt .

# Install to user site so we can copy /root/.local to the runtime stage.
RUN pip install --user --no-cache-dir -r requirements.txt


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Runtime deps only:
#   - libpq5 : runtime library for psycopg2
#   - curl   : for healthcheck (optional, can be removed if unused)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the builder stage.
COPY --from=builder /root/.local /root/.local

# Copy application code.
# .dockerignore should exclude: .venv, __pycache__, tests/, notebooks/, data/
COPY . .

# Ensure data directories exist (will be overlaid by docker-compose volume mount).
RUN mkdir -p /app/data/raw_docs \
             /app/data/vector_store \
             /app/data/debug

# Document the port (docker-compose will publish it).
EXPOSE 8000

# Default command. Override in docker-compose for dev (e.g., add --reload).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
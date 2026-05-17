# ============================================================
# CDSS Platform – Dockerfile
# Multi-stage build for production
# ============================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies layer ──
FROM base AS deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Production image ──
FROM deps AS production

# Copy application code
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY data/guidelines/ ./data/guidelines/
COPY scripts/ ./scripts/
COPY run.py .
COPY .env.example .env

# Create directories
RUN mkdir -p ./data/chroma_db ./logs

# Non-root user for security
RUN groupadd -r cdss && useradd -r -g cdss cdss
RUN chown -R cdss:cdss /app
USER cdss

EXPOSE 8000

# Pre-download sentence transformer model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["python", "run.py"]

# ── VeritAI — HuggingFace Spaces Dockerfile ──────────────────────────────────
# Python 3.12 · FastAPI · CPU-only PyTorch · SQLite
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement) ─────────────────────────────
RUN useradd -m -u 1000 veritai
WORKDIR /app

# ── Copy requirements first (layer caching) ──────────────────────────────────
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────────────────────
# CPU-only torch keeps image size manageable on free tier
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Download spaCy model ──────────────────────────────────────────────────────
RUN python -m spacy download en_core_web_sm

# ── Copy application code ─────────────────────────────────────────────────────
COPY --chown=veritai:veritai . .

# ── Create persistent data directory for SQLite ───────────────────────────────
# HF Spaces mounts /data as persistent storage — point DB here
RUN mkdir -p /data && chown veritai:veritai /data

# ── Switch to non-root user ───────────────────────────────────────────────────
USER veritai

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    DB_PATH=/data/veritai.db \
    MODEL_PATH=/app/model/bert_fake_news.pt \
    TOKENIZER_PATH=/app/model/tokenizer

# ── HuggingFace Spaces runs on port 7860 ─────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
# Workers=1 on CPU free tier — model is loaded once in memory
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

# ─────────────────────────────────────────────────────────────────────────────
# realtime-subtitle  Dockerfile
# CPU-only build by default.  For GPU, see docker-compose.yml override.
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System-level audio + build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        portaudio19-dev \
        libsndfile1 \
        ffmpeg \
        gcc \
        g++ \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (layer cache friendly) ──────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Optional: pre-install translation & speaker packages ─────────────────────
# Uncomment to bake them into the image:
# RUN pip install --no-cache-dir argostranslate resemblyzer

# ── Copy source ───────────────────────────────────────────────────────────────
COPY . .

# ── Argostranslate model pre-download (optional) ─────────────────────────────
# RUN python scripts/install_argos_packages.py

# ── Whisper model pre-download (optional, saves startup time) ─────────────────
# RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# Default env (overridable at runtime)
ENV LANGUAGE=vi \
    TARGET_LANGUAGE=en \
    MODEL_SIZE=base \
    DEVICE=cpu \
    COMPUTE_TYPE=int8 \
    REALTIME_MODEL_SIZE=tiny \
    SPEAKER_PROVIDER=local \
    TRANSLATION_PROVIDER=argos \
    LOG_LEVEL=INFO

# Expose no ports — this is a pure CLI application
CMD ["python", "-m", "app.main"]

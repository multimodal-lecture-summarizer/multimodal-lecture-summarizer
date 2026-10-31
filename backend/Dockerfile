# Backend API image. Build from the repo root: docker build -f backend/Dockerfile .
# The API imports the summarizer/checkpoint modules from ai_workers/, so that code is copied too
# (without the GPU dependencies).
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ffprobe validates uploaded videos.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend backend
COPY ai_workers/__init__.py ai_workers/__init__.py
COPY ai_workers/core ai_workers/core
COPY ai_workers/modules/__init__.py ai_workers/modules/__init__.py
COPY ai_workers/modules/summarization ai_workers/modules/summarization
COPY ai_workers/modules/fusion ai_workers/modules/fusion

# main.py serves mock storage from ../storage, i.e. /app/storage.
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

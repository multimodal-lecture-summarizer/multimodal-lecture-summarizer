# syntax=docker/dockerfile:1
# One image for the whole system: React UI (nginx), FastAPI backend, Celery AI worker,
# PostgreSQL and Redis, all run by supervisord. Build from the repo root:
#
#   docker build -f docker/all-in-one/Dockerfile -t multimodal-lecture-summarizer:all-in-one .
#
# Secrets are never baked in: pass backend/.env at runtime (--env-file). See
# docker-compose.all-in-one.yml for a ready-to-run GPU setup.

# --- Frontend build ------------------------------------------------------------------------
FROM node:22-alpine AS frontend-build
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# VITE_* values are baked in at build time: the URLs the *browser* uses to reach the API.
ARG VITE_API_BASE_URL=http://localhost:8000/api/v1
ARG VITE_BACKEND_URL=http://localhost:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_BACKEND_URL=$VITE_BACKEND_URL
RUN npm run build

# --- Runtime -------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

# The first layers are identical to ai_workers/Dockerfile so an existing build cache is reused.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY ai_workers/requirements.txt ai_workers/requirements.txt
RUN pip install -r ai_workers/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Faster-Whisper (CTranslate2) loads cuDNN/cuBLAS by name; use the copies installed with torch.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib

# Database, broker, web server and process manager. The default PostgreSQL cluster is skipped:
# the entrypoint creates one on the data volume instead.
RUN mkdir -p /etc/postgresql-common \
    && echo "create_main_cluster = false" > /etc/postgresql-common/createcluster.conf \
    && apt-get update && apt-get install -y --no-install-recommends \
       postgresql redis-server nginx supervisor \
    && rm -rf /var/lib/apt/lists/* /etc/nginx/sites-enabled/default

COPY ai_workers ai_workers
COPY backend backend
# main.py serves mock storage from /app/storage; the backend also writes under backend/storage.
RUN rm -rf /app/backend/storage && ln -s /app/storage /app/backend/storage

COPY --from=frontend-build /app/dist /usr/share/nginx/html
COPY docker/all-in-one/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/all-in-one/supervisord.conf /etc/supervisor/supervisord.conf
COPY docker/all-in-one/entrypoint.sh /usr/local/bin/entrypoint.sh
# Checked out on Windows these files may carry CRLF line endings, which break shell scripts.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh /etc/supervisor/supervisord.conf /etc/nginx/conf.d/default.conf \
    && chmod +x /usr/local/bin/entrypoint.sh

ENV WORKER_QUEUES=gpu,celery \
    CUBLAS_WORKSPACE_CONFIG=":4096:8"

VOLUME ["/var/lib/postgresql/data", "/var/lib/redis", "/app/storage", "/app/cache", "/app/outputs"]
EXPOSE 80 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=5)" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/supervisord.conf"]

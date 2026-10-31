# AI worker image. Build from the repo root: docker build -f ai_workers/Dockerfile .
# Runs on CPU-only hosts and on NVIDIA hosts (docker run --gpus all). CUDA/cuDNN come from
# the torch wheels; the host only needs the NVIDIA driver + NVIDIA Container Toolkit.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Backend requirements are needed because RAG indexing reads the database through backend/app.
# Two layers so a backend requirements change does not reinstall torch.
COPY ai_workers/requirements.txt ai_workers/requirements.txt
RUN pip install -r ai_workers/requirements.txt
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Faster-Whisper (CTranslate2) loads cuDNN/cuBLAS by name; use the copies installed with torch.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib

COPY ai_workers ai_workers
COPY backend backend

# "celery" = default queue (CPU jobs + RAG indexing). GPU hosts add "gpu", see docker-compose.gpu.yml.
ENV WORKER_QUEUES=celery
CMD ["sh", "-c", "exec celery -A ai_workers.core.celery_app worker --loglevel=info --pool=solo --concurrency=1 -Q \"$WORKER_QUEUES\""]

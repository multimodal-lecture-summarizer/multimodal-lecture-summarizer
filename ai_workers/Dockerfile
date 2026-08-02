FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./ai_workers/requirements.txt
RUN pip install --no-cache-dir -r ./ai_workers/requirements.txt

# Copy application code
COPY . ./ai_workers

CMD ["celery", "-A", "ai_workers.core.celery_app", "worker", "--loglevel=info", "--pool=solo", "--concurrency=1"]

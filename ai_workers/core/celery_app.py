"""Celery application configuration — kết nối Redis broker."""

import sys
import os
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from celery import Celery

from ai_workers.core.config import worker_settings

app = Celery(
    "ai_workers",
    broker=worker_settings.CELERY_BROKER_URL,
    backend=worker_settings.CELERY_RESULT_BACKEND,
)
celery_app = app

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # 1 task at a time per worker for GPU tasks
    broker_pool_limit=1,
    redis_max_connections=2,
    worker_gossip=False,
    worker_mingle=False,
    worker_send_task_events=False,
    task_send_sent_event=False,
)

# Auto-discover tasks in ai_workers package
app.autodiscover_tasks(["ai_workers"])

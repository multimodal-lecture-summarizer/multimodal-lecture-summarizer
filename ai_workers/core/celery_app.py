"""Celery application configuration — kết nối Redis broker."""

from __future__ import annotations

from celery import Celery

from ai_workers.core.config import worker_settings

app = Celery(
    "ai_workers",
    broker=worker_settings.CELERY_BROKER_URL,
    backend=worker_settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # 1 task at a time per worker for GPU tasks
)

# Auto-discover tasks in ai_workers package
app.autodiscover_tasks(["ai_workers"])

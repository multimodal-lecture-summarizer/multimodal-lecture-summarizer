from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_pool_limit=1,
    redis_max_connections=2,
    worker_gossip=False,
    worker_mingle=False,
    worker_send_task_events=False,
    task_send_sent_event=False,
)

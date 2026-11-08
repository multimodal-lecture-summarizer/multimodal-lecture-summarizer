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
    broker_pool_limit=0,
    redis_max_connections=2,
    # Release dead sockets promptly; the managed Redis plan caps concurrent clients.
    broker_transport_options={"socket_keepalive": True, "health_check_interval": 30},
    redis_socket_keepalive=True,
    redis_backend_health_check_interval=30,
    redis_retry_on_timeout=True,
    worker_gossip=False,
    worker_mingle=False,
    worker_send_task_events=False,
    task_send_sent_event=False,
)

"""Celery application configuration — kết nối Redis broker."""

import sys
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from celery import Celery
from celery.signals import worker_init


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
    worker_pool="solo",
    worker_concurrency=1,
)

@worker_init.connect
def validate_florence_runtime_on_startup(**_kwargs):
    """Stop worker startup before it can accept jobs when assets drift."""
    from pathlib import Path

    from ai_workers.modules.visual_v2.florence_runtime import (
        resolve_florence_runtime,
        verify_florence_model,
    )

    try:
        runtime = resolve_florence_runtime(worker_settings.FLORENCE_DEVICE)
        model_dir = Path(__file__).resolve().parents[1] / "modules" / "visual_v2" / "florence2_vendor"
        verify_florence_model(model_dir)
    except Exception as exc:
        raise SystemExit(f"Florence-2 startup validation failed: {exc}") from exc

    print(
        f"[Startup] Florence-2 runtime verified: "
        f"{runtime.device}/float32/eager, all asset SHA-256 checks OK."
    )


# Auto-discover tasks in ai_workers package
app.autodiscover_tasks(["ai_workers"])

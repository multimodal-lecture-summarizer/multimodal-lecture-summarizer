"""Celery application configuration — kết nối Redis broker."""

import os
import sys
import time
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit

_STARTUP_T0 = time.perf_counter()

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


def _startup_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = time.perf_counter() - _STARTUP_T0
    print(f"[CeleryWorker Startup][{timestamp}][+{elapsed:.1f}s] {message}", flush=True)


def _safe_url(raw_url: str) -> str:
    try:
        parts = urlsplit(raw_url)
        if parts.password is None:
            return raw_url
        username = parts.username or ""
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{username}:***@{host}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return "<unprintable-url>"


_startup_log("Importing Celery app module.")
_startup_log(f"Process pid={os.getpid()} python={sys.version.split()[0]} cwd={os.getcwd()}")
_startup_log(f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG')}")

from celery import Celery
from celery.signals import (
    after_setup_logger,
    celeryd_after_setup,
    celeryd_init,
    import_modules,
    setup_logging,
    task_postrun,
    worker_init,
    worker_ready,
    worker_shutdown,
)


from ai_workers.core.config import worker_settings

_startup_log("Worker settings loaded.")
_startup_log(f"Broker={_safe_url(worker_settings.CELERY_BROKER_URL)}")
_startup_log(f"Result backend={_safe_url(worker_settings.CELERY_RESULT_BACKEND)}")
_startup_log(
    "AI runtime config: "
    f"FLORENCE_DEVICE={worker_settings.FLORENCE_DEVICE}, "
    f"ENABLE_FLORENCE_CAPTIONING={worker_settings.ENABLE_FLORENCE_CAPTIONING}, "
    f"FLORENCE_MAX_CAPTIONS={worker_settings.FLORENCE_MAX_CAPTIONS}, "
    f"PROCESS_MIN_AVAILABLE_MEMORY_MB={worker_settings.PROCESS_MIN_AVAILABLE_MEMORY_MB}, "
    f"PROCESS_SOFT_MIN_AVAILABLE_MEMORY_MB={worker_settings.PROCESS_SOFT_MIN_AVAILABLE_MEMORY_MB}, "
    f"PROCESS_MEMORY_RETRY_SECONDS={worker_settings.PROCESS_MEMORY_RETRY_SECONDS}, "
    f"FLORENCE_MIN_AVAILABLE_MEMORY_MB={worker_settings.FLORENCE_MIN_AVAILABLE_MEMORY_MB}, "
    f"FLORENCE_MIN_AVAILABLE_VRAM_MB={worker_settings.FLORENCE_MIN_AVAILABLE_VRAM_MB}, "
    f"SEMANTIC_CLIP_DEVICE={worker_settings.SEMANTIC_CLIP_DEVICE}, "
    f"PADDLEOCR_USE_GPU={worker_settings.PADDLEOCR_USE_GPU}"
)

app = Celery(
    "ai_workers",
    broker=worker_settings.CELERY_BROKER_URL,
    backend=worker_settings.CELERY_RESULT_BACKEND,
)
celery_app = app
_startup_log("Celery app object created.")

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
_startup_log(
    "Celery config applied: "
    f"pool={app.conf.worker_pool}, concurrency={app.conf.worker_concurrency}, "
    f"prefetch={app.conf.worker_prefetch_multiplier}, "
    f"acks_late={app.conf.task_acks_late}, timezone={app.conf.timezone}"
)


@setup_logging.connect
def log_setup_logging(**_kwargs):
    _startup_log("Signal setup_logging received. Celery is configuring logging.")


@after_setup_logger.connect
def log_after_setup_logger(**_kwargs):
    _startup_log("Signal after_setup_logger received. Celery root logger is ready.")


@celeryd_init.connect
def log_celeryd_init(sender=None, conf=None, **_kwargs):
    hostname = sender or "<unknown-hostname>"
    pool = getattr(conf, "worker_pool", "<unknown-pool>") if conf is not None else "<unknown-pool>"
    concurrency = getattr(conf, "worker_concurrency", "<unknown-concurrency>") if conf is not None else "<unknown-concurrency>"
    _startup_log(
        f"Signal celeryd_init received for {hostname}. "
        f"pool={pool}, concurrency={concurrency}."
    )


@import_modules.connect
def log_import_modules(**_kwargs):
    _startup_log("Signal import_modules received. Celery is importing task modules.")


@celeryd_after_setup.connect
def log_celeryd_after_setup(sender=None, instance=None, **_kwargs):
    hostname = sender or "<unknown-hostname>"
    queues = []
    try:
        queues = [queue.name for queue in instance.app.amqp.queues.values()]
    except Exception:
        queues = []
    queue_text = ", ".join(queues) if queues else "<unavailable>"
    _startup_log(f"Signal celeryd_after_setup received for {hostname}. queues={queue_text}.")


@worker_init.connect
def validate_florence_runtime_on_startup(**_kwargs):
    """Stop worker startup before it can accept jobs when assets drift."""
    from pathlib import Path

    from ai_workers.modules.visual_v2.florence_runtime import (
        FlorenceResourceError,
        assert_florence_cuda_memory_available,
        assert_florence_memory_available,
        get_available_memory_mb,
        get_cuda_memory_mb,
        resolve_florence_runtime,
        verify_florence_model,
    )

    _startup_log("worker_init received. Starting Florence-2 startup validation.")
    try:
        available_mb = get_available_memory_mb()
        if available_mb is None:
            _startup_log("System RAM check: unavailable on this host.")
        else:
            _startup_log(f"System RAM available before validation: {available_mb} MB.")

        _startup_log(f"Resolving Florence-2 runtime for FLORENCE_DEVICE={worker_settings.FLORENCE_DEVICE}.")
        runtime = resolve_florence_runtime(worker_settings.FLORENCE_DEVICE)
        _startup_log(
            f"Florence-2 runtime resolved: device={runtime.device}, "
            f"dtype={runtime.dtype}, attention={runtime.attention_implementation}."
        )
        if worker_settings.FLORENCE_DEVICE.strip().lower() != runtime.device:
            _startup_log(
                f"Florence-2 device fallback applied: requested={worker_settings.FLORENCE_DEVICE}, "
                f"effective={runtime.device}."
            )

        try:
            checked_ram = assert_florence_memory_available(
                worker_settings.FLORENCE_MIN_AVAILABLE_MEMORY_MB
            )
            if checked_ram is not None:
                _startup_log(
                    f"RAM guard passed: {checked_ram} MB available "
                    f">= {worker_settings.FLORENCE_MIN_AVAILABLE_MEMORY_MB} MB."
                )
        except FlorenceResourceError as exc:
            _startup_log(f"RAM guard warning: {exc}")

        cuda_memory = get_cuda_memory_mb(runtime.device) if runtime.device == "cuda" else None
        if cuda_memory is not None:
            free_mb, total_mb = cuda_memory
            _startup_log(f"CUDA VRAM before guard: {free_mb}/{total_mb} MB free.")

        try:
            checked_vram = assert_florence_cuda_memory_available(
                runtime,
                worker_settings.FLORENCE_MIN_AVAILABLE_VRAM_MB,
            )
            if checked_vram is not None:
                free_mb, total_mb = checked_vram
                _startup_log(
                    f"CUDA VRAM guard passed: {free_mb}/{total_mb} MB free "
                    f">= {worker_settings.FLORENCE_MIN_AVAILABLE_VRAM_MB} MB."
                )
        except FlorenceResourceError as exc:
            _startup_log(f"CUDA VRAM guard warning: {exc}")

        model_dir = Path(__file__).resolve().parents[1] / "modules" / "visual_v2" / "florence2_vendor"
        _startup_log(f"Verifying Florence-2 assets in {model_dir}.")
        verify_florence_model(model_dir)
        _startup_log("Florence-2 asset SHA-256 validation passed.")
    except Exception as exc:
        _startup_log(f"Florence-2 startup validation failed: {exc}")
        raise SystemExit(f"Florence-2 startup validation failed: {exc}") from exc

    _startup_log(
        f"Florence-2 runtime verified: "
        f"{runtime.device}/float32/eager, all asset SHA-256 checks OK."
    )


@worker_ready.connect
def log_worker_ready(sender=None, **_kwargs):
    hostname = getattr(sender, "hostname", "<unknown-hostname>")
    _startup_log(f"Signal worker_ready received for {hostname}. Celery worker is ready.")


@worker_shutdown.connect
def log_worker_shutdown(sender=None, **_kwargs):
    hostname = getattr(sender, "hostname", "<unknown-hostname>")
    _startup_log(f"Signal worker_shutdown received for {hostname}.")


@task_postrun.connect
def cleanup_after_task(task_id=None, task=None, state=None, **_kwargs):
    task_name = getattr(task, "name", "<unknown-task>")
    if not str(task_name).startswith("ai_workers."):
        return
    from ai_workers.core.resource_cleanup import release_worker_resources

    release_worker_resources(f"Celery task_postrun {task_name}[{task_id}] state={state}")


# Auto-discover tasks in ai_workers package
_startup_log("Autodiscovering ai_workers tasks.")
app.autodiscover_tasks(["ai_workers"])
_startup_log("Celery app module import complete; worker can continue booting.")

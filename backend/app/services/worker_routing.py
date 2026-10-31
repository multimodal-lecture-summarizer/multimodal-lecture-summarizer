"""Pick the Celery queue for a video job: remote GPU worker when one is online, local CPU otherwise."""

import logging

logger = logging.getLogger(__name__)

# Remote GPU workers (Colab, GPU Docker host) consume this queue: `celery worker -Q gpu`.
GPU_QUEUE = "gpu"
# Celery's default queue, consumed by the local CPU worker and by RAG indexing.
CPU_QUEUE = "celery"


def choose_video_queue(celery_app, video_path: str, timeout: float = 1.0) -> str:
    """Return GPU_QUEUE only when the video is reachable over HTTP and a worker listens on it.

    A remote worker cannot open a path on this machine's disk (mock R2 storage), so those
    jobs always stay on the local CPU worker.
    """
    if not video_path.startswith(("http://", "https://")):
        return CPU_QUEUE

    try:
        active = celery_app.control.inspect(timeout=timeout).active_queues() or {}
    except Exception as exc:
        logger.warning("Could not inspect Celery workers, using CPU queue: %s", exc)
        return CPU_QUEUE

    for worker_name, queues in active.items():
        if any(queue.get("name") == GPU_QUEUE for queue in queues or []):
            logger.info("GPU worker %s is online, routing video job to '%s'.", worker_name, GPU_QUEUE)
            return GPU_QUEUE
    return CPU_QUEUE

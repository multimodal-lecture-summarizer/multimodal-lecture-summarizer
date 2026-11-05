"""Pick the Celery queue for a video job: remote GPU worker when one is online, local CPU otherwise."""

import logging

logger = logging.getLogger(__name__)

# Remote GPU workers (Colab, GPU Docker host) consume this queue: `celery worker -Q gpu`.
GPU_QUEUE = "gpu"
# Celery's default queue, consumed by the local CPU worker and by RAG indexing.
CPU_QUEUE = "celery"

PROCESS_VIDEO_TASK = "ai_workers.process_video"
PREPARE_YOUTUBE_TASK = "ai_workers.prepare_youtube_video"


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


def dispatch_video_job(
    celery_app,
    job_id: str,
    video_path: str,
    process_kwargs: dict,
    max_duration: int,
    r2_enabled: bool,
) -> str:
    """Send a video job to Celery and return the queue that received it.

    A YouTube link bound for a GPU worker is first downloaded by the local worker and published to
    R2, because YouTube often blocks datacenter IPs such as Colab. Both tasks use the job id, so
    job tracking and cancellation do not change. Without R2 the GPU worker could not fetch the file.
    """
    target_queue = choose_video_queue(celery_app, video_path)
    is_youtube = "youtube.com" in video_path or "youtu.be" in video_path

    if is_youtube and target_queue == GPU_QUEUE and r2_enabled:
        celery_app.send_task(
            PREPARE_YOUTUBE_TASK,
            args=[job_id, video_path, GPU_QUEUE],
            kwargs={**process_kwargs, "max_duration": max_duration},
            task_id=job_id,
            queue=CPU_QUEUE,
        )
        return CPU_QUEUE

    celery_app.send_task(
        PROCESS_VIDEO_TASK,
        args=[job_id, video_path],
        kwargs=process_kwargs,
        task_id=job_id,
        queue=target_queue,
    )
    return target_queue

"""Pipeline service — submit Celery tasks for video processing."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import UploadFile


async def submit_pipeline_job(
    user_id: str,
    file: UploadFile | None = None,
    youtube_url: str | None = None,
    config_stack: str = "hybrid",
) -> str:
    """Submit a video processing job to Celery queue.

    Returns the job_id (UUID) for tracking progress.
    """
    job_id = str(uuid.uuid4())

    # TODO: save video to disk
    # TODO: create DB record in 'videos' table
    # TODO: send Celery task:
    #   from ai_workers.tasks import process_video
    #   process_video.delay(job_id=job_id, video_path=saved_path, config_stack=config_stack)

    return job_id

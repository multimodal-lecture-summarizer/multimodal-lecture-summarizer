"""Upload endpoint — nhận video file hoặc YouTube URL."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile, Form

from app.core.security import verify_token
from app.services.pipeline_service import submit_pipeline_job

router = APIRouter()


@router.post("/upload")
async def upload_video(
    file: UploadFile | None = File(None),
    youtube_url: str | None = Form(None),
    config_stack: str = Form("hybrid"),
    user: dict = Depends(verify_token),
):
    """Upload a video file or provide a YouTube URL for processing."""
    if not file and not youtube_url:
        return {"error": "Cần cung cấp file video hoặc YouTube URL"}

    # TODO: validate video duration, format
    # TODO: save file to disk / download from YouTube
    job_id = await submit_pipeline_job(
        user_id=user.get("sub"),
        file=file,
        youtube_url=youtube_url,
        config_stack=config_stack,
    )
    return {"job_id": job_id, "status": "queued"}

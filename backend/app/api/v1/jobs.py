import uuid
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, JobDTO
from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.job import Job
from app.models.video import Video

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.get(
    "/video/{video_id}",
    response_model=BaseDTO[List[JobDTO]],
    summary="Get jobs associated with a video",
    description="Lists all background processing tasks triggered for a specific video.",
)
def list_video_jobs(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve all jobs run on a video."""
    # Ensure video exists and belongs to the user
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    jobs = db.query(Job).filter(Job.video_id == video_id).all()
    return BaseDTO(
        success=True,
        data=[JobDTO.model_validate(j) for j in jobs],
        message="Jobs list retrieved successfully",
    )


@router.get(
    "/{job_id}",
    response_model=BaseDTO[JobDTO],
    summary="Get job execution status",
    description="Retrieves the live processing state of a background task.",
)
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Fetch status for a specific job."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise NotFoundException(message=f"Job with ID {job_id} not found")

    # Verify ownership of the video linked to job
    video = (
        db.query(Video)
        .filter(
            Video.video_id == job.video_id, Video.user_id == current_user.user_id
        )
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Job with ID {job_id} not found")

    return BaseDTO(
        success=True,
        data=JobDTO.model_validate(job),
        message="Job details retrieved successfully",
    )

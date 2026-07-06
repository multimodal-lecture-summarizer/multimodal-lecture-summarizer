import uuid
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.exceptions import NotFoundException, ForbiddenException, ValidationException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, JobDTO, create_pagination_metadata
from app.api.deps import get_current_active_user, check_admin
from app.models.user import User
from app.models.job import Job
from app.models.video import Video

router = APIRouter(route_class=CamelCaseAPIRoute)


def sync_job_status(job: Job, db: Session):
    from app.core.constants import JobStatus, VideoStatus
    from app.models.video import Video
    import datetime as dt
    
    # Initialize default attributes
    job.logs = []
    if job.status == JobStatus.COMPLETED:
        job.progress = 100
        job.stage = "completed"
    elif job.status == JobStatus.FAILED:
        job.progress = 100
        job.stage = "failed"
    else:
        job.progress = 0
        job.stage = "queued"

    if job.status not in [JobStatus.PENDING, JobStatus.RUNNING]:
        return
        
    try:
        from app.core.celery_app import celery_app
        from celery.result import AsyncResult
        
        async_res = AsyncResult(str(job.job_id), app=celery_app)
        if async_res.state == "PROGRESS":
            meta = async_res.info or {}
            job.stage = meta.get("stage", "processing")
            job.progress = meta.get("progress", 0)
            job.logs = meta.get("logs", [])
        elif async_res.state == "SUCCESS":
            job.stage = "completed"
            job.progress = 100
            result = async_res.result
            if not result:
                return
                
            # Update video
            video = db.query(Video).filter(Video.video_id == job.video_id).first()
            if video:
                video.status = VideoStatus.DONE
                if "duration" in result:
                    video.duration = result["duration"]
                if "video_title" in result:
                    video.title = result["video_title"]
                if "video_file_path" in result and result["video_file_path"]:
                    video.file_path = result["video_file_path"]
            
            # Save Summary
            from app.models.summary import Summary
            existing_summary = db.query(Summary).filter(Summary.video_id == job.video_id).first()
            if not existing_summary:
                import json
                transcript_content = result.get("transcript_text", "")
                if "transcript_segments" in result and result["transcript_segments"]:
                    transcript_content = json.dumps(result["transcript_segments"])
                    
                summary = Summary(
                    video_id=job.video_id,
                    summary_text=result.get("summary", ""),
                    chapters_json=result.get("chapters", []),
                    keyframes_json=result.get("keyframes", []),
                    transcript_text=transcript_content,
                    model_used=result.get("model_used", "Groq Llama 3.1 8B"),
                    processing_time=result.get("processing_time", 0.0)
                )
                db.add(summary)
                
            # Save VideoScenes
            from app.models.video import VideoScene
            db.query(VideoScene).filter(VideoScene.video_id == job.video_id).delete()
            
            for scene in result.get("scenes", []):
                db_scene = VideoScene(
                    video_id=job.video_id,
                    scene_index=scene["scene_index"],
                    start_seconds=scene["start_seconds"],
                    end_seconds=scene["end_seconds"],
                    start_timecode=scene["start_timecode"],
                    end_timecode=scene["end_timecode"],
                    start_frame=scene["start_frame"],
                    end_frame=scene["end_frame"],
                    keyframe_path=scene["keyframe_path"],
                    keyframe_url=scene["keyframe_url"],
                    caption=scene.get("caption", f"Phân cảnh {scene['scene_index']}"),
                    script=scene.get("script", "")
                )
                db.add(db_scene)
                
            # Add to ChromaDB vector index
            chunks = []
            metadatas = []
            for idx, seg in enumerate(result.get("transcript_segments", [])):
                chunks.append(seg["text"])
                metadatas.append({
                    "video_id": str(job.video_id),
                    "chunk_index": idx,
                    "timestamp_start": float(seg["start"])
                })
            if chunks:
                from app.services.chromadb import chromadb_service
                try:
                    chromadb_service.add_transcript_chunks(job.video_id, chunks, metadatas)
                except Exception as inner_e:
                    import logging
                    logging.getLogger(__name__).error(f"Failed to add to ChromaDB: {inner_e}")
                    
            # Complete Job
            job.status = JobStatus.COMPLETED
            job.completed_at = dt.datetime.utcnow()
            db.commit()
            db.refresh(job)
            
        elif async_res.state == "FAILURE":
            video = db.query(Video).filter(Video.video_id == job.video_id).first()
            if video:
                video.status = VideoStatus.FAILED
            job.status = JobStatus.FAILED
            job.completed_at = dt.datetime.utcnow()
            job.error_log = str(async_res.result)
            db.commit()
            db.refresh(job)
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error checking Celery task for job {job.job_id}: {e}")


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
    for job in jobs:
        sync_job_status(job, db)
        
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

    sync_job_status(job, db)
    return BaseDTO(
        success=True,
        data=JobDTO.model_validate(job),
        message="Job details retrieved successfully",
    )



@router.get(
    "/admin/all",
    response_model=BaseDTO[List[JobDTO]],
    summary="List all background jobs (Admin only)",
    description="Retrieves all background tasks in the system. Requires Admin permissions.",
)
def list_all_jobs(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    """Lists all processing jobs. Requires Admin role."""
    total = db.query(Job).count()
    from app.core.constants import JobStatus
    completed = db.query(Job).filter(Job.status == JobStatus.COMPLETED).count()
    failed = db.query(Job).filter(Job.status == JobStatus.FAILED).count()
    processing = db.query(Job).filter(Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING])).count()
    
    jobs = db.query(Job).order_by(Job.created_at.desc() if hasattr(Job, 'created_at') else Job.job_id.desc()).offset(offset).limit(limit).all()
    for j in jobs:
        sync_job_status(j, db)
        
    return BaseDTO(
        success=True,
        data=[JobDTO.model_validate(j) for j in jobs],
        message="All background jobs retrieved successfully",
        metadata=create_pagination_metadata(
            limit=limit,
            offset=offset,
            total=total,
            count=len(jobs),
            completed=completed,
            failed=failed,
            processing=processing
        ),
    )


@router.post(
    "/{job_id}/cancel",
    response_model=BaseDTO[JobDTO],
    summary="Cancel / Stop a running Celery job",
    description="Revokes the Celery task, terminates the worker thread processing it, and updates the database state.",
)
def cancel_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Cancels/revokes a Celery task."""
    job = db.query(Job).filter(Job.job_id == job_id).first()
    if not job:
        raise NotFoundException(message=f"Job with ID {job_id} not found")

    # Verify ownership of the video linked to job (or check if admin)
    from app.core.constants import UserRole
    video = db.query(Video).filter(Video.video_id == job.video_id).first()
    if not video:
        raise NotFoundException(message=f"Video associated with job {job_id} not found")
        
    if video.user_id != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise ForbiddenException(message="You do not have permission to cancel this job")

    from app.core.constants import JobStatus, VideoStatus
    import datetime as dt

    # Revoke task in Celery
    try:
        from app.core.celery_app import celery_app
        celery_app.control.revoke(str(job.job_id), terminate=True, signal="SIGKILL")
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to revoke task {job.job_id} directly: {e}")

    # Update job and video statuses
    job.status = JobStatus.FAILED
    job.error_log = "Tác vụ đã bị dừng bởi người dùng."
    job.completed_at = dt.datetime.utcnow()
    
    video.status = VideoStatus.FAILED
    
    db.commit()
    db.refresh(job)
    
    return BaseDTO(
        success=True,
        data=JobDTO.model_validate(job),
        message="Tác vụ đã được dừng thành công",
    )


import uuid
import os
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, status, Header
from sqlalchemy.orm import Session
from app.core.database import get_db, SessionLocal
from app.core.exceptions import NotFoundException, ValidationException, ForbiddenException
from app.core.config import settings
from app.core.constants import VideoStatus, JobType, JobStatus, UserRole
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, VideoDTO, VideoStandardDTO, VideoStandardBase, VideoMetadataDTO, VideoSceneDTO
from app.api.deps import get_current_active_user, check_admin
from app.models.user import User
from app.models.video import Video, VideoStandard, VideoMetadata, VideoScene
from app.models.job import Job
from app.models.summary import Summary
from app.services.r2 import r2_service
from app.services.chromadb import chromadb_service

router = APIRouter(route_class=CamelCaseAPIRoute)


def process_video_simulation(video_id: uuid.UUID):
    """
    Background Task simulating the full Multimodal AI Pipeline:
    ASR (WhisperX) -> Keyframes (CLIP/BLIP-2) -> Fusion & LLM (Groq) -> Vector Store (ChromaDB)
    """
    db: Session = SessionLocal()
    try:
        # Fetch Video
        video = db.query(Video).filter(Video.video_id == video_id).first()
        if not video:
            return

        # Fetch pending job
        job = (
            db.query(Job)
            .filter(Job.video_id == video_id, Job.job_type == JobType.SUMMARIZE)
            .first()
        )
        if not job:
            job = Job(video_id=video_id, job_type=JobType.SUMMARIZE)
            db.add(job)

        # Step 1: Update status to processing
        video.status = VideoStatus.PROCESSING
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow() if "datetime" in globals() else None
        # Fallback timestamp setting
        import datetime as dt

        job.started_at = dt.datetime.utcnow()
        db.commit()

        # Simulate delay of ASR and Visual pipelines
        time.sleep(4)

        # Generate standard mock transcript text and chapters
        transcript_text = (
            "Welcome to this lecture on Web Application Architectures. Today, we will discuss "
            "Microservices versus Monolithic systems. In the first part, we examine why "
            "companies shift to Microservices to solve scaling problems. "
            "In the second part, we explore Cloudflare R2 object storage capabilities. "
            "Finally, we implement ChromaDB vector queries to enable RAG features."
        )

        mock_chapters = [
            {
                "title": "Introduction to Architectures",
                "startTime": 0.0,
                "endTime": 60.0,
                "summary": "Introduction to the fundamental concepts of monolithic and microservices designs.",
            },
            {
                "title": "Monolith vs Microservices Trade-offs",
                "startTime": 60.0,
                "endTime": 180.0,
                "summary": "Detailed trade-offs analysis of switching to service-oriented designs.",
            },
            {
                "title": "Object Storage Integration (Cloudflare R2)",
                "startTime": 180.0,
                "endTime": 300.0,
                "summary": "Overview of R2 setup and how it manages multimedia and keyframes assets efficiently.",
            },
            {
                "title": "RAG implementation with ChromaDB",
                "startTime": 300.0,
                "endTime": 450.0,
                "summary": "Coding step-by-step query searches using embeddings for conversational UI systems.",
            },
        ]

        mock_keyframes = [
            {
                "timestamp": 15.5,
                "imageUrl": "/static/mock_r2/keyframes/slide1.png",
                "description": "Slide title: Monolith architecture diagrams and unified codebase blocks.",
                "importanceScore": 0.82,
            },
            {
                "timestamp": 120.0,
                "imageUrl": "/static/mock_r2/keyframes/slide2.png",
                "description": "Slide showing pros & cons list table highlighting developer velocities.",
                "importanceScore": 0.95,
            },
            {
                "timestamp": 240.5,
                "imageUrl": "/static/mock_r2/keyframes/code1.png",
                "description": "IDE view detailing Python code writing boto3 calls to cloud targets.",
                "importanceScore": 0.76,
            },
        ]

        # Add text chunks to ChromaDB/Mock vector store
        chunks = [
            "Introduction to Monolithic vs Microservices architectures. Scaling codebases.",
            "Analyzing performance bottlenecks, network hops and service boundaries.",
            "Setting up Cloudflare R2 API endpoints with access credentials in env files.",
            "Implementing ChromaDB collection query search with semantic text weights.",
        ]
        metadatas = [
            {"video_id": str(video_id), "chunk_index": 0, "timestamp_start": 0.0},
            {"video_id": str(video_id), "chunk_index": 1, "timestamp_start": 60.0},
            {"video_id": str(video_id), "chunk_index": 2, "timestamp_start": 180.0},
            {"video_id": str(video_id), "chunk_index": 3, "timestamp_start": 300.0},
        ]
        chromadb_service.add_transcript_chunks(video_id, chunks, metadatas)

        # Create summary record
        summary = Summary(
            video_id=video_id,
            summary_text=(
                "## AI Summary Overview\n"
                "The lecture provides a comprehensive deep dive comparing monolith and microservices "
                "architectures. It addresses performance, scaling limits, and implementation details for "
                "supporting services. It outlines R2 storage configuration and concludes with an "
                "interactive walkthrough implementing RAG query capabilities using vector database search index."
            ),
            chapters_json=mock_chapters,
            keyframes_json=mock_keyframes,
            transcript_text=transcript_text,
            model_used=settings.GROQ_MODEL,
            processing_time=12.5,
        )
        db.add(summary)

        # Complete Job and Video status
        video.status = VideoStatus.DONE
        job.status = JobStatus.COMPLETED
        job.completed_at = dt.datetime.utcnow()
        db.commit()

    except Exception as e:
        db.rollback()
        # Mark video and job as failed
        try:
            video = db.query(Video).filter(Video.video_id == video_id).first()
            job = (
                db.query(Job)
                .filter(
                    Job.video_id == video_id, Job.job_type == JobType.SUMMARIZE
                )
                .first()
            )
            if video:
                video.status = VideoStatus.FAILED
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = dt.datetime.utcnow()
                job.error_log = str(e)
            db.commit()
        except Exception as inner_ex:
            pass
    finally:
        db.close()


@router.post(
    "/upload",
    response_model=BaseDTO[VideoDTO],
    summary="Upload a local video file or submit a YouTube URL",
    description="Validates the video using configured standards, uploads it to storage, and queues summarization task.",
)
def upload_video(
    background_tasks: BackgroundTasks,
    original_url: Optional[str] = Form(None),
    language: str = Form("en"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Processes upload request, validates standards, stores metadata, runs background AI tasks."""
    # 1. Fetch current validation standards
    standard = db.query(VideoStandard).order_by(VideoStandard.standard_id.desc()).first()
    if not standard:
        # Create default standard if none exists
        standard = VideoStandard(
            max_duration=settings.DEFAULT_MAX_DURATION_SECONDS,
            allowed_formats=settings.DEFAULT_ALLOWED_FORMATS,
            max_file_size=settings.DEFAULT_MAX_FILE_SIZE_MB,
        )
        db.add(standard)
        db.commit()
        db.refresh(standard)

    # 2. Check input (either file or original_url must be provided)
    if not file and not original_url:
        raise ValidationException(
            message="Either a video file or a YouTube URL must be provided."
        )

    duration = 0.0
    file_path = None

    if file:
        # Validate format
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
        allowed_formats_list = [
            f.strip() for f in standard.allowed_formats.split(",")
        ]
        if file_ext not in allowed_formats_list:
            raise ValidationException(
                message=f"Format '.{file_ext}' is not allowed. Supported formats: {standard.allowed_formats}"
            )

        # Validate file size
        file.file.seek(0, 2)
        file_size_mb = file.file.tell() / (1024 * 1024)
        file.file.seek(0)
        if file_size_mb > standard.max_file_size:
            raise ValidationException(
                message=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({standard.max_file_size} MB)"
            )

        # Upload file to mock or actual R2
        # Save temp file locally first to upload
        temp_dir = os.path.join(os.getcwd(), "storage", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(
            temp_dir, f"{uuid.uuid4()}.{file_ext}"
        )
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        object_name = f"videos/{uuid.uuid4()}.{file_ext}"
        file_path = r2_service.upload_file(temp_file_path, object_name)

        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Simulate extracting duration (mock)
        duration = 185.0
    else:
        # YouTube URL scenario
        if "youtube.com" not in original_url and "youtu.be" not in original_url:
            raise ValidationException(
                message="Only YouTube URLs are supported for remote video inputs."
            )
        duration = 320.0  # Mock remote video duration

        if duration > standard.max_duration:
            raise ValidationException(
                message=f"Video duration exceeds maximum allowed duration ({standard.max_duration} seconds)"
            )

    # 3. Create Video metadata record
    db_video = Video(
        user_id=current_user.user_id,
        original_url=original_url,
        file_path=file_path,
        duration=duration,
        language=language,
        status=VideoStatus.PENDING,
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    # 4. Create Job record
    db_job = Job(
        video_id=db_video.video_id,
        job_type=JobType.SUMMARIZE,
        status=JobStatus.PENDING,
    )
    db.add(db_job)
    db.commit()

    # 5. Push task to background queue
    background_tasks.add_task(process_video_simulation, db_video.video_id)

    return BaseDTO(
        success=True,
        data=VideoDTO.model_validate(db_video),
        message="Video submitted successfully, processing started.",
    )


@router.get(
    "",
    response_model=BaseDTO[List[VideoDTO]],
    summary="List all videos for the current user",
    description="Retrieves a paginated list of videos uploaded by the authenticated user.",
)
def list_videos(
    status: Optional[VideoStatus] = None,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Lists videos with optional status filtering."""
    query = db.query(Video).filter(Video.user_id == current_user.user_id)
    if status:
        query = query.filter(Video.status == status)

    videos = query.order_by(Video.uploaded_at.desc()).offset(offset).limit(limit).all()

    return BaseDTO(
        success=True,
        data=[VideoDTO.model_validate(v) for v in videos],
        message="Videos list retrieved successfully",
        metadata={"limit": limit, "offset": offset, "count": len(videos)},
    )


@router.get(
    "/standards",
    response_model=BaseDTO[VideoStandardDTO],
    summary="Retrieve active video standards",
    description="Gets the currently active video upload standards (max size, allowed formats, etc.)",
)
def get_standards(db: Session = Depends(get_db)):
    """Fetch current limits config."""
    standard = db.query(VideoStandard).order_by(VideoStandard.standard_id.desc()).first()
    if not standard:
        standard = VideoStandard(
            max_duration=settings.DEFAULT_MAX_DURATION_SECONDS,
            allowed_formats=settings.DEFAULT_ALLOWED_FORMATS,
            max_file_size=settings.DEFAULT_MAX_FILE_SIZE_MB,
        )
        db.add(standard)
        db.commit()
        db.refresh(standard)

    return BaseDTO(
        success=True,
        data=VideoStandardDTO.model_validate(standard),
        message="Standards configuration retrieved successfully",
    )


@router.put(
    "/standards",
    response_model=BaseDTO[VideoStandardDTO],
    summary="Update video standards (Admin only)",
    description="Updates the configurations for video validation constraints.",
)
def update_standards(
    standard_in: VideoStandardBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    """Modifies limits settings. Requires Admin role."""
    standard = db.query(VideoStandard).order_by(VideoStandard.standard_id.desc()).first()
    if not standard:
        standard = VideoStandard()

    standard.max_duration = standard_in.max_duration
    standard.allowed_formats = standard_in.allowed_formats
    standard.max_file_size = standard_in.max_file_size
    standard.min_audio_quality = standard_in.min_audio_quality

    db.add(standard)
    db.commit()
    db.refresh(standard)

    return BaseDTO(
        success=True,
        data=VideoStandardDTO.model_validate(standard),
        message="Standards configuration updated successfully",
    )


@router.get(
    "/{video_id}",
    response_model=BaseDTO[VideoDTO],
    summary="Get video details by ID",
    description="Retrieves the metadata status of a single video.",
)
def get_video(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve single video."""
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    return BaseDTO(
        success=True,
        data=VideoDTO.model_validate(video),
        message="Video details retrieved successfully",
    )


@router.get(
    "/{video_id}/stream",
    summary="Stream video file",
    description="Streams the video file chunk-by-chunk directly from storage or R2, supporting range requests.",
)
def stream_video(
    video_id: uuid.UUID,
    range: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Streams video file."""
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    if not video.file_path:
        raise NotFoundException(message="Video file path not found")

    # If it is a local static path, redirect to it
    if video.file_path.startswith("/static/"):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=video.file_path)

    from app.services.r2 import r2_service
    from fastapi.responses import StreamingResponse

    # If R2 is not enabled, error
    if not r2_service.enabled:
        raise ValidationException(message="Storage service not enabled")

    # Extract object key
    bucket_name = r2_service.bucket_name
    bucket_prefix = f"/{bucket_name}/"
    if bucket_prefix in video.file_path:
        object_key = video.file_path.split(bucket_prefix, 1)[1]
    else:
        object_key = video.file_path.split("/", 3)[-1]

    # Remove existing query params if any
    object_key = object_key.split("?")[0]

    params = {
        "Bucket": bucket_name,
        "Key": object_key
    }
    if range:
        params["Range"] = range

    try:
        response = r2_service.s3_client.get_object(**params)
        
        headers = {}
        if "ContentRange" in response:
            headers["Content-Range"] = response["ContentRange"]
        if "AcceptRanges" in response:
            headers["Accept-Ranges"] = response["AcceptRanges"]
        if "ContentType" in response:
            headers["Content-Type"] = response["ContentType"]
        if "ContentLength" in response:
            headers["Content-Length"] = str(response["ContentLength"])

        def iter_chunks():
            body = response["Body"]
            for chunk in body.iter_chunks(chunk_size=1024*1024):  # 1MB chunks
                yield chunk

        status_code = 206 if range else 200
        return StreamingResponse(
            iter_chunks(),
            status_code=status_code,
            headers=headers
        )
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to stream video from R2 for {video_id}: {e}")
        raise ValidationException(message="Failed to stream video from storage")


@router.get(
    "/{video_id}/metadata",
    response_model=BaseDTO[VideoMetadataDTO],
    summary="Get video metadata details",
    description="Retrieves the technical metadata dimensions, FPS, and frame count of a video.",
)
def get_video_metadata(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve video metadata."""
    # Check that video exists and belongs to current user
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    metadata = db.query(VideoMetadata).filter(VideoMetadata.video_id == video_id).first()
    if not metadata:
        raise NotFoundException(message=f"Metadata for video {video_id} not found")

    return BaseDTO(
        success=True,
        data=VideoMetadataDTO.model_validate(metadata),
        message="Video metadata retrieved successfully",
    )


@router.get(
    "/{video_id}/scenes",
    response_model=BaseDTO[List[VideoSceneDTO]],
    summary="Get video scenes list",
    description="Retrieves the segmented list of scenes with timestamps, keyframe images, and transcripts.",
)
def get_video_scenes(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Retrieve video scenes."""
    # Check that video exists and belongs to current user
    video = (
        db.query(Video)
        .filter(Video.video_id == video_id, Video.user_id == current_user.user_id)
        .first()
    )
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    scenes = (
        db.query(VideoScene)
        .filter(VideoScene.video_id == video_id)
        .order_by(VideoScene.scene_index.asc())
        .all()
    )

    return BaseDTO(
        success=True,
        data=[VideoSceneDTO.model_validate(s) for s in scenes],
        message="Video scenes retrieved successfully",
    )


@router.get(
    "/admin/all",
    response_model=BaseDTO[List[VideoDTO]],
    summary="List all videos across all users (Admin only)",
    description="Retrieves a list of all uploaded videos in the system. Requires Admin permissions.",
)
def list_all_videos(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    """Lists all videos for administration. Requires Admin role."""
    videos = db.query(Video).order_by(Video.uploaded_at.desc()).offset(offset).limit(limit).all()
    return BaseDTO(
        success=True,
        data=[VideoDTO.model_validate(v) for v in videos],
        message="All system videos retrieved successfully",
        metadata={"limit": limit, "offset": offset, "count": len(videos)},
    )


@router.delete(
    "/admin/{video_id}",
    response_model=BaseDTO[bool],
    summary="Delete any video (Admin only)",
    description="Deletes a video record and all associated jobs/summaries from the database. Requires Admin permissions.",
)
def delete_video_admin(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    """Deletes a video and its children. Requires Admin role."""
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    # Delete associated jobs
    db.query(Job).filter(Job.video_id == video_id).delete()
    # Delete associated summaries
    db.query(Summary).filter(Summary.video_id == video_id).delete()
    # Delete video
    db.delete(video)
    db.commit()

    return BaseDTO(
        success=True,
        data=True,
        message="Video and associated records deleted successfully",
    )

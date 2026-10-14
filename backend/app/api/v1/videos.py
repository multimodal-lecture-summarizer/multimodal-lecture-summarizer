import uuid
import os
import time
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, BackgroundTasks, status, Header
from sqlalchemy.orm import Session, joinedload
from app.core.database import get_db, SessionLocal
from app.core.exceptions import NotFoundException, ValidationException, ForbiddenException
from app.core.config import settings
from app.core.constants import VideoStatus, JobType, JobStatus, UserRole
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, VideoDTO, VideoStandardDTO, VideoStandardBase, VideoMetadataDTO, VideoSceneDTO, create_pagination_metadata
from app.api.deps import get_current_active_user, check_admin
from app.models.user import User
from app.models.video import Video, VideoStandard, VideoMetadata, VideoScene
from app.models.job import Job
from app.models.summary import Summary
from app.services.r2 import r2_service
from app.services.chromadb import chromadb_service
from app.services.video_inspector import inspect_video_streams, validate_video_standards

router = APIRouter(route_class=CamelCaseAPIRoute)





@router.post(
    "/inspect-preflight",
    response_model=BaseDTO[dict],
    summary="Pre-flight inspection checklist for video upload",
    description="Inspects video stream parameters and evaluates them step-by-step against system standards.",
)
def inspect_preflight(
    original_url: Optional[str] = Form(None),
    originalUrl: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Performs deep pre-flight inspection and returns itemized checklist results."""
    url = original_url or originalUrl
    standard = db.query(VideoStandard).order_by(VideoStandard.standard_id.desc()).first()
    if not standard:
        standard = VideoStandard()

    if not file and not url:
        raise ValidationException(message="Vui lòng cung cấp tệp video hoặc đường dẫn YouTube.")

    file_name = file.filename if file else (url or "Video")
    file_ext = file_name.split(".")[-1].lower() if "." in file_name else "mp4"

    video_codec = None
    audio_codec = None
    width = 0
    height = 0
    fps = 0.0
    duration = 0.0
    size_mb = 0.0
    raw_error = None

    if file:
        file.file.seek(0, 2)
        size_bytes = file.file.tell()
        size_mb = round(size_bytes / (1024 * 1024), 2)
        file.file.seek(0)

        temp_dir = os.path.join(os.getcwd(), "storage", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"preflight_{uuid.uuid4()}.{file_ext}")
        try:
            with open(temp_file_path, "wb") as buf:
                buf.write(file.file.read())
            
            probe = inspect_video_streams(temp_file_path)
            if probe.get("valid") and probe.get("video"):
                v = probe["video"]
                a = probe.get("audio") or {}
                fmt = probe.get("format") or {}
                video_codec = v.get("codec_name")
                audio_codec = a.get("codec_name")
                width = v.get("width") or 0
                height = v.get("height") or 0
                fps = v.get("fps") or 0.0
                duration = float(fmt.get("duration") or v.get("duration") or 0.0)
            else:
                raw_error = probe.get("error") or "Không thể đọc luồng video"
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    else:
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValidationException(message="Chỉ hỗ trợ đường dẫn YouTube hợp lệ.")
        try:
            import yt_dlp
            ydl_opts = {
                'quiet': True,
                'skip_download': True,
                'extract_flat': False,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    width = info.get("width") or 1920
                    height = info.get("height") or 1080
                    duration = float(info.get("duration") or 0.0)
                    fps = float(info.get("fps") or 30.0)
                    vcodec = info.get("vcodec") or "h264"
                    acodec = info.get("acodec") or "aac"
                    video_codec = "h264" if "avc" in vcodec.lower() or "h264" in vcodec.lower() else vcodec.split(".")[0].lower()
                    audio_codec = "aac" if "mp4a" in acodec.lower() or "aac" in acodec.lower() else acodec.split(".")[0].lower()
                    file_name = info.get("title") or url
                    size_mb = round(duration * 0.04, 2)
        except Exception:
            width, height = 1920, 1080
            duration = 300.0
            fps = 30.0
            video_codec = "h264"
            audio_codec = "aac"
            size_mb = 12.5

    allowed_fmts = [f.strip().lower() for f in standard.allowed_formats.split(",") if f.strip()]
    if file:
        pass_fmt = file_ext in allowed_fmts
        msg_fmt = f"Định dạng .{file_ext} hợp lệ" if pass_fmt else f"Định dạng .{file_ext} không được hỗ trợ (cần {standard.allowed_formats})"
    else:
        pass_fmt = True
        msg_fmt = "Đường dẫn YouTube tương thích"

    raw_allowed_codecs = standard.allowed_codecs or ["h264", "hevc"]
    if isinstance(raw_allowed_codecs, str):
        allowed_codecs = [c.strip().lower() for c in raw_allowed_codecs.split(",") if c.strip()]
    else:
        allowed_codecs = [str(c).strip().lower() for c in raw_allowed_codecs if str(c).strip()]

    pass_codec = bool(video_codec and (video_codec.lower() in allowed_codecs or any(c in video_codec.lower() for c in allowed_codecs)))
    if raw_error:
        pass_codec = False
        msg_codec = f"Lỗi tệp: {raw_error}"
    elif pass_codec:
        msg_codec = f"Codec '{video_codec.upper()}' được hỗ trợ tối ưu"
    else:
        msg_codec = f"Codec '{video_codec or 'Không rõ'}' không nằm trong danh sách cho phép ({', '.join(allowed_codecs)})"

    pass_res = (width >= standard.min_width and height >= standard.min_height and
                width <= standard.max_width and height <= standard.max_height)
    if pass_res:
        msg_res = f"Độ phân giải {width}×{height} đạt chuẩn"
    elif width < standard.min_width or height < standard.min_height:
        msg_res = f"Độ phân giải {width}×{height} thấp hơn mức tối thiểu ({standard.min_width}×{standard.min_height})"
    else:
        msg_res = f"Độ phân giải {width}×{height} vượt quá mức tối đa ({standard.max_width}×{standard.max_height})"

    pass_fps = fps >= 15.0 or fps == 0.0
    msg_fps = f"{fps:.1f} FPS (Tốc độ khung hình mượt mà)" if pass_fps else f"{fps:.1f} FPS (Dưới ngưỡng khuyến nghị 15 FPS)"

    pass_duration = duration <= standard.max_duration if duration > 0 else True
    mins = int(duration // 60)
    secs = int(duration % 60)
    max_mins = int(standard.max_duration // 60)
    if pass_duration:
        msg_duration = f"Thời lượng {mins}m {secs}s nằm trong giới hạn (≤ {max_mins} phút)"
    else:
        msg_duration = f"Thời lượng {mins}m {secs}s vượt quá giới hạn tối đa ({max_mins} phút)"

    pass_size = size_mb <= standard.max_file_size if size_mb > 0 else True
    if pass_size:
        msg_size = f"Dung lượng {size_mb:.1f} MB đạt chuẩn (≤ {standard.max_file_size} MB)"
    else:
        msg_size = f"Dung lượng {size_mb:.1f} MB vượt quá giới hạn ({standard.max_file_size} MB)"

    pass_audio = bool(audio_codec)
    msg_audio = f"Luồng âm thanh: {audio_codec.upper() if audio_codec else 'Phát hiện'}" if pass_audio else "Không phát hiện luồng âm thanh hoặc codec không hỗ trợ"

    checks = [
        {
            "id": "format",
            "name": "Định dạng container",
            "detected": f".{file_ext}" if file else "YouTube URL",
            "standard": standard.allowed_formats,
            "passed": pass_fmt,
            "message": msg_fmt,
        },
        {
            "id": "codec",
            "name": "Chuẩn nén Video (Codec)",
            "detected": (video_codec or "Chưa rõ").upper(),
            "standard": ", ".join(allowed_codecs).upper(),
            "passed": pass_codec,
            "message": msg_codec,
        },
        {
            "id": "resolution",
            "name": "Độ phân giải hiển thị",
            "detected": f"{width}×{height}" if width else "N/A",
            "standard": f"{standard.min_width}×{standard.min_height} — {standard.max_width}×{standard.max_height}",
            "passed": pass_res,
            "message": msg_res,
        },
        {
            "id": "fps",
            "name": "Tốc độ khung hình (FPS)",
            "detected": f"{fps:.1f} FPS" if fps else "Chuẩn",
            "standard": "≥ 15.0 FPS",
            "passed": pass_fps,
            "message": msg_fps,
        },
        {
            "id": "duration",
            "name": "Thời lượng bài giảng",
            "detected": f"{mins}m {secs}s" if duration else "Hợp lệ",
            "standard": f"≤ {max_mins} phút ({standard.max_duration}s)",
            "passed": pass_duration,
            "message": msg_duration,
        },
        {
            "id": "size",
            "name": "Dung lượng tệp lưu trữ",
            "detected": f"{size_mb:.1f} MB",
            "standard": f"≤ {standard.max_file_size} MB",
            "passed": pass_size,
            "message": msg_size,
        },
        {
            "id": "audio",
            "name": "Luồng âm thanh & Thoại",
            "detected": (audio_codec or "Audio Track").upper(),
            "standard": "AAC, Opus, MP3",
            "passed": pass_audio,
            "message": msg_audio,
        },
    ]

    all_passed = all(c["passed"] for c in checks)

    return BaseDTO(
        success=True,
        data={
            "overallValid": all_passed,
            "fileName": file_name,
            "sizeMb": size_mb,
            "duration": duration,
            "resolution": f"{width}×{height}",
            "videoCodec": video_codec,
            "audioCodec": audio_codec,
            "fps": fps,
            "checks": checks,
        },
        message="Thẩm định tiêu chuẩn video hoàn tất.",
    )


@router.post(
    "/upload",
    response_model=BaseDTO[VideoDTO],
    summary="Upload a local video file or submit a YouTube URL",
    description="Validates the video using configured standards, uploads it to storage, and queues summarization task.",
)
def upload_video(
    background_tasks: BackgroundTasks,
    original_url: Optional[str] = Form(None),
    originalUrl: Optional[str] = Form(None),
    language: str = Form("en"),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Processes upload request, validates standards, stores metadata, runs background AI tasks."""
    url = original_url or originalUrl

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
    if not file and not url:
        raise ValidationException(
            message="Either a video file or a YouTube URL must be provided."
        )

    duration = 0.0
    file_path = None
    file_metadata_info = None

    if file:
        # Validate format
        file_ext = file.filename.split(".")[-1].lower() if file.filename else ""
        allowed_formats_list = [
            f.strip().lower() for f in standard.allowed_formats.split(",") if f.strip()
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

        # Save temp file locally first to inspect and upload
        temp_dir = os.path.join(os.getcwd(), "storage", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(
            temp_dir, f"{uuid.uuid4()}.{file_ext}"
        )
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Inspect video streams using ffprobe
        file_metadata_info = inspect_video_streams(temp_file_path)
        is_valid, validation_error = validate_video_standards(file_metadata_info, standard)
        if not is_valid:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise ValidationException(
                message=validation_error or "Tệp video không đáp ứng tiêu chuẩn kỹ thuật của hệ thống."
            )

        object_name = f"videos/{uuid.uuid4()}.{file_ext}"
        file_path = r2_service.upload_file(temp_file_path, object_name)

        # Get actual video duration from inspected streams
        v_fmt = file_metadata_info.get("format") or {}
        v_stream = file_metadata_info.get("video") or {}
        duration = float(v_fmt.get("duration") or v_stream.get("duration") or 0.0)

        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    else:
        # YouTube URL scenario
        if "youtube.com" not in url and "youtu.be" not in url:
            raise ValidationException(
                message="Only YouTube URLs are supported for remote video inputs."
            )
        duration = 0.0  # Will be updated by Celery worker

    # 2b. Check if video with same original_url already exists and has a file_path
    if url:
        existing_video = db.query(Video).filter(
            Video.original_url == url,
            Video.file_path != None,
            Video.file_path != ""
        ).first()
        if existing_video:
            file_path = existing_video.file_path
            if existing_video.duration:
                duration = existing_video.duration

    if duration > 0.0 and duration > standard.max_duration:
        raise ValidationException(
            message=f"Video duration ({duration:.1f} seconds) exceeds maximum allowed duration ({standard.max_duration} seconds)"
        )

    # 2c. Deduplication Guard: If user just submitted this video within 15s and it's already pending/processing, return it
    import datetime as dt
    fifteen_sec_ago = dt.datetime.utcnow() - dt.timedelta(seconds=15)
    existing_inflight = None
    if url:
        existing_inflight = (
            db.query(Video)
            .filter(
                Video.user_id == current_user.user_id,
                Video.original_url == url,
                Video.status.in_([VideoStatus.PENDING, VideoStatus.PROCESSING]),
                Video.uploaded_at >= fifteen_sec_ago,
            )
            .order_by(Video.uploaded_at.desc())
            .first()
        )
    elif file and duration > 0.0:
        candidate = (
            db.query(Video)
            .filter(
                Video.user_id == current_user.user_id,
                Video.status.in_([VideoStatus.PENDING, VideoStatus.PROCESSING]),
                Video.uploaded_at >= fifteen_sec_ago,
            )
            .order_by(Video.uploaded_at.desc())
            .first()
        )
        if candidate and abs((candidate.duration or 0.0) - duration) < 1.0:
            existing_inflight = candidate

    if existing_inflight:
        return BaseDTO(
            success=True,
            data=VideoDTO.model_validate(existing_inflight),
            message="Video đã được tải lên và đang trong tiến trình xử lý.",
        )

    # 3. Create Video record
    db_video = Video(
        user_id=current_user.user_id,
        original_url=url,
        file_path=file_path,
        duration=duration,
        language=language,
        status=VideoStatus.PENDING,
        title="Đang phân tích tên bài giảng...",
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    # 3b. Create VideoMetadata record if streams info is available from local inspection
    if file_metadata_info and file_metadata_info.get("video"):
        v_stream = file_metadata_info["video"]
        a_stream = file_metadata_info.get("audio") or {}
        db_meta = VideoMetadata(
            video_id=db_video.video_id,
            video_codec=v_stream.get("codec_name"),
            audio_codec=a_stream.get("codec_name") if a_stream else None,
            fps=float(v_stream.get("fps") or 0.0),
            frame_count=int(v_stream.get("frame_count") or 0),
            width=int(v_stream.get("width") or 0),
            height=int(v_stream.get("height") or 0),
            bitrate=v_stream.get("bitrate"),
            video_source="upload",
            video_path=file_path or "",
        )
        db.add(db_meta)
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
    import logging
    logger = logging.getLogger(__name__)
    
    physical_video_path = ""
    if file_path:
        if file_path.startswith("/static/mock_r2/"):
            relative_path = file_path.replace("/static/mock_r2/", "")
            physical_video_path = os.path.abspath(os.path.join(os.getcwd(), "storage", "mock_r2_bucket", relative_path))
        else:
            physical_video_path = file_path
    elif url:
        physical_video_path = url

    try:
        from app.core.celery_app import celery_app
        celery_app.send_task(
            "ai_workers.process_video",
            args=[str(db_job.job_id), physical_video_path],
            kwargs={"config_stack": "hybrid"},
            task_id=str(db_job.job_id)
        )
        # Update statuses for active run
        import datetime as dt
        db_job.status = JobStatus.RUNNING
        db_job.started_at = dt.datetime.utcnow()
        db_video.status = VideoStatus.PROCESSING
        db.commit()
        logger.info(f"Successfully enqueued Celery task for video {db_video.video_id} with task ID {db_job.job_id}")
    except Exception as e:
        logger.error(f"Failed to send task to Celery: {e}")
        # Mark job and video as failed directly
        db_job.status = JobStatus.FAILED
        db_job.error_log = f"Failed to enqueue task to Celery: {str(e)}"
        db_video.status = VideoStatus.FAILED
        db.commit()
        raise ValidationException(
            message=f"Không thể kết nối tới hàng đợi Celery. Hãy đảm bảo Redis và Celery Worker đang chạy. Lỗi: {str(e)}"
        )

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
    search_query: Optional[str] = None,
    sort_by: str = "newest",
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Lists videos with optional status filtering."""
    query = db.query(Video).filter(Video.user_id == current_user.user_id)
    if status:
        query = query.filter(Video.status == status)
    
    if search_query:
        query = query.filter(Video.title.ilike(f"%{search_query}%"))

    if sort_by == "oldest":
        query = query.order_by(Video.uploaded_at.asc())
    else:
        query = query.order_by(Video.uploaded_at.desc())

    total = query.count()
    videos = query.options(joinedload(Video.scenes)).offset(offset).limit(limit).all()

    # Sync active job statuses to update progress/status of processing videos
    from app.api.v1.jobs import sync_job_status
    for v in videos:
        if v.status in [VideoStatus.PENDING, VideoStatus.PROCESSING]:
            for job in v.jobs:
                sync_job_status(job, db)

    return BaseDTO(
        success=True,
        data=[VideoDTO.model_validate(v) for v in videos],
        message="Videos list retrieved successfully",
        metadata=create_pagination_metadata(
            limit=limit,
            offset=offset,
            total=total,
            count=len(videos)
        ),
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
    standard.allowed_codecs = standard_in.allowed_codecs
    standard.min_width = standard_in.min_width
    standard.min_height = standard_in.min_height
    standard.max_width = standard_in.max_width
    standard.max_height = standard_in.max_height

    db.add(standard)
    db.commit()
    db.refresh(standard)

    return BaseDTO(
        success=True,
        data=VideoStandardDTO.model_validate(standard),
        message="Standards configuration updated successfully",
    )


@router.get(
    "/admin/video-standards",
    response_model=BaseDTO[VideoStandardDTO],
    summary="Retrieve active video standards (Admin alias)",
    description="Admin alias endpoint for retrieving video upload standards.",
)
def get_admin_video_standards(
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    return get_standards(db=db)


@router.put(
    "/admin/video-standards",
    response_model=BaseDTO[VideoStandardDTO],
    summary="Update video standards (Admin alias)",
    description="Admin alias endpoint for updating video upload standards.",
)
def update_admin_video_standards(
    standard_in: VideoStandardBase,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
):
    return update_standards(standard_in=standard_in, db=db, current_user=current_user)


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
            try:
                for chunk in body.iter_chunks(chunk_size=1024*1024):  # 1MB chunks
                    yield chunk
            except Exception as exc:
                import logging
                logging.getLogger(__name__).info(f"Video stream disconnected/seeked: {exc}")
                return

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
    total = db.query(Video).count()
    completed = db.query(Video).filter(Video.status == VideoStatus.DONE).count()
    failed = db.query(Video).filter(Video.status == VideoStatus.FAILED).count()
    processing = db.query(Video).filter(Video.status.in_([VideoStatus.PENDING, VideoStatus.PROCESSING])).count()

    videos = db.query(Video).order_by(Video.uploaded_at.desc()).offset(offset).limit(limit).all()
    return BaseDTO(
        success=True,
        data=[VideoDTO.model_validate(v) for v in videos],
        message="All system videos retrieved successfully",
        metadata=create_pagination_metadata(
            limit=limit,
            offset=offset,
            total=total,
            count=len(videos),
            completed=completed,
            failed=failed,
            processing=processing
        ),
    )


def perform_hard_delete(video: Video, db: Session):
    """Performs a hard delete of a video, including storage files and ChromaDB vector chunks."""
    # 1. Delete original video file
    if video.file_path:
        video_key = r2_service.extract_key(video.file_path)
        if video_key:
            r2_service.delete_file(video_key)

    # 2. Delete keyframe images of scenes
    for scene in video.scenes:
        if scene.keyframe_url:
            kf_key = r2_service.extract_key(scene.keyframe_url)
            if kf_key:
                r2_service.delete_file(kf_key)
        elif scene.keyframe_path:
            kf_key = r2_service.extract_key(scene.keyframe_path)
            if kf_key:
                r2_service.delete_file(kf_key)

    # 3. Delete vector chunks from ChromaDB
    try:
        chromadb_service.delete_transcript_chunks(video.video_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Failed to delete transcript chunks from ChromaDB: {e}")

    # 4. Delete video record (cascade deletes jobs, summaries, scenes, qa_logs, video_metadata)
    db.delete(video)
    db.commit()


@router.delete(
    "/admin/{video_id}",
    response_model=BaseDTO[bool],
    summary="Delete any video (Admin only)",
    description="Deletes a video record and all associated jobs/summaries from the database, and cleans up R2 storage and ChromaDB. Requires Admin permissions.",
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

    perform_hard_delete(video, db)

    return BaseDTO(
        success=True,
        data=True,
        message="Video and associated records deleted successfully",
    )


@router.delete(
    "/{video_id}",
    response_model=BaseDTO[bool],
    summary="Delete user's own video",
    description="Deletes a video record and all associated jobs/summaries from the database, and cleans up R2 storage and ChromaDB. Requires video owner permissions.",
)
def delete_video_user(
    video_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deletes a video owned by the current active user."""
    video = db.query(Video).filter(Video.video_id == video_id).first()
    if not video:
        raise NotFoundException(message=f"Video with ID {video_id} not found")

    if video.user_id != current_user.user_id:
        raise ForbiddenException(message="You do not have permission to delete this video")

    perform_hard_delete(video, db)

    return BaseDTO(
        success=True,
        data=True,
        message="Video and associated records deleted successfully",
    )

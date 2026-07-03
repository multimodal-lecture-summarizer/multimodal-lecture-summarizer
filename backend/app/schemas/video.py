import uuid
from datetime import datetime
from typing import Optional
from pydantic import Field, HttpUrl
from app.schemas.base import CamelModel
from app.core.constants import VideoStatus


class VideoBase(CamelModel):
    original_url: Optional[str] = Field(
        None, description="The original YouTube URL or source url if applicable"
    )
    duration: Optional[float] = Field(
        None, description="The duration of the video in seconds"
    )
    language: str = Field(
        "en", description="The language of the speech in the video"
    )
    status: VideoStatus = Field(
        VideoStatus.PENDING,
        description="The current processing status of the video",
    )


class VideoCreate(CamelModel):
    original_url: Optional[str] = Field(
        None, description="The YouTube video URL to import"
    )
    language: Optional[str] = Field(
        "en", description="Speech language parameter for transcription"
    )


class VideoDTO(VideoBase):
    video_id: uuid.UUID = Field(
        ..., description="The unique UUID of the video"
    )
    user_id: uuid.UUID = Field(
        ..., description="The UUID of the user who uploaded the video"
    )
    file_path: Optional[str] = Field(
        None, description="The storage file path of the processed video"
    )
    uploaded_at: datetime = Field(
        ..., description="The timestamp when the video was uploaded"
    )


class VideoStandardBase(CamelModel):
    max_duration: int = Field(
        3600, description="The maximum allowed video duration in seconds"
    )
    allowed_formats: str = Field(
        "mp4,avi,mkv",
        description="Comma-separated string of allowed video formats",
    )
    max_file_size: int = Field(
        500, description="The maximum allowed video file size in Megabytes"
    )
    min_audio_quality: float = Field(
        0.0,
        description="The minimum allowed Signal-to-Noise Ratio (SNR) for audio",
    )


class VideoStandardDTO(VideoStandardBase):
    standard_id: int = Field(
        ..., description="The unique identifier for the video standard config"
    )

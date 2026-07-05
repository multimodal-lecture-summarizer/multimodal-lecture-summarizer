import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import Field, HttpUrl
from app.schemas.base import CamelModel
from app.core.constants import VideoStatus


class VideoMetadataBase(CamelModel):
    fps: float = Field(..., description="Frames per second of the video")
    frame_count: int = Field(..., description="Total frame count of the video")
    width: int = Field(..., description="Width of the video in pixels")
    height: int = Field(..., description="Height of the video in pixels")
    video_source: str = Field(..., description="Source of the video (e.g. youtube)")
    video_path: str = Field(..., description="Original video file path/key")


class VideoMetadataDTO(VideoMetadataBase):
    video_id: uuid.UUID = Field(
        ..., description="The unique UUID of the video this metadata belongs to"
    )


class VideoSceneBase(CamelModel):
    scene_index: int = Field(..., description="Sequence number of the scene (1-indexed)")
    start_seconds: float = Field(..., description="Scene start time in seconds")
    end_seconds: float = Field(..., description="Scene end time in seconds")
    start_timecode: str = Field(..., description="Scene start timecode (HH:MM:SS.mmm)")
    end_timecode: str = Field(..., description="Scene end timecode (HH:MM:SS.mmm)")
    start_frame: int = Field(..., description="Scene start frame number")
    end_frame: int = Field(..., description="Scene end frame number")
    keyframe_path: str = Field(..., description="Storage key/path of the keyframe image")
    keyframe_url: str = Field(..., description="Publicly accessible URL of the keyframe image")
    caption: str = Field(..., description="AI description of the keyframe slide/image")
    script: Optional[str] = Field(None, description="The transcribed script text of this scene")


class VideoSceneDTO(VideoSceneBase):
    scene_id: uuid.UUID = Field(..., description="The unique UUID of this scene")
    video_id: uuid.UUID = Field(..., description="The unique UUID of the video this scene belongs to")


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
    video_metadata: Optional[VideoMetadataDTO] = Field(
        None, description="The detailed video dimensions and source info"
    )
    scenes: Optional[List[VideoSceneDTO]] = Field(
        None, description="The detected scenes list with keyframes and scripts"
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

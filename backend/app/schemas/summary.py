import uuid
from typing import List, Any, Optional
from pydantic import Field, model_validator
from app.schemas.base import CamelModel
from app.schemas.video import VideoObjectsSummaryDTO


class ChapterDTO(CamelModel):
    title: str = Field(..., description="The title of this video chapter")
    start_time: float = Field(
        ..., description="The start timestamp in seconds for this chapter"
    )
    end_time: float = Field(
        ..., description="The end timestamp in seconds for this chapter"
    )
    summary: str = Field(
        ..., description="A short summary of what is discussed in this chapter"
    )


class KeyframeDTO(CamelModel):
    timestamp: float = Field(
        ..., description="The timestamp of the keyframe in the video"
    )
    image_url: str = Field(
        ..., description="The public URL of the extracted keyframe image"
    )
    description: str = Field(
        ..., description="An AI-generated description of the keyframe visual"
    )
    importance_score: float = Field(
        ...,
        description="The calculated visual/semantic importance score (0.0 to 1.0)",
    )


class WordTokenDTO(CamelModel):
    word: str = Field(..., description="The word text")
    start: float = Field(..., description="Start timestamp of the word in seconds")
    end: float = Field(..., description="End timestamp of the word in seconds")


class TranscriptLineDTO(CamelModel):
    speaker: str = Field(default="SPEAKER_01", description="Identifier of the speaker")
    start: float = Field(..., description="Start timestamp of the line segment in seconds")
    end: float = Field(..., description="End timestamp of the line segment in seconds")
    text: str = Field(..., description="The raw transcript text of this segment")
    words: List[WordTokenDTO] = Field(default=[], description="Word level tokens with timestamps")


class SummaryBase(CamelModel):
    summary_text: str = Field(
        ..., description="The overall abstractive summary of the video"
    )
    transcript_text: str = Field(
        ..., description="The full transcription text of the video speech"
    )
    model_used: str = Field(
        ..., description="The AI LLM model name used for generation"
    )
    processing_time: float = Field(
        ..., description="The total time in seconds taken to process the video"
    )


class SummaryDTO(SummaryBase):
    summary_id: uuid.UUID = Field(
        ..., description="The unique UUID of this summary"
    )
    video_id: uuid.UUID = Field(
        ..., description="The UUID of the video this summary belongs to"
    )
    chapters: List[ChapterDTO] = Field(
        default=[], description="List of segmented video chapters"
    )
    keyframes: List[KeyframeDTO] = Field(
        default=[], description="List of visually important keyframes"
    )
    transcript_segments: List[TranscriptLineDTO] = Field(
        default=[], description="The list of segmented speech transcript segments with word-level timestamps"
    )
    key_takeaways: List[Any] = Field(
        default=[], description="List of key takeaways or core events"
    )
    objects_summary_json: Optional[VideoObjectsSummaryDTO] = Field(
        default=None, description="Global object tracking and detection summary across the video"
    )
    summary_length: Optional[str] = Field(
        default="medium", description="Active summary length tier: short | medium | detailed"
    )
    summary_variants_json: Optional[Any] = Field(
        default=None, description="Multi-variant summary cache keyed by length tier"
    )
    word_count: Optional[int] = Field(
        default=None, description="Calculated word count of active summary"
    )
    reading_time_sec: Optional[int] = Field(
        default=None, description="Estimated reading time in seconds"
    )
    video_title: Optional[str] = Field(
        default=None, description="Current clean/healed title of the video"
    )


    @model_validator(mode="before")
    @classmethod
    def normalize_objects_summary(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "objects_summary" in data and "objects_summary_json" not in data:
                data["objects_summary_json"] = data["objects_summary"]
            elif "video_objects_summary" in data and "objects_summary_json" not in data:
                data["objects_summary_json"] = data["video_objects_summary"]
        return data


class ResummarizeRequest(CamelModel):
    target_length: str = Field(
        default="medium",
        description="Target cognitive depth / summary length: short | medium | detailed"
    )
    force_regenerate: bool = Field(
        default=False,
        description="If True, bypass multi-variant cache and regenerate fresh summary"
    )


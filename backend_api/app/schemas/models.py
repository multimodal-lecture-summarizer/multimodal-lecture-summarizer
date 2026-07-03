"""Pydantic schemas — API contract definitions (request/response models).

Migrated from: src/mls/models.py
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Modality(str, Enum):
    AUDIO = "audio"
    SPEAKER = "speaker"
    VISUAL = "visual"
    SEMANTIC = "semantic"
    TIMELINE = "timeline"
    TEXT = "text"


class WordSpan(BaseModel):
    text: str
    start_sec: float
    end_sec: float
    confidence: float | None = None


class Utterance(BaseModel):
    speaker_id: str
    start_sec: float
    end_sec: float
    text: str
    words: list[WordSpan] = Field(default_factory=list)


class Scene(BaseModel):
    scene_id: int
    start_sec: float
    end_sec: float
    keyframe_path: str | None = None


class SlideContent(BaseModel):
    scene_id: int
    timestamp_sec: float
    ocr_text: str = ""
    caption: str = ""
    clip_embedding: list[float] | None = None
    image_path: str | None = None


class Chapter(BaseModel):
    chapter_id: int
    title: str
    start_sec: float
    end_sec: float
    utterance_ids: list[int] = Field(default_factory=list)
    slide_ids: list[int] = Field(default_factory=list)
    summary: str = ""


class LectureArtifacts(BaseModel):
    """Aggregated outputs from all pipeline stages."""
    video_path: str
    duration_sec: float
    utterances: list[Utterance] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)
    slides: list[SlideContent] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    full_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- Request/Response schemas ---

class VideoUploadResponse(BaseModel):
    job_id: str
    status: str


class VideoStatusResponse(BaseModel):
    video_id: str
    status: str
    progress: int = 0
    current_stage: str | None = None


class VideoResultResponse(BaseModel):
    video_id: str
    transcript: list[Utterance] = Field(default_factory=list)
    summary: str = ""
    chapters: list[Chapter] = Field(default_factory=list)
    keyframes: list[str] = Field(default_factory=list)


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    question: str

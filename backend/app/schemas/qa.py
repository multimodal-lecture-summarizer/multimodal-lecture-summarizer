import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import Field
from app.schemas.base import CamelModel


class QAAskRequest(CamelModel):
    question: str = Field(
        ...,
        min_length=3,
        description="The natural language question about the video content",
    )


class QACitationDTO(CamelModel):
    start_seconds: float = Field(..., description="Timestamp start in seconds")
    end_seconds: float = Field(..., description="Timestamp end in seconds")
    timecode: str = Field(..., description="Formated timecode MM:SS")
    keyframe_url: Optional[str] = Field(default="", description="Keyframe image URL if available")
    snippet: str = Field(..., description="Text snippet of the chunk")


class QALogDTO(CamelModel):
    qa_id: uuid.UUID = Field(
        ..., description="The unique UUID of the Q&A log entry"
    )
    video_id: uuid.UUID = Field(
        ..., description="The UUID of the video this Q&A is related to"
    )
    user_id: uuid.UUID = Field(
        ..., description="The UUID of the user who asked the question"
    )
    question: str = Field(..., description="The question asked by the user")
    answer: str = Field(..., description="The AI-generated answer using RAG")
    retrieved_chunks: Optional[List[str]] = Field(
        default=[],
        description="List of transcript context chunks retrieved from ChromaDB",
    )
    reference_time: Optional[float] = Field(
        default=None,
        description="Primary timestamp in seconds to seek video",
    )
    citations: Optional[List[QACitationDTO]] = Field(
        default=[],
        description="List of structured citations with timestamps and keyframe URLs",
    )
    asked_at: datetime = Field(
        ..., description="The timestamp when the question was asked"
    )

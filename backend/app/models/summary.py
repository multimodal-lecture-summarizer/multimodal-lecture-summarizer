import uuid
from typing import Optional, Any
from sqlalchemy import String, Float, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class Summary(Base):
    __tablename__ = "summaries"

    summary_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.video_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    chapters_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    keyframes_json: Mapped[Any] = mapped_column(JSON, nullable=False)
    transcript_text: Mapped[str] = mapped_column(Text, nullable=False)
    model_used: Mapped[str] = mapped_column(String(100), nullable=False)
    processing_time: Mapped[float] = mapped_column(Float, nullable=False)

    # Relationships
    video: Mapped["Video"] = relationship("Video", back_populates="summary")

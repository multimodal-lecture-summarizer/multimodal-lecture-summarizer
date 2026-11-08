import uuid
from datetime import datetime
from typing import Optional, Any
from sqlalchemy import String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    eval_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    video_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.video_id", ondelete="CASCADE"),
        nullable=True,
    )
    wer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rouge_1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rouge_2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rouge_l: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bertscore_f1: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    od_precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    od_recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    od_map50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    od_map50_95: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    reference_transcript: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reference_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details_json: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    video: Mapped[Optional["Video"]] = relationship("Video")

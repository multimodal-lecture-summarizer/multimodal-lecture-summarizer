import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import String, Integer, Float, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from app.core.constants import VideoStatus


class Video(Base):
    __tablename__ = "videos"

    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    )
    original_url: Mapped[Optional[str]] = mapped_column(
        String(2048), nullable=True
    )
    file_path: Mapped[Optional[str]] = mapped_column(
        String(1024), nullable=True
    )
    duration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    language: Mapped[str] = mapped_column(String(50), default="en")
    status: Mapped[VideoStatus] = mapped_column(
        Enum(VideoStatus), default=VideoStatus.PENDING, nullable=False
    )
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="videos")
    jobs: Mapped[List["Job"]] = relationship(
        "Job", back_populates="video", cascade="all, delete-orphan"
    )
    summary: Mapped[Optional["Summary"]] = relationship(
        "Summary", back_populates="video", cascade="all, delete-orphan"
    )
    qa_logs: Mapped[List["QALog"]] = relationship(
        "QALog", back_populates="video", cascade="all, delete-orphan"
    )


class VideoStandard(Base):
    __tablename__ = "video_standards"

    standard_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    max_duration: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3600
    )  # in seconds
    allowed_formats: Mapped[str] = mapped_column(
        String(255), nullable=False, default="mp4,avi,mkv"
    )
    max_file_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=500
    )  # in MB
    min_audio_quality: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )  # SNR value or similar threshold

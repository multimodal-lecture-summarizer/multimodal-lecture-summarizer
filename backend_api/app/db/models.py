"""SQLAlchemy models — PostgreSQL database schema."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    videos = relationship("Video", back_populates="user")


class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    title = Column(String(500), nullable=False)
    source_path = Column(String(1000))
    youtube_url = Column(String(500))
    duration_sec = Column(Float)
    status = Column(
        Enum("queued", "processing", "done", "failed", name="video_status"),
        default="queued",
    )
    config_stack = Column(String(50), default="hybrid")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))

    user = relationship("User", back_populates="videos")
    results = relationship("VideoResult", back_populates="video", uselist=False)


class VideoResult(Base):
    __tablename__ = "video_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), unique=True, nullable=False)
    transcript_json = Column(Text)  # JSON serialized transcript
    summary = Column(Text)
    chapters_json = Column(Text)  # JSON serialized chapters
    keyframes_dir = Column(String(500))
    pipeline_used = Column(String(200))
    processing_time_sec = Column(Float)
    wer = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    video = relationship("Video", back_populates="results")

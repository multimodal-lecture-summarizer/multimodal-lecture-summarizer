"""Video service — CRUD database operations for videos."""

from __future__ import annotations

from typing import Any


async def get_user_videos(user_id: str) -> list[dict[str, Any]]:
    """Fetch all videos for a user from PostgreSQL."""
    # TODO: query via SQLAlchemy async session
    return []


async def get_video_by_id(video_id: str) -> dict[str, Any] | None:
    """Fetch a single video with its results."""
    # TODO: query via SQLAlchemy async session
    return None


async def delete_video(video_id: str) -> bool:
    """Delete a video record and its associated files."""
    # TODO: delete from DB + filesystem
    return True

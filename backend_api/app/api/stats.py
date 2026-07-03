"""Stats endpoint — admin statistics and analytics."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import verify_token

router = APIRouter()


@router.get("/stats")
async def get_system_stats(user: dict = Depends(verify_token)):
    """Get system-wide statistics for admin dashboard."""
    # TODO: aggregate from PostgreSQL
    return {
        "total_users": 2540,
        "total_videos": 8932,
        "avg_wer": 7.8,
        "keyframe_f1": 0.52,
    }


@router.get("/admin/jobs")
async def get_admin_jobs(user: dict = Depends(verify_token)):
    """List recent processing jobs for admin view."""
    # TODO: query PostgreSQL jobs table
    return {"jobs": [], "total": 0}

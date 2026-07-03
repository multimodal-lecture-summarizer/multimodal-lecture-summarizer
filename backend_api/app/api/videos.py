"""Videos endpoint — CRUD operations for processed videos."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.security import verify_token

router = APIRouter()


@router.get("/videos")
async def list_videos(user: dict = Depends(verify_token)):
    """List all videos for the authenticated user."""
    # TODO: query PostgreSQL via video_service
    return {"videos": [], "total": 0}


@router.get("/videos/{video_id}/status")
async def get_video_status(video_id: str, user: dict = Depends(verify_token)):
    """Get processing status of a video job."""
    # TODO: check Celery task status + DB
    return {"video_id": video_id, "status": "processing", "progress": 45}


@router.get("/videos/{video_id}/results")
async def get_video_results(video_id: str, user: dict = Depends(verify_token)):
    """Get processing results: transcript, summary, chapters, keyframes."""
    # TODO: fetch from DB
    return {"video_id": video_id, "transcript": [], "summary": "", "chapters": [], "keyframes": []}


@router.delete("/videos/{video_id}")
async def delete_video(video_id: str, user: dict = Depends(verify_token)):
    """Delete a video and its artifacts."""
    # TODO: delete from DB + filesystem
    return {"deleted": video_id}


@router.post("/videos/{video_id}/qa")
async def ask_question(video_id: str, body: dict, user: dict = Depends(verify_token)):
    """RAG-based Q&A on a processed video."""
    question = body.get("question", "")
    # TODO: query ChromaDB + LLM
    return {"answer": "", "references": [], "question": question}

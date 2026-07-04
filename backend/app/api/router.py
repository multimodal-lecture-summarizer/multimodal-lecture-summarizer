from fastapi import APIRouter
from app.api.v1 import auth, videos, jobs, summaries, qa, stats, users

api_router = APIRouter()

# Register sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(videos.router, prefix="/videos", tags=["Videos Management"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Background Jobs"])
api_router.include_router(
    summaries.router, prefix="/summaries", tags=["AI Summarization"]
)
api_router.include_router(qa.router, prefix="/qa", tags=["Interactive Q&A"])
api_router.include_router(stats.router, prefix="/stats", tags=["Admin Reports"])
api_router.include_router(users.router, prefix="/users", tags=["Users Management"])

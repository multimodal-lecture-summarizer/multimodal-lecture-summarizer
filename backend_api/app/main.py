"""Backend API — FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import upload, videos, stats, auth
from app.core.config import settings

app = FastAPI(
    title="Video Summarization API",
    description="FastAPI backend cho hệ thống tóm tắt video bài giảng đa phương thức",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(videos.router, prefix="/api", tags=["videos"])
app.include_router(stats.router, prefix="/api", tags=["stats"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "backend_api"}

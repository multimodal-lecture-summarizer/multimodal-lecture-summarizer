import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import engine, Base
from app.api.router import api_router
from app.middleware.exception_handler import register_exception_handlers

# Ensure models are imported so SQLAlchemy metadata registers tables
import app.models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Automatically initializes PostgreSQL tables on startup.
    """
    # Create tables automatically for local testing
    Base.metadata.create_all(bind=engine)

    # Create mock storage directory for video and keyframe static files serving
    mock_dir = os.path.join(os.getcwd(), "storage", "mock_r2_bucket", "keyframes")
    if not os.path.exists(mock_dir):
        os.makedirs(mock_dir, exist_ok=True)

    # Place a dummy keyframe slide if none exists for demo convenience
    for slide in ["slide1.png", "slide2.png", "code1.png"]:
        slide_path = os.path.join(mock_dir, slide)
        if not os.path.exists(slide_path):
            with open(slide_path, "wb") as f:
                # Write minimal valid 1x1 PNG bytes
                f.write(
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x01\x00\x00\x0c\x00\x01\x04p\xad\x04\x00\x00\x00\x00IEND\xaeB`\x82"
                )

    yield


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "FastAPI Backend for the Multimodal AI-Based Video Summarization System.\n\n"
        "Features:\n"
        "- Auto conversion of JSON keys between frontend (camelCase) and backend (snake_case).\n"
        "- Standard BaseDTO envelope wrapper for all responses.\n"
        "- Global error handling structure returning detailed error context.\n"
        "- RAG Q&A integration via ChromaDB and Groq API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register global exception handlers
register_exception_handlers(app)

# Mount local mock static directory to serve mock R2 keyframes images
mock_storage_path = os.path.join(os.getcwd(), "storage", "mock_r2_bucket")
if not os.path.exists(mock_storage_path):
    os.makedirs(mock_storage_path, exist_ok=True)
app.mount("/static/mock_r2", StaticFiles(directory=mock_storage_path), name="mock_r2")

# Register main API router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["General"])
def root_endpoint():
    """Health check root endpoint."""
    return {
        "appName": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "status": "healthy",
        "documentation": "/docs",
    }

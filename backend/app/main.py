import os
import logging
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Verifies database connections (SQL and Vector) with retry and fallback logic.
    Automatically initializes database tables on startup.
    """
    from app.core.database import verify_db_connection
    from app.services.chromadb import chromadb_service
    from app.services.r2 import r2_service

    # Suppress verbose loggers from dependencies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # Verify main relational database connection (PostgreSQL/SQLite)
    verify_db_connection(retries=5, delay=2.0)

    # Verify vector database connection (ChromaDB)
    chromadb_service.verify_connection(retries=5, delay=2.0)

    # Verify Cloudflare R2 storage connection
    r2_service.verify_connection()

    # Verify Celery Broker connection (Redis)
    try:
        from redis import Redis
        redis_client = Redis.from_url(settings.CELERY_BROKER_URL)
        redis_client.ping()
        logger.info("Connected to Redis (Celery Broker) successfully.")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}. Check CELERY_BROKER_URL config.")

    # Create tables automatically for local testing
    Base.metadata.create_all(bind=engine)

    # Initialize default database values (Admin/User accounts and standards)
    from app.core.database import initialize_database_data
    initialize_database_data()

    # Create mock storage directory for video and keyframe static files serving
    # The storage folder is in the project root, one level up from the backend directory
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    mock_dir = os.path.join(project_root, "storage", "mock_r2_bucket", "keyframes")
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

    # Start background polling loop for Celery jobs
    import asyncio
    
    async def poll_celery_jobs():
        logger.info("Started background Celery job status synchronization loop.")
        from app.core.database import SessionLocal
        from app.models.job import Job
        from app.core.constants import JobStatus
        from app.api.v1.jobs import sync_job_status
        
        while True:
            try:
                await asyncio.sleep(10)  # check every 10 seconds
                db = SessionLocal()
                try:
                    active_jobs = db.query(Job).filter(Job.status.in_([JobStatus.PENDING, JobStatus.RUNNING])).all()
                    for job in active_jobs:
                        sync_job_status(job, db)
                finally:
                    db.close()
            except asyncio.CancelledError:
                logger.info("Background Celery job status synchronization loop cancelled.")
                break
            except Exception as exc:
                logger.error(f"Error in background Celery job polling loop: {exc}")

    polling_task = asyncio.create_task(poll_celery_jobs())

    try:
        yield
    finally:
        polling_task.cancel()
        try:
            await polling_task
        except asyncio.CancelledError:
            pass


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
# The storage folder is in the project root, one level up from the backend directory
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
mock_storage_path = os.path.join(project_root, "storage", "mock_r2_bucket")
if not os.path.exists(mock_storage_path):
    os.makedirs(mock_storage_path, exist_ok=True)
app.mount("/static/mock_r2", StaticFiles(directory=mock_storage_path), name="mock_r2")

# Register main API router with documented standard error responses matching BaseDTO
from typing import Any
from app.schemas.base import BaseDTO

app.include_router(
    api_router,
    prefix=settings.API_V1_STR,
    responses={
        400: {
            "model": BaseDTO[Any],
            "description": "Validation or Client Error",
            "content": {
                "application/json": {
                    "examples": {
                        "validation_error": {
                            "summary": "Validation Error",
                            "value": {
                                "success": False,
                                "data": None,
                                "error": {
                                    "errorCode": "VALIDATION_ERROR",
                                    "validationErrors": [
                                        {
                                            "field": "email",
                                            "location": ["body", "email"],
                                            "type": "value_error.any_str.min_length",
                                            "message": "Field length must be at least 3 characters"
                                        }
                                    ]
                                },
                                "code": 400,
                                "message": "Request validation failed",
                                "metadata": None
                            }
                        },
                        "already_exists": {
                            "summary": "Resource Already Exists",
                            "value": {
                                "success": False,
                                "data": None,
                                "error": {
                                    "errorCode": "ALREADY_EXISTS",
                                    "details": None
                                },
                                "code": 400,
                                "message": "The email address is already registered.",
                                "metadata": None
                            }
                        }
                    }
                }
            }
        },
        401: {
            "model": BaseDTO[Any],
            "description": "Unauthorized Access",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": {
                            "errorCode": "UNAUTHORIZED",
                            "details": None
                        },
                        "code": 401,
                        "message": "Incorrect email or password",
                        "metadata": None
                    }
                }
            }
        },
        403: {
            "model": BaseDTO[Any],
            "description": "Forbidden Access",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": {
                            "errorCode": "FORBIDDEN",
                            "details": None
                        },
                        "code": 403,
                        "message": "The user does not have enough privileges",
                        "metadata": None
                    }
                }
            }
        },
        404: {
            "model": BaseDTO[Any],
            "description": "Resource Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": {
                            "errorCode": "NOT_FOUND",
                            "details": None
                        },
                        "code": 404,
                        "message": "Video with ID 123e4567-e89b-12d3-a456-426614174000 not found",
                        "metadata": None
                    }
                }
            }
        },
        500: {
            "model": BaseDTO[Any],
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "data": None,
                        "error": {
                            "errorCode": "INTERNAL_SERVER_ERROR",
                            "exceptionType": "DatabaseError"
                        },
                        "code": 500,
                        "message": "An unexpected error occurred. Please contact the administrator.",
                        "metadata": None
                    }
                }
            }
        }
    },
)


# Customize OpenAPI schema to remove default 422 validation errors (as we handle them as 400)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Remove all 422 responses from OpenAPI schema since we return 400 for validation errors
    for path in openapi_schema.get("paths", {}).values():
        for method in path.values():
            if "422" in method.get("responses", {}):
                del method["responses"]["422"]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


@app.get("/", tags=["General"])
def root_endpoint():
    """Health check root endpoint."""
    # Trigger uvicorn reload on edit
    return {
        "appName": settings.APP_NAME,
        "environment": settings.APP_ENV,
        "status": "healthy",
        "documentation": "/docs",
    }

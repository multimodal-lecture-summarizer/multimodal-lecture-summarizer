"""Worker configuration — shared settings for AI workers."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class WorkerSettings(BaseSettings):
    """AI Worker settings loaded from environment variables."""

    # Celery / Redis
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # GPU
    CUDA_VISIBLE_DEVICES: str = "0"

    # Audio module
    WHISPERX_MODEL: str = "large-v3"
    WHISPERX_BATCH_SIZE: int = 16
    WHISPERX_COMPUTE_TYPE: str = "float16"

    # Visual module
    SCENE_THRESHOLD: float = 27.0
    KEYFRAME_STRATEGY: str = "middle"
    FLORENCE_DEVICE: str = "cpu"

    # Fusion module
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    CHROMADB_PERSIST_DIR: str = "./cache/chroma"

    # API Keys (injected from .env)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "qwen/qwen-2.5-7b-instruct"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    HF_TOKEN: str = ""
    ASSEMBLYAI_API_KEY: str = ""
    DEEPGRAM_API_KEY: str = ""

    # Cloudflare R2 (injected from .env)
    CF_R2_ACCOUNT_ID: str = ""
    CF_R2_ACCESS_KEY_ID: str = ""
    CF_R2_SECRET_ACCESS_KEY: str = ""
    CF_R2_BUCKET_NAME: str = "lecture-summarizer-assets"
    CF_R2_PUBLIC_URL: str = "https://pub-your-bucket-id.r2.dev"

    # Paths
    OUTPUT_DIR: str = "./outputs"
    CACHE_DIR: str = "./cache"

    class Config:
        env_file = ["backend/.env", ".env"]
        case_sensitive = True
        extra = "ignore"


worker_settings = WorkerSettings()

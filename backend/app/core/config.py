import os
from typing import List, Union
from pydantic import AnyHttpUrl, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Annotated


def parse_cors_origins(v: Union[str, List[str]]) -> List[str]:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, (list, str)):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # General App Config
    APP_NAME: str = "AI Video Summarizer Backend"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_MIN_32_CHARS_GOES_HERE"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 11520  # 8 days

    # CORS
    BACKEND_CORS_ORIGINS: Annotated[
        List[str], BeforeValidator(parse_cors_origins)
    ] = ["*"]

    # PostgreSQL Connection
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "lecture_summarizer"

    @property
    def sqlalchemy_database_uri(self) -> str:
        uri = f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        if "aivencloud.com" in self.POSTGRES_SERVER:
            uri += "?sslmode=require"
        return uri

    # Cloudflare R2 / S3 Storage
    CF_R2_ACCOUNT_ID: str = ""
    CF_R2_ACCESS_KEY_ID: str = ""
    CF_R2_SECRET_ACCESS_KEY: str = ""
    CF_R2_BUCKET_NAME: str = "lecture-summarizer-assets"
    CF_R2_PUBLIC_URL: str = "https://pub-your-bucket-id.r2.dev"

    # Groq API
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama3-8b-8192"

    # ChromaDB
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000
    CHROMADB_COLLECTION_NAME: str = "lecture_transcript_embeddings"

    # Backup Default Video Standards (Actual standards loaded from DB)
    DEFAULT_MAX_DURATION_SECONDS: int = 3600
    DEFAULT_ALLOWED_FORMATS: str = "mp4,avi,mkv"
    DEFAULT_MAX_FILE_SIZE_MB: int = 500


settings = Settings()

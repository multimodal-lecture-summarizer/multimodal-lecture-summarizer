from enum import Enum


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class VideoStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class JobType(str, Enum):
    SUMMARIZE = "summarize"
    QA = "qa"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SystemLimits(int, Enum):
    MIN_PASSWORD_LENGTH = 6
    TOKEN_EXPIRY_DAYS = 8


class ErrorCodes(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    DATABASE_ERROR = "DATABASE_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VIDEO_LIMIT_EXCEEDED = "VIDEO_LIMIT_EXCEEDED"

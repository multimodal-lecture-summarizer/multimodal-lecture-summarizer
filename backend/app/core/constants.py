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

    @property
    def default_message(self) -> str:
        messages = {
            ErrorCodes.VALIDATION_ERROR: "Validation failed",
            ErrorCodes.UNAUTHORIZED: "Unauthorized access",
            ErrorCodes.FORBIDDEN: "Access forbidden",
            ErrorCodes.NOT_FOUND: "Resource not found",
            ErrorCodes.ALREADY_EXISTS: "Resource already exists",
            ErrorCodes.DATABASE_ERROR: "Database error occurred",
            ErrorCodes.EXTERNAL_API_ERROR: "External API integration error",
            ErrorCodes.INTERNAL_SERVER_ERROR: "Internal server error occurred",
            ErrorCodes.VIDEO_LIMIT_EXCEEDED: "Video standard limit exceeded",
        }
        return messages.get(self, "An error occurred")

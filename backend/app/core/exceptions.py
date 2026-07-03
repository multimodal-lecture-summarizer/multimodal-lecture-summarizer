from typing import Any, Optional
from app.core.constants import ErrorCodes


class AppException(Exception):
    def __init__(
        self,
        message: str,
        error_code: ErrorCodes = ErrorCodes.INTERNAL_SERVER_ERROR,
        status_code: int = 500,
        details: Optional[Any] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class AuthException(AppException):
    def __init__(
        self,
        message: str = "Unauthorized access",
        error_code: ErrorCodes = ErrorCodes.UNAUTHORIZED,
        status_code: int = 401,
        details: Optional[Any] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class ForbiddenException(AppException):
    def __init__(
        self,
        message: str = "Access forbidden",
        error_code: ErrorCodes = ErrorCodes.FORBIDDEN,
        status_code: int = 403,
        details: Optional[Any] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class NotFoundException(AppException):
    def __init__(
        self,
        message: str = "Resource not found",
        error_code: ErrorCodes = ErrorCodes.NOT_FOUND,
        status_code: int = 404,
        details: Optional[Any] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class AlreadyExistsException(AppException):
    def __init__(
        self,
        message: str = "Resource already exists",
        error_code: ErrorCodes = ErrorCodes.ALREADY_EXISTS,
        status_code: int = 400,
        details: Optional[Any] = None,
    ):
        super().__init__(message, error_code, status_code, details)


class ValidationException(AppException):
    def __init__(
        self,
        message: str = "Validation failed",
        error_code: ErrorCodes = ErrorCodes.VALIDATION_ERROR,
        status_code: int = 400,
        details: Optional[Any] = None,
    ):
        super().__init__(message, error_code, status_code, details)

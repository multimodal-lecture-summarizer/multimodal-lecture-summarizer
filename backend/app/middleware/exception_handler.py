import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.exceptions import AppException
from app.core.config import settings
from app.core.constants import ErrorCodes
from app.middleware.case_converter import convert_keys_to_camel


def register_exception_handlers(app: FastAPI) -> None:
    """Registers global exception handlers for the FastAPI application."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        """Handler for known application logic exceptions."""
        response_content = {
            "success": False,
            "data": None,
            "error": {
                "errorCode": exc.error_code,
                "details": exc.details,
            },
            "code": exc.status_code,
            "message": exc.message,
            "metadata": None,
        }
        # In development, we can enrich debug info
        if settings.DEBUG:
            response_content["error"]["debugInfo"] = {
                "requestUrl": str(request.url),
                "requestMethod": request.method,
            }

        return JSONResponse(
            status_code=exc.status_code,
            content=response_content,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handler for request validation errors (Pydantic parsing failures)."""
        errors = []
        for error in exc.errors():
            # Format target field location to camelCase
            loc = [
                str(item) if not isinstance(item, str) else item
                for item in error.get("loc", [])
            ]
            # Remove "body", "query", etc. from path if user just wants field name
            field_name = loc[-1] if loc else "unknown"
            from app.middleware.case_converter import snake_to_camel

            if isinstance(field_name, str) and "_" in field_name:
                field_name = snake_to_camel(field_name)

            errors.append(
                {
                    "field": field_name,
                    "location": loc,
                    "type": error.get("type"),
                    "message": error.get("msg"),
                }
            )

        response_content = {
            "success": False,
            "data": None,
            "error": {
                "errorCode": ErrorCodes.VALIDATION_ERROR,
                "validationErrors": errors,
            },
            "code": 400,
            "message": "Request validation failed",
            "metadata": None,
        }

        return JSONResponse(
            status_code=400,
            content=response_content,
        )

    from starlette.exceptions import HTTPException as StarletteHTTPException

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handler for standard HTTP exceptions (like 404 Not Found, 401 Unauthorized)."""
        response_content = {
            "success": False,
            "data": None,
            "error": {
                "errorCode": f"HTTP_{exc.status_code}",
                "details": None,  # Set to None to avoid duplicating the message
            },
            "code": exc.status_code,
            "message": str(exc.detail),
            "metadata": None,
        }
        return JSONResponse(
            status_code=exc.status_code,
            content=response_content,
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Catch-all handler for unhandled server errors."""
        # Print stack trace in console
        traceback.print_exc()

        error_details = {
            "errorCode": ErrorCodes.INTERNAL_SERVER_ERROR,
        }

        # If in debug mode, expose full stack trace to frontend
        if settings.DEBUG:
            error_details["exceptionType"] = exc.__class__.__name__
            error_details["stackTrace"] = traceback.format_exc().split("\n")
            message = f"Debug Exception: {exc.__class__.__name__} - {str(exc)}"
        else:
            message = "An unexpected error occurred. Please contact the administrator."

        response_content = {
            "success": False,
            "data": None,
            "error": error_details,
            "code": 500,
            "message": message,
            "metadata": None,
        }

        return JSONResponse(
            status_code=500,
            content=response_content,
        )

"""Standardized LLM exception hierarchy and exception mapper.

Provides structured errors for transient (retryable) and non-transient (fatal) LLM failures.
"""

from __future__ import annotations

from typing import Any, Optional
import re

SENSITIVE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\.\-]+", re.IGNORECASE),
    re.compile(r"(api[_\-]?key|authorization|token)[\s=:\"]+([^\s,\"\']+)", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Strip API keys, tokens, and Authorization credentials from text."""
    if not text:
        return ""
    sanitized = str(text)
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


class LLMBaseError(Exception):
    """Base exception for all LLM communication and execution errors."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        status_code: Optional[int] = None,
        is_retryable: bool = False,
        raw_error: Optional[Exception] = None,
    ):
        sanitized_msg = sanitize_text(message)
        super().__init__(sanitized_msg)
        self.message = sanitized_msg
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.is_retryable = is_retryable
        self.raw_error = raw_error
        self.error_code: str = self.__class__.__name__

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "is_retryable": self.is_retryable,
        }

    def __str__(self) -> str:
        parts = [f"[{self.error_code}]", self.message]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        return " ".join(parts)


class LLMTimeoutError(LLMBaseError):
    """Raised when an LLM request times out (transient, retryable)."""

    def __init__(
        self,
        message: str = "LLM request timed out",
        provider: str = "",
        model: str = "",
        status_code: int = 408,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=True,
            raw_error=raw_error,
        )
        self.error_code = "LLM_TIMEOUT"


class LLMNetworkError(LLMBaseError):
    """Raised when connection drops, DNS fails, or network is unreachable (transient, retryable)."""

    def __init__(
        self,
        message: str = "Network connection failed during LLM call",
        provider: str = "",
        model: str = "",
        status_code: Optional[int] = None,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=True,
            raw_error=raw_error,
        )
        self.error_code = "LLM_NETWORK_ERROR"


class LLMRateLimitError(LLMBaseError):
    """Raised when HTTP 429 rate limit is encountered (transient, retryable)."""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded (HTTP 429)",
        provider: str = "",
        model: str = "",
        status_code: int = 429,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=True,
            raw_error=raw_error,
        )
        self.error_code = "LLM_RATE_LIMIT"


class LLMProviderError(LLMBaseError):
    """Raised when LLM provider returns 5xx server error (transient, retryable)."""

    def __init__(
        self,
        message: str = "LLM provider server error (5xx)",
        provider: str = "",
        model: str = "",
        status_code: int = 500,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=True,
            raw_error=raw_error,
        )
        self.error_code = "LLM_PROVIDER_ERROR"


class LLMAuthenticationError(LLMBaseError):
    """Raised when authentication/authorization fails (HTTP 401/403) (non-transient, fatal)."""

    def __init__(
        self,
        message: str = "LLM authentication failed (invalid API key or permissions)",
        provider: str = "",
        model: str = "",
        status_code: int = 401,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=False,
            raw_error=raw_error,
        )
        self.error_code = "LLM_AUTH_ERROR"


class LLMBadRequestError(LLMBaseError):
    """Raised when request payload or parameters are invalid (HTTP 400) (non-transient, fatal)."""

    def __init__(
        self,
        message: str = "LLM bad request (invalid payload or model configuration)",
        provider: str = "",
        model: str = "",
        status_code: int = 400,
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=status_code,
            is_retryable=False,
            raw_error=raw_error,
        )
        self.error_code = "LLM_BAD_REQUEST"


class LLMResponseParsingError(LLMBaseError):
    """Raised when LLM output is not valid JSON or violates expected schema (non-transient, fatal)."""

    def __init__(
        self,
        message: str = "Failed to parse or validate LLM JSON response",
        provider: str = "",
        model: str = "",
        raw_error: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            provider=provider,
            model=model,
            status_code=None,
            is_retryable=False,
            raw_error=raw_error,
        )
        self.error_code = "LLM_PARSE_ERROR"


def map_exception_to_llm_error(
    exc: Exception, provider: str = "", model: str = ""
) -> LLMBaseError:
    """Map any raw exception (OpenAI, httpx, requests, standard errors) to standardized LLMBaseError."""
    if isinstance(exc, LLMBaseError):
        if provider and not exc.provider:
            exc.provider = provider
        if model and not exc.model:
            exc.model = model
        return exc

    exc_name = exc.__class__.__name__
    msg = str(exc)

    # 1. Check for Timeout errors
    if (
        "Timeout" in exc_name
        or isinstance(exc, (TimeoutError,))
        or "timeout" in msg.lower()
        or "timed out" in msg.lower()
    ):
        return LLMTimeoutError(
            message=f"Request timed out: {msg}",
            provider=provider,
            model=model,
            raw_error=exc,
        )

    # 2. Check HTTP status code if available on exception (OpenAI APIStatusError, requests.HTTPError, httpx.HTTPStatusError)
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response") and exc.response is not None:
        status_code = getattr(exc.response, "status_code", None)

    if status_code is not None:
        if status_code == 401 or status_code == 403 or "AuthenticationError" in exc_name or "PermissionDeniedError" in exc_name:
            return LLMAuthenticationError(
                message=f"Authentication/permission error ({status_code}): {msg}",
                provider=provider,
                model=model,
                status_code=status_code,
                raw_error=exc,
            )
        if status_code == 429 or "RateLimitError" in exc_name:
            return LLMRateLimitError(
                message=f"Rate limit exceeded ({status_code}): {msg}",
                provider=provider,
                model=model,
                status_code=status_code,
                raw_error=exc,
            )
        if status_code == 400 or status_code == 422 or "BadRequestError" in exc_name or "UnprocessableEntityError" in exc_name:
            return LLMBadRequestError(
                message=f"Bad request ({status_code}): {msg}",
                provider=provider,
                model=model,
                status_code=status_code,
                raw_error=exc,
            )
        if status_code in (500, 502, 503, 504) or (500 <= status_code < 600) or "InternalServerError" in exc_name:
            return LLMProviderError(
                message=f"Provider server error ({status_code}): {msg}",
                provider=provider,
                model=model,
                status_code=status_code,
                raw_error=exc,
            )

    # 3. Check for Connection / Network errors
    if (
        "Connection" in exc_name
        or "Network" in exc_name
        or isinstance(exc, (ConnectionError, ConnectionResetError, ConnectionRefusedError))
        or "failed to establish a new connection" in msg.lower()
        or "connection reset" in msg.lower()
        or "cannot connect" in msg.lower()
    ):
        return LLMNetworkError(
            message=f"Network connection error: {msg}",
            provider=provider,
            model=model,
            raw_error=exc,
        )

    # 4. Check for JSON parse / schema errors
    if isinstance(exc, (ValueError, TypeError)) and ("json" in msg.lower() or "schema" in msg.lower() or "chapter" in msg.lower() or "title" in msg.lower()):
        return LLMResponseParsingError(
            message=f"Response validation error: {msg}",
            provider=provider,
            model=model,
            raw_error=exc,
        )

    # Default to LLMProviderError as generic failure
    return LLMProviderError(
        message=f"Unhandled LLM provider exception ({exc_name}): {msg}",
        provider=provider,
        model=model,
        status_code=status_code or 500,
        raw_error=exc,
    )

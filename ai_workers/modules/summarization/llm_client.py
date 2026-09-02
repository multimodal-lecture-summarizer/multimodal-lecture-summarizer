"""Resilient LLM Client with Tenacity Retry, HTTP Timeout, and Sanitized Structured Logging."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
from tenacity import (
    Retrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_none,
)

from ai_workers.modules.summarization.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBaseError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    map_exception_to_llm_error,
)

logger = logging.getLogger("ai_workers.summarization.llm_client")

DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_DELAY_SECONDS = 2.0
DEFAULT_MAX_DELAY_SECONDS = 8.0

# Patterns for sensitive keys/tokens to sanitize from any logging
SENSITIVE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"gsk_[a-zA-Z0-9_\-]{10,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\.\-]+", re.IGNORECASE),
    re.compile(r"(api[_\-]?key|authorization|token)[\s=:\"]+([^\s,\"\']+)", re.IGNORECASE),
]


def sanitize_text(text: str) -> str:
    """Strip API keys, tokens, and Authorization credentials from log messages."""
    if not text:
        return ""
    sanitized = str(text)
    for pattern in SENSITIVE_PATTERNS:
        sanitized = pattern.sub("[REDACTED]", sanitized)
    return sanitized


def is_retryable_error(exc: BaseException) -> bool:
    """Return True only if the exception represents a transient/retryable error."""
    if isinstance(exc, LLMBaseError):
        return bool(exc.is_retryable)
    # Check if raw exception maps to a retryable error
    mapped = map_exception_to_llm_error(exc if isinstance(exc, Exception) else Exception(str(exc)))
    return bool(mapped.is_retryable)


class LLMClient:
    """Encapsulates resilient API communication with LLM providers (OpenRouter, Groq, OpenAI)."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        initial_delay: float = DEFAULT_INITIAL_DELAY_SECONDS,
        max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
        custom_wait: Any = None,
        default_headers: Optional[dict[str, str]] = None,
    ):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = float(timeout)
        self.max_attempts = int(max_attempts)
        self.initial_delay = float(initial_delay)
        self.max_delay = float(max_delay)
        self.custom_wait = custom_wait
        self.default_headers = default_headers or {}

        # Instantiate underlying OpenAI-compatible client with explicit timeout
        self._client: Optional[OpenAI] = None
        if self.api_key:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                default_headers=self.default_headers,
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self._client is not None)

    def _build_retrying(self) -> Retrying:
        wait_strategy = (
            self.custom_wait
            if self.custom_wait is not None
            else wait_exponential(
                multiplier=1.0,
                min=self.initial_delay,
                max=self.max_delay,
            )
        )
        return Retrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_strategy,
            retry=retry_if_exception(is_retryable_error),
            reraise=True,
        )

    def generate_chat_completion(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        response_format: Optional[dict[str, Any]] = None,
        job_id: str = "unknown",
    ) -> str:
        """Call chat completion with automatic retry, timeout, structured logging, and sanitized errors.

        Args:
            model: Target model identifier.
            prompt: User prompt.
            system_prompt: Optional system prompt.
            temperature: LLM temperature.
            response_format: Optional response format (e.g. {"type": "json_object"}).
            job_id: Celery / pipeline job identifier for structured logging.

        Returns:
            The raw text content returned by the LLM.

        Raises:
            LLMBaseError: Standardized error if all attempts fail or a non-retryable error occurs.
        """
        if not self._client:
            raise LLMAuthenticationError(
                message=f"LLM provider '{self.provider}' is not configured (missing API key).",
                provider=self.provider,
                model=model,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        retrying = self._build_retrying()
        last_attempt_start = time.time()

        for attempt in retrying:
            attempt_number = attempt.retry_state.attempt_number
            start_time = time.time()
            with attempt:
                try:
                    kwargs: dict[str, Any] = {
                        "messages": messages,
                        "model": model,
                        "temperature": temperature,
                        "timeout": self.timeout,
                    }
                    if response_format:
                        kwargs["response_format"] = response_format

                    chat_completion = self._client.chat.completions.create(**kwargs)

                    duration = time.time() - start_time
                    content = chat_completion.choices[0].message.content or ""

                    # Structured success logging
                    log_msg = (
                        f"LLM request succeeded "
                        f"job_id={job_id} "
                        f"provider={self.provider} "
                        f"model={model} "
                        f"attempt={attempt_number} "
                        f"duration={duration:.2f}s"
                    )
                    logger.info(sanitize_text(log_msg))
                    print(f"[{self.provider}][{model}] Request succeeded in {duration:.2f}s (attempt {attempt_number})")
                    return content.strip()

                except Exception as raw_exc:
                    duration = time.time() - start_time
                    llm_err = map_exception_to_llm_error(raw_exc, provider=self.provider, model=model)

                    # Structured failure logging
                    log_msg = (
                        f"LLM request failed "
                        f"job_id={job_id} "
                        f"provider={self.provider} "
                        f"model={model} "
                        f"attempt={attempt_number} "
                        f"error={llm_err.error_code} "
                        f"duration={duration:.2f}s"
                    )
                    logger.warning(sanitize_text(log_msg))
                    print(
                        f"[{self.provider}][{model}] Attempt {attempt_number}/{self.max_attempts} failed "
                        f"({llm_err.error_code}, retryable={llm_err.is_retryable}, {duration:.2f}s): {sanitize_text(str(raw_exc))}"
                    )

                    # Raise standardized LLM exception so Tenacity evaluates retry
                    raise llm_err from raw_exc

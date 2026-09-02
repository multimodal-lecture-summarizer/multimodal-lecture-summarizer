"""Summarization module with resilient LLM clients, standardized errors, and summarizer service."""

from ai_workers.modules.summarization.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBaseError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
    map_exception_to_llm_error,
)
from ai_workers.modules.summarization.llm_client import LLMClient
from ai_workers.modules.summarization.summarizer import Summarizer

__all__ = [
    "Summarizer",
    "LLMClient",
    "LLMBaseError",
    "LLMTimeoutError",
    "LLMNetworkError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMAuthenticationError",
    "LLMBadRequestError",
    "LLMResponseParsingError",
    "map_exception_to_llm_error",
]

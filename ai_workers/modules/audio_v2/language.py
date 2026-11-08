"""Source language shared by upload, the worker, and Faster-Whisper."""

from __future__ import annotations


def normalize_source_language(value: str | None) -> str:
    """Return auto, vi, or en. Any other value is treated as automatic detection."""
    raw = (value or "auto").strip().lower()
    if raw in ("vi", "en"):
        return raw
    return "auto"


def whisper_language_arg(value: str | None) -> str | None:
    """Language passed to Faster-Whisper. None lets the model detect it."""
    normalized = normalize_source_language(value)
    if normalized == "auto":
        return None
    return normalized

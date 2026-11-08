"""Pick the ASR engine for a job."""

from __future__ import annotations

from ai_workers.core.config import worker_settings
from ai_workers.modules.audio_v2.language import normalize_source_language


def _qwen_enabled() -> bool:
    return worker_settings.ASR_ENGLISH_BACKEND == "qwen3"


def uses_qwen(source_language: str | None) -> bool:
    """Qwen3-ASR for explicit English (its aligner has no Vietnamese, so never for vi)."""
    return _qwen_enabled() and normalize_source_language(source_language) == "en"


def routes_by_detection(source_language: str | None) -> bool:
    """`auto` detects the language first; only English moves to Qwen3-ASR."""
    return _qwen_enabled() and normalize_source_language(source_language) == "auto"


def asr_identifier(source_language: str | None) -> str:
    """Model identity for checkpoint fingerprints, so switching engine invalidates stale checkpoints."""
    qwen_id = f"{worker_settings.QWEN_ASR_MODEL}+{worker_settings.QWEN_ALIGNER_MODEL}"
    if uses_qwen(source_language):
        return qwen_id
    if routes_by_detection(source_language):
        # Which engine runs is only known after detection; cover both configurations.
        return f"{worker_settings.FASTER_WHISPER_MODEL}|{qwen_id}"
    return worker_settings.FASTER_WHISPER_MODEL


def build_transcriber(source_language: str | None):
    # Imports stay lazy so the engine classes are resolved at call time, not at module import.
    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

    if uses_qwen(source_language):
        from ai_workers.modules.audio_v2.qwen_transcriber import QwenEnglishTranscriber

        return QwenEnglishTranscriber()
    if routes_by_detection(source_language):
        from ai_workers.modules.audio_v2.routing_transcriber import LanguageRoutingTranscriber

        return LanguageRoutingTranscriber({"model_name": worker_settings.FASTER_WHISPER_MODEL})
    return AudioTranscriber({"model_name": worker_settings.FASTER_WHISPER_MODEL})

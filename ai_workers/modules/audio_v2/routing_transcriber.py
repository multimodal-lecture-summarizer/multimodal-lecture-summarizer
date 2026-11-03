"""`auto` source language: detect with Whisper first, English goes to Qwen3-ASR, the rest stays on Whisper."""

from __future__ import annotations

import gc
import logging
from typing import Any

from faster_whisper import WhisperModel

from ai_workers.modules.audio_v2.qwen_transcriber import QwenEnglishTranscriber
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

logger = logging.getLogger("ai_workers.modules.audio_v2.routing_transcriber")

SAMPLE_RATE = 16000
# Detection looks at the first 3 x 30 s speech windows (VAD skips music/silence).
DETECT_SECONDS = 90
DETECT_SEGMENTS = 3
MIN_ENGLISH_PROBABILITY = 0.7


class LanguageRoutingTranscriber(AudioTranscriber):
    """Used when source_language is `auto` and the English backend is qwen3."""

    def detect_language(self, audio_path: str) -> tuple[str, float]:
        import soundfile as sf

        # process() always hands over the 16 kHz mono WAV it extracted.
        audio, _ = sf.read(audio_path, frames=SAMPLE_RATE * DETECT_SECONDS, dtype="float32")
        model = WhisperModel(
            self._resolve_model_source(),
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.cache_dir),
        )
        try:
            language, probability, _ = model.detect_language(
                audio=audio, vad_filter=True, language_detection_segments=DETECT_SEGMENTS
            )
        finally:
            del model
            gc.collect()
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()
        return language, float(probability)

    def transcribe(self, audio_path: str, source_language: str | None = None) -> dict[str, Any]:
        language, probability = None, 0.0
        try:
            language, probability = self.detect_language(audio_path)
            print(f"[ASR] Detected language '{language}' (p={probability:.2f}) before engine selection.")
        except Exception:
            logger.warning("Language detection failed; using Faster-Whisper auto", exc_info=True)

        if language == "en" and probability >= MIN_ENGLISH_PROBABILITY:
            return QwenEnglishTranscriber().transcribe(audio_path, "en")
        return super().transcribe(audio_path, source_language)

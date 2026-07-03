"""Audio transcriber — WhisperX, FFmpeg, Noise reduction.

Migrated from: src/mls/modules/audio.py
NGƯỜI 1: WhisperX, FFmpeg, Noise reduction
"""

from __future__ import annotations

from typing import Any


class AudioTranscriber:
    """Extract audio waveform and run ASR (WhisperX or API)."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "whisperx")

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video using FFmpeg.

        Args:
            video_path: Path to input video file.
            output_path: Path for extracted audio (WAV).

        Returns:
            Path to extracted audio file.
        """
        # TODO: subprocess call to ffmpeg
        # ffmpeg -i video_path -vn -acodec pcm_s16le -ar 16000 -ac 1 output_path
        return output_path

    def reduce_noise(self, audio_path: str) -> str:
        """Apply noise reduction to audio.

        Returns:
            Path to cleaned audio file.
        """
        # TODO: noisereduce or similar library
        return audio_path

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Run ASR on audio file.

        Returns:
            Dict with 'segments' (word-level timestamps) and 'language'.
        """
        # TODO: WhisperX / AssemblyAI / Deepgram based on provider
        return {"segments": [], "language": "auto"}

    def process(self, video_path: str) -> dict[str, Any]:
        """Full audio pipeline: extract → denoise → transcribe."""
        audio_path = self.extract_audio(video_path, video_path.replace(".mp4", ".wav"))
        audio_path = self.reduce_noise(audio_path)
        result = self.transcribe(audio_path)
        return result

"""Speaker diarization — pyannote, AssemblyAI.

Migrated from: src/mls/modules/speaker.py
NGƯỜI 1: Speaker diarization and merge with ASR segments.
"""

from __future__ import annotations

from typing import Any


class SpeakerDiarizer:
    """Speaker diarization and merge with ASR transcript segments."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "pyannote")

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        """Run speaker diarization on audio.

        Returns:
            List of speaker segments: [{speaker_id, start_sec, end_sec}]
        """
        # TODO: pyannote RTTM → speaker segments
        return []

    def merge_with_transcript(
        self,
        transcript_segments: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict[str, Any]]:
        """Assign speaker_id to each transcript utterance.

        Merges ASR word-level segments with diarization output.
        """
        # TODO: overlap matching algorithm
        return transcript_segments

    def process(self, audio_path: str, transcript_segments: list[dict]) -> list[dict[str, Any]]:
        """Full speaker pipeline: diarize → merge."""
        if self.config.get("skip_if_merged"):
            return transcript_segments
        speakers = self.diarize(audio_path)
        return self.merge_with_transcript(transcript_segments, speakers)

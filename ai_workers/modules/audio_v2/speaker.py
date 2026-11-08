"""Speaker diarization. Pyannote runs only when HF_TOKEN is available."""

from __future__ import annotations

import logging
import os
from typing import Any

from ai_workers.core.config import worker_settings

logger = logging.getLogger("ai_workers.audio.speaker")

MIN_SPEAKER_OVERLAP = 0.30


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers(
    transcript_segments: list[dict],
    speaker_segments: list[dict],
    min_overlap_ratio: float = MIN_SPEAKER_OVERLAP,
) -> list[dict[str, Any]]:
    """Give each utterance the speaker with the largest overlap.

    Overlap below min_overlap_ratio becomes UNKNOWN. Segments are copied.
    """
    assigned: list[dict[str, Any]] = []
    for utterance in transcript_segments or []:
        item = dict(utterance)
        start = float(item.get("start", 0.0) or 0.0)
        end = float(item.get("end", start) or start)
        duration = max(end - start, 1e-6)
        best_speaker = None
        best_overlap = 0.0
        for speaker in speaker_segments or []:
            overlap = _overlap_seconds(
                start,
                end,
                float(speaker.get("start_sec", speaker.get("start", 0.0)) or 0.0),
                float(speaker.get("end_sec", speaker.get("end", 0.0)) or 0.0),
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker.get("speaker_id") or speaker.get("speaker")
        if best_speaker and (best_overlap / duration) >= min_overlap_ratio:
            item["speaker_id"] = str(best_speaker)
        else:
            item["speaker_id"] = "UNKNOWN"
        assigned.append(item)
    return assigned


class SpeakerDiarizer:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "pyannote")
        self.status = "skipped"
        self.reason = "HF_TOKEN is not set"

    def _token(self) -> str:
        return str(
            self.config.get("hf_token")
            or os.getenv("HF_TOKEN")
            or getattr(worker_settings, "HF_TOKEN", "")
            or ""
        ).strip()

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        """Run pyannote when a token is present. Raises if the model cannot run."""
        token = self._token()
        if not token:
            raise RuntimeError("HF_TOKEN is not set")

        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )
        diarization = pipeline(audio_path)
        annotation = getattr(diarization, "speaker_diarization", diarization)
        segments: list[dict[str, Any]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append({
                "speaker_id": str(speaker),
                "start_sec": float(turn.start),
                "end_sec": float(turn.end),
            })
        return segments

    def merge_with_transcript(
        self,
        transcript_segments: list[dict],
        speaker_segments: list[dict],
    ) -> list[dict[str, Any]]:
        """Assign speaker_id from diarization overlap. Low overlap stays UNKNOWN."""
        return assign_speakers(transcript_segments, speaker_segments)

    def process(self, audio_path: str, transcript_segments: list[dict]) -> list[dict[str, Any]]:
        """Diarize when possible. Without a token, return the transcript unchanged."""
        if self.config.get("skip_if_merged"):
            self.status = "skipped"
            self.reason = "skip_if_merged"
            return transcript_segments

        if not self._token():
            self.status = "skipped"
            self.reason = "HF_TOKEN is not set"
            logger.info("[Diarization] Skipped because HF_TOKEN is not set.")
            return [dict(segment) for segment in (transcript_segments or [])]

        try:
            speakers = self.diarize(audio_path)
            merged = self.merge_with_transcript(transcript_segments, speakers)
            self.status = "completed"
            self.reason = ""
            logger.info(f"[Diarization] Assigned speakers across {len(merged)} utterances.")
            return merged
        except Exception as exc:
            self.status = "failed"
            self.reason = str(exc)
            logger.warning(f"[Diarization] Failed without assigning speakers: {exc}")
            return [dict(segment) for segment in (transcript_segments or [])]

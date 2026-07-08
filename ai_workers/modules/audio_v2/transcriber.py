from __future__ import annotations

import os
import subprocess
from typing import Any

from faster_whisper import WhisperModel


class AudioTranscriber:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model_size = self.config.get("model_name", "base.en")
        
        import torch
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

    def extract_audio(self, video_path: str, output_path: str) -> str:
        """Extract audio from video using FFmpeg.

        Args:
            video_path: Path to input video file.
            output_path: Path for extracted audio (WAV).

        Returns:
            Path to extracted audio file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        print(f"Extracting audio using FFmpeg from: {video_path}...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[OK] Extracted audio WAV file: {output_path}")
        return output_path

    def reduce_noise(self, audio_path: str) -> str:
        return audio_path

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Run ASR on audio file with word-level timestamps using Faster-Whisper.

        Returns:
            Dict with 'text', 'segments' (word-level timestamps) and 'language'.
        """
        print(f"Loading Faster-Whisper model ({self.model_size}) on {self.device}...")
        model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

        print(f"Transcribing audio file: {audio_path} using Faster-Whisper with VAD...")
        segments_gen, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=2000),
            word_timestamps=True
        )

        segments = []
        full_text_list = []
        current_merged = None

        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

        for segment in segments_gen:
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            full_text_list.append(segment_text)
            
            words = []
            if segment.words:
                for word in segment.words:
                    words.append({
                        "word": word.word.strip(),
                        "start": word.start,
                        "end": word.end
                    })

            if current_merged is None:
                current_merged = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text,
                    "words": words
                }
            else:
                current_merged["end"] = segment.end
                current_merged["text"] += " " + segment_text
                current_merged["words"].extend(words)
            
            if segment_text.endswith(('.', '?', '!')):
                segments.append(current_merged)
                print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] {current_merged['text']}")
                current_merged = None
        if current_merged is not None:
            segments.append(current_merged)
            print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] {current_merged['text']}")

        full_text = " ".join(full_text_list)
        print(f"[OK] Completed transcription. Text length: {len(full_text)} characters.")
        
        del model
        import gc
        gc.collect()
        if self.device == "cuda":
            import torch
            torch.cuda.empty_cache()

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language
        }

    def process(self, video_path: str) -> dict[str, Any]:
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        self.extract_audio(video_path, audio_path)
        result = self.transcribe(audio_path)
        return result

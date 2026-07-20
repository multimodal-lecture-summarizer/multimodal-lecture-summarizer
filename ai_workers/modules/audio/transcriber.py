"""Audio transcriber — Whisper, FFmpeg, Silero VAD.

Migrated from: src/mls/modules/audio.py
NGƯỜI 1: Whisper, FFmpeg, Silero VAD
"""

from __future__ import annotations

import os
import subprocess
import torch
import soundfile as sf
import librosa
from typing import Any, Union
import numpy as np
from transformers import pipeline


class AudioTranscriber:
    """Extract audio waveform and run ASR using Whisper."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "whisper")
        self.model_name = self.config.get("model_name", "openai/whisper-tiny")
        self.device = 0 if torch.cuda.is_available() else -1

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
        
        # Suppress output to clean logs
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"[OK] Extracted audio WAV file: {output_path}")
        return output_path

    def reduce_noise(self, audio_path: str) -> np.ndarray:
        from ai_workers.modules.common.denoise import get_denoised_audio_array
        return get_denoised_audio_array(audio_path)

    def transcribe(self, audio_input: Union[str, np.ndarray]) -> dict[str, Any]:
        """Run ASR on audio file with word-level timestamps.

        Returns:
            Dict with 'text', 'segments' (word-level timestamps) and 'language'.
        """
        print(f"Transcribing audio using Whisper ({self.model_name})...")
        
        # 1. Read audio data
        if isinstance(audio_input, str):
            audio_data, sr = sf.read(audio_input)
        else:
            audio_data = audio_input
            sr = 16000  # Assume 16kHz for numpy array input
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            
        # Resample if not 16kHz
        if sr != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
            
        vad_model = None
        get_speech_timestamps = None
        asr_pipeline = None

        try:
            # 2. Try loading Silero VAD
            try:
                print("Loading Silero VAD for voice activity check...")
                # Set torch hub dir inside cache to avoid write permission issues
                torch.hub.set_dir(os.path.join(os.getcwd(), "cache", "torch_hub"))
                vad_model, vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    trust_repo=True,
                    force_reload=False
                )
                get_speech_timestamps = vad_utils[0]
            except Exception as e:
                print(f"[Warning] Silero VAD initialization skipped or failed: {e}. Processing raw chunks directly.")
                
            # 3. Load HuggingFace ASR pipeline
            print("Loading Whisper ASR pipeline...")
            asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
                device=self.device
            )
            
            # 4. Transcribe in 30-second chunks
            chunk_sec = 30
            chunk_samples = chunk_sec * sr
            
            segments = []
            full_text_list = []
            
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i+chunk_samples]
                if len(chunk) < 0.5 * sr:
                    continue
                    
                t_start = i / sr
                t_end = min(len(audio_data), i + chunk_samples) / sr
                print(f"-> Processing chunk {t_start:.1f}s - {t_end:.1f}s (Progress: {i / len(audio_data) * 100:.1f}%)...")
                
                # VAD check
                is_silent = False
                if vad_model is not None and get_speech_timestamps is not None:
                    try:
                        chunk_tensor = torch.from_numpy(chunk).float()
                        speech_ts = get_speech_timestamps(chunk_tensor, vad_model, sampling_rate=16000)
                        if not speech_ts:
                            is_silent = True
                    except Exception as vad_err:
                        print(f"[Warning] VAD check failed on chunk {t_start:.1f}s-{t_end:.1f}s: {vad_err}")
                
                if is_silent:
                    segments.append({
                        "start": t_start,
                        "end": t_end,
                        "text": "[Nhạc nền / Im lặng]",
                        "words": []
                    })
                    continue
                
                # Run transcription
                try:
                    res = asr_pipeline(
                        {"array": chunk, "sampling_rate": sr},
                        return_timestamps="word",
                        generate_kwargs={"task": "transcribe"}
                    )
                except Exception as asr_err:
                    print(f"[Error] Failed to transcribe chunk {t_start:.1f}s-{t_end:.1f}s: {asr_err}")
                    continue
                    
                chunk_text = res.get("text", "").strip()
                if not chunk_text:
                    chunk_text = "[Nhạc nền / Im lặng]"
                    
                if chunk_text != "[Nhạc nền / Im lặng]":
                    full_text_list.append(chunk_text)
                
                # Format word timestamps
                words = []
                if chunk_text != "[Nhạc nền / Im lặng]":
                    for c in res.get("chunks", []):
                        w_text = c.get("text", "").strip()
                        ts = c.get("timestamp")
                        if w_text and ts and len(ts) == 2:
                            words.append({
                                "word": w_text,
                                "start": t_start + ts[0],
                                "end": t_start + ts[1]
                            })
                    
                    # If Hugging Face returns words without individual timestamps, interpolate them
                    if chunk_text and not words:
                        words_list = chunk_text.split()
                        w_dur = (t_end - t_start) / len(words_list)
                        for idx, w in enumerate(words_list):
                            words.append({
                                "word": w,
                                "start": t_start + idx * w_dur,
                                "end": t_start + (idx + 1) * w_dur
                            })
                
                segments.append({
                    "start": t_start,
                    "end": t_end,
                    "text": chunk_text,
                    "words": words
                })
                
            full_text = " ".join(full_text_list)
            print(f"[OK] Completed transcription. Text length: {len(full_text)} characters.")
            
            return {
                "text": full_text,
                "segments": segments,
                "language": "auto"
            }
        finally:
            print("Releasing Silero VAD and Whisper ASR models from memory...")
            if asr_pipeline is not None:
                del asr_pipeline
            if vad_model is not None:
                del vad_model
            if get_speech_timestamps is not None:
                del get_speech_timestamps
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[OK] Models released successfully.")

    def process(self, video_path: str, use_denoise: bool = True) -> dict[str, Any]:
        """Full audio pipeline: extract → transcribe."""
        audio_path = video_path.rsplit(".", 1)[0] + ".wav"
        self.extract_audio(video_path, audio_path)
        
        if use_denoise:
            audio_input = self.reduce_noise(audio_path)
        else:
            audio_input = audio_path
            
        result = self.transcribe(audio_input)
        return result

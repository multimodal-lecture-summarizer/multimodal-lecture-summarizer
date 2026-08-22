from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from ai_workers.core.config import worker_settings


class AudioTranscriber:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model_size = self.config.get("model_name", "base.en")
        self.cache_dir = Path(worker_settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Inject cuDNN / cuBLAS DLL paths for CTranslate2 (Faster-Whisper)
        import sys
        site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
        cudnn_bin = site_packages / "nvidia" / "cudnn" / "bin"
        cublas_bin = site_packages / "nvidia" / "cublas" / "bin"
        ctranslate2_dir = site_packages / "ctranslate2"

        paths_to_add = []
        for bin_dir in (cudnn_bin, cublas_bin, ctranslate2_dir):
            if bin_dir.exists():
                paths_to_add.append(str(bin_dir))
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except OSError:
                        pass

        if paths_to_add:
            os.environ["PATH"] = ";".join(paths_to_add) + ";" + os.environ.get("PATH", "")

        import torch

        # CTranslate2 on Windows still expects cuDNN 8 symbols; prefer GPU only when present.
        cudnn8_dll = ctranslate2_dir / "cudnn_ops_infer64_8.dll"
        if not cudnn8_dll.exists():
            cudnn8_dll = cudnn_bin / "cudnn_ops_infer64_8.dll"
        force_device = str(self.config.get("device") or os.environ.get("FASTER_WHISPER_DEVICE", "")).lower()

        if force_device == "cpu" or (torch.cuda.is_available() and not cudnn8_dll.exists()):
            if torch.cuda.is_available() and force_device != "cpu" and not cudnn8_dll.exists():
                print(
                    "[ASR] cuDNN 8 DLL missing for Faster-Whisper; using CPU for ASR "
                    "(Florence/CLIP can still use CUDA)."
                )
            self.device = "cpu"
            self.compute_type = "int8"
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"

    def _resolve_model_source(self) -> str:
        """Prefer a pre-downloaded local model directory when available."""
        local_dir = self.cache_dir / f"faster-whisper-{self.model_size}"
        if (local_dir / "model.bin").exists():
            print(f"Using local Faster-Whisper model: {local_dir}")
            return str(local_dir)
        return self.model_size

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

    def reduce_noise(self, audio_path: str) -> np.ndarray:
        from ai_workers.modules.common.denoise import get_denoised_audio_array
        return get_denoised_audio_array(audio_path)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        """Run ASR on audio file with word-level timestamps using Faster-Whisper.

        Returns:
            Dict with 'text', 'segments' (word-level timestamps) and 'language'.
        """
        model_source = self._resolve_model_source()
        if model_source == self.model_size:
            print(
                f"Loading Faster-Whisper model ({self.model_size}) on {self.device}... "
                "Lan dau se tai model tu HuggingFace (~150MB), co the mat 2-5 phut. "
                f"Neu treo lau, dat model vao: {self.cache_dir / f'faster-whisper-{self.model_size}'}"
            )
        else:
            print(f"Loading Faster-Whisper model from {model_source} on {self.device}...")
        model = WhisperModel(
            model_source,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.cache_dir),
        )

        print(f"Transcribing audio file: {audio_path} using Faster-Whisper with VAD...")
        audio_input = self.reduce_noise(audio_path)
        
        segments_gen, info = model.transcribe(
            audio_input,
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
            
            if not segment.words:
                if current_merged is None:
                    current_merged = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment_text,
                        "words": []
                    }
                else:
                    current_merged["end"] = segment.end
                    current_merged["text"] += " " + segment_text
                
                dur = current_merged["end"] - current_merged["start"]
                if dur > 5.0 and current_merged["text"].endswith(('.', '?', '!')):
                    segments.append(current_merged)
                    print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] (Punctuation) {current_merged['text']}")
                    current_merged = None
                continue

            for w in segment.words:
                w_text = w.word.strip()
                if not w_text:
                    continue
                    
                if current_merged is not None:
                    gap = w.start - current_merged["words"][-1]["end"]
                    dur = current_merged["end"] - current_merged["start"]
                    
                    if dur > 15.0 and gap > 1.0:
                        segments.append(current_merged)
                        print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] (Long Gap) {current_merged['text']}")
                        current_merged = None
                
                if current_merged is None:
                    current_merged = {
                        "start": w.start,
                        "end": w.end,
                        "text": "",
                        "words": []
                    }
                    
                current_merged["words"].append({
                    "word": w_text,
                    "start": w.start,
                    "end": w.end
                })
                current_merged["end"] = w.end
                
                if not current_merged["text"]:
                    current_merged["text"] = w_text
                else:
                    current_merged["text"] += " " + w_text
                    
                dur = current_merged["end"] - current_merged["start"]
                
                if dur > 5.0 and w_text.endswith(('.', '?', '!')):
                    segments.append(current_merged)
                    print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] (Punctuation) {current_merged['text']}")
                    current_merged = None
                elif dur > 20.0:
                    segments.append(current_merged)
                    print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] (Hard Cut) {current_merged['text']}")
                    current_merged = None

        if current_merged is not None and current_merged["text"]:
            segments.append(current_merged)
            print(f"[{current_merged['start']:.2f}s -> {current_merged['end']:.2f}s] (End) {current_merged['text']}")
            
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

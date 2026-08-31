"""Probe: can we ASR VISTA (1.93 TB) on T4 with Whisper-large-v3?

This script does NOT actually transcribe video. It only:
  1. Checks CUDA availability.
  2. Measures torch + faster-whisper import cost and peak VRAM during a 30s
     fake-audio inference.
  3. Reports VRAM peak and approximate real-time factor on a 30s synthetic
     audio file generated locally.
  4. Skips gracefully if no GPU.

Goal: answer "could we even run Whisper-large on T4 within 6 months?"
"""
from __future__ import annotations

import json
import sys
import time
import tempfile
import wave
import struct
import math
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = OUT_DIR / "whisper_t4_summary.json"


def make_sine_wav(path: Path, duration_sec: int = 30, sr: int = 16000) -> None:
    n = duration_sec * sr
    freq = 440
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            sample = int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframesraw(struct.pack("<h", sample))


def main() -> int:
    started = time.perf_counter()
    summary: dict = {"available": False}

    try:
        import torch
    except ImportError:
        summary["error"] = "torch not installed"
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        return 0

    summary["torch_version"] = torch.__version__
    summary["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        summary["error"] = "no CUDA; cannot measure T4"
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
        print("no CUDA available; skipping", flush=True)
        return 0
    summary["device_name"] = torch.cuda.get_device_name(0)
    summary["device_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    # Synthesize 30s audio
    print("[1/3] Synthesizing 30s audio ...", flush=True)
    audio_path = Path(tempfile.gettempdir()) / "probe_whisper_sine.wav"
    make_sine_wav(audio_path, duration_sec=30, sr=16000)
    summary["audio_path"] = str(audio_path)
    summary["audio_duration_sec"] = 30

    # Try faster-whisper first (CPU-friendly + VRAM-light)
    print("[2/3] Probing faster-whisper-large-v3 (if available) ...", flush=True)
    try:
        from faster_whisper import WhisperModel
        model_size = "large-v3"
        # Try int8 on T4 first (FP16 large-v3 = ~3GB+)
        compute_type = "int8"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        load_time = time.perf_counter() - load_start
        vram_after_load_gb = round(torch.cuda.memory_allocated() / 1e9, 3)

        # Transcribe
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=False)
        text_chunks = []
        for seg in segments:
            text_chunks.append(seg.text)
        full_text = "".join(text_chunks)
        tx_time = time.perf_counter() - t0
        vram_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        audio_real_time = info.duration if hasattr(info, "duration") else 30
        rtfx = round(tx_time / max(audio_real_time, 1), 4)
        summary["faster_whisper"] = {
            "model": model_size,
            "compute_type": compute_type,
            "load_time_sec": round(load_time, 2),
            "vram_after_load_gb": vram_after_load_gb,
            "vram_peak_gb": vram_peak_gb,
            "transcribe_time_sec": round(tx_time, 2),
            "audio_duration_sec": audio_real_time,
            "real_time_factor": rtfx,
            "transcript_chars": len(full_text),
        }
        print(f"  faster-whisper large-v3 int8 on T4: VRAM peak {vram_peak_gb} GB, RTFx {rtfx}", flush=True)
        del model
        torch.cuda.empty_cache()
    except ImportError:
        summary["faster_whisper"] = {"error": "faster_whisper not installed"}
        print("  faster_whisper not installed; skipping", flush=True)
    except Exception as exc:
        summary["faster_whisper"] = {"error": repr(exc)[:500]}
        print(f"  faster-whisper FAILED: {exc!r}", flush=True)

    # Try openai-whisper if available
    print("[3/3] Probing openai-whisper large-v3 (if available) ...", flush=True)
    try:
        import whisper
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_start = time.perf_counter()
        model = whisper.load_model("large-v3", device="cuda")
        load_time = time.perf_counter() - load_start
        vram_after_load_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
        t0 = time.perf_counter()
        result = model.transcribe(str(audio_path), fp16=True, beam_size=5, without_timestamps=True)
        tx_time = time.perf_counter() - t0
        vram_peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        rtfx = round(tx_time / 30, 4)
        summary["openai_whisper"] = {
            "model": "large-v3",
            "load_time_sec": round(load_time, 2),
            "vram_after_load_gb": vram_after_load_gb,
            "vram_peak_gb": vram_peak_gb,
            "transcribe_time_sec": round(tx_time, 2),
            "real_time_factor": rtfx,
            "transcript_chars": len(result.get("text", "")),
        }
        print(f"  openai-whisper large-v3 FP16 on T4: VRAM peak {vram_peak_gb} GB, RTFx {rtfx}", flush=True)
        del model
        torch.cuda.empty_cache()
    except ImportError:
        summary["openai_whisper"] = {"error": "openai-whisper not installed"}
    except Exception as exc:
        summary["openai_whisper"] = {"error": repr(exc)[:500]}

    summary["elapsed_sec"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  wrote {SUMMARY_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

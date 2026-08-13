"""Rewrite test_optimize_model.ipynb to align with ai_workers backend and ablate weak points."""
from __future__ import annotations

import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}


def md(source: str):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source: str):
    nb["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }
    )


md(
    """# Backend-Aligned Multimodal Optimization Benchmark

Notebook này **khớp pipeline production** trong `ai_workers/tasks.py` và dùng để:
1. Benchmark / so sánh cấu hình từng stage (`audio_v2` → `visual_v2` → `fusion`)
2. Đo các điểm yếu đã biết (scene threshold, CLIP slide filter, Florence/OCR fallback, ASR size)
3. Đề xuất config tối ưu cho lecture video

### Production stage order
1. `AudioTranscriber` (FFmpeg → DeepFilterNet denoise → Faster-Whisper)
2. `SpeakerDiarizer` (stub hiện tại)
3. `SceneDetector` (PySceneDetect + keyframes)
4. `SemanticAnalyzer` (CLIP → Florence-2 → PaddleOCR)
5. `TimelineBuilder`
6. `Summarizer` (+ RAG)

> Không còn là notebook Kaggle ASR-only. ASR WER sweep vẫn giữ như Stage A."""
)

md("## 0. Setup — import `ai_workers` + local model cache")

code(
    r'''import json
import os
import sys
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("../..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HOME"] = str(PROJECT_ROOT / "cache" / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(PROJECT_ROOT / "cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(PROJECT_ROOT / "cache" / "torch_hub")
os.chdir(PROJECT_ROOT)

from ai_workers.core.config import worker_settings
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber
from ai_workers.modules.audio_v2.speaker import SpeakerDiarizer
from ai_workers.modules.visual_v2.scene_detector import SceneDetector
from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
from ai_workers.modules.fusion.timeline import TimelineBuilder
from ai_workers.modules.fusion.summarizer import Summarizer

print(f"Project Root: {PROJECT_ROOT}")
print(f"CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"CACHE_DIR: {worker_settings.CACHE_DIR}")
print(f"SCENE_THRESHOLD(default): {worker_settings.SCENE_THRESHOLD}")
print(f"WHISPERX_MODEL(config): {worker_settings.WHISPERX_MODEL} | AudioTranscriber default still uses model_name=base.en unless overridden")
'''
)

md("## 1. Input — lecture video (demo) + optional TED-LIUM ASR regression set")

code(
    r'''JOB_ID = f"opt_{uuid.uuid4().hex[:8]}"
VIDEO_PATH = PROJECT_ROOT / "experiments" / "notebooks" / "demo_data" / "sample.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / JOB_ID
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Job: {JOB_ID}")
print(f"Video: {VIDEO_PATH} | exists={VIDEO_PATH.exists()}")
print(f"Output: {OUTPUT_DIR}")

# Optional: stream a small TED-LIUM set for ASR WER (skip if offline/no datasets)
USE_TEDLIUM_WER = False  # set True when network + datasets are available
TED_SAMPLES = []
if USE_TEDLIUM_WER:
    from datasets import load_dataset
    ds = load_dataset("distil-whisper/tedlium", "release1", split="validation", streaming=True)
    for row in ds:
        if row.get("text") and not row.get("ignore_time_segment_in_scoring", False):
            TED_SAMPLES.append(row)
        if len(TED_SAMPLES) >= 20:
            break
    print(f"TED-LIUM samples loaded: {len(TED_SAMPLES)}")
else:
    print("TED-LIUM WER disabled — Stage A will report RTF/confidence on local video only.")
'''
)

md(
    """## Stage A — ASR size / denoise ablation (`audio_v2.AudioTranscriber`)

So sánh `base.en` (default production) vs các size lớn hơn. Đo:
- RTF, load time, peak VRAM (trong lúc transcribe)
- avg ASR confidence proxy
- optional WER trên TED-LIUM nếu bật `USE_TEDLIUM_WER`
"""
)

code(
    r'''import gc
import re
from contextlib import contextmanager

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@contextmanager
def track_peak_vram():
    if not torch.cuda.is_available():
        yield lambda: 0.0
        return
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    yield lambda: torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.synchronize()


def avg_confidence(segments: list) -> float:
    scores = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            scores.append(sum(w.get("probability", 0.85) for w in words) / len(words))
        else:
            scores.append(0.85)
    return float(np.mean(scores)) if scores else 0.0


def benchmark_asr(model_name: str, video_path: Path, enable_denoise: bool = True) -> dict:
    t0 = time.perf_counter()
    with track_peak_vram() as peak_fn:
        tr = AudioTranscriber({"model_name": model_name})
        if not enable_denoise:
            # Bypass DeepFilterNet; still return raw waveform array for Faster-Whisper
            import soundfile as sf
            import librosa

            def _raw(audio_path: str):
                data, sr = sf.read(audio_path)
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                return data.astype(np.float32)

            tr.reduce_noise = _raw  # type: ignore

        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        result = tr.process(str(video_path))
        infer_s = time.perf_counter() - t1
        peak = peak_fn()

    segs = result.get("segments") or []
    duration = float(segs[-1]["end"]) if segs else 0.0
    row = {
        "model": model_name,
        "denoise": enable_denoise,
        "load_s": round(load_s, 3),
        "infer_s": round(infer_s, 3),
        "duration_s": round(duration, 3),
        "rtf": round(infer_s / max(duration, 1e-6), 4),
        "peak_vram_mb": round(peak, 1),
        "n_segments": len(segs),
        "avg_conf": round(avg_confidence(segs), 3),
        "text_len": len(result.get("text") or ""),
        "language": result.get("language"),
    }

    del tr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, result


ASR_MODELS = ["base.en"]
# Uncomment to sweep larger models (slower / needs local cache):
# ASR_MODELS = ["base.en", "small.en", "medium.en", "large-v3"]

asr_rows = []
best_audio = None
for m in ASR_MODELS:
    print(f"\n=== ASR bench: {m} ===")
    row, result = benchmark_asr(m, VIDEO_PATH, enable_denoise=True)
    asr_rows.append(row)
    best_audio = result
    print(row)

asr_df = pd.DataFrame(asr_rows)
print("\n=== ASR SUMMARY ===")
print(asr_df.to_string(index=False))
'''
)

md(
    """## Stage B — Scene detection threshold sweep (`visual_v2.SceneDetector`)

Điểm yếu: threshold `27` + slide cùng layout → 0 scene. Production đã có retry; cell này đo số scene theo threshold."""
)

code(
    r'''from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

THRESHOLDS = [27.0, 15.0, 10.0, 5.0]
scene_rows = []
for th in THRESHOLDS:
    video = open_video(str(VIDEO_PATH))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=th))
    t0 = time.perf_counter()
    sm.detect_scenes(video, frame_skip=4)
    dt = time.perf_counter() - t0
    n = len(sm.get_scene_list())
    scene_rows.append({"threshold": th, "n_scenes": n, "detect_s": round(dt, 3)})
    print(f"threshold={th}: scenes={n}, time={dt:.3f}s")

scene_df = pd.DataFrame(scene_rows)
print(scene_df.to_string(index=False))

# Run production SceneDetector (with retry + fallback)
print("\n=== Production SceneDetector.process ===")
detector = SceneDetector({"threshold": 27.0})
visual_result = detector.process(str(VIDEO_PATH), str(OUTPUT_DIR))
print(f"Production scenes kept: {len(visual_result.get('scenes', []))}")
for sc in visual_result.get("scenes", [])[:5]:
    print(f"  scene {sc['scene_index']}: {sc['start_timecode']} -> {sc['end_timecode']} | {sc.get('keyframe_path')}")
'''
)

md(
    """## Stage C — Semantic ablation (`visual_v2.SemanticAnalyzer`)

Đo CLIP keep-ratio, Florence/OCR caption source, latency. Production đã chuyển sang **lecture-aware CLIP prompts** + **OCR caption fallback**."""
)

code(
    r'''print("=== SemanticAnalyzer.process ===")
t0 = time.perf_counter()
semantic = SemanticAnalyzer({"max_keyframes": 15, "similarity_threshold": 0.82})
slides = semantic.process(list(visual_result.get("scenes", [])))
sem_s = time.perf_counter() - t0

cap_sources = {}
for s in slides:
    src = s.get("caption_source", "unknown")
    cap_sources[src] = cap_sources.get(src, 0) + 1

semantic_row = {
    "input_scenes": len(visual_result.get("scenes", [])),
    "kept_slides": len(slides),
    "semantic_s": round(sem_s, 3),
    "caption_sources": cap_sources,
    "ocr_nonempty": sum(1 for s in slides if (s.get("ocr_text") or "").strip()),
}
print(semantic_row)
for s in slides[:5]:
    print(
        f"  [{s.get('start_timecode')}] src={s.get('caption_source')} "
        f"cap={(s.get('caption') or '')[:70]!r} "
        f"ocr={(s.get('ocr_text') or '')[:50]!r}"
    )
'''
)

md("## Stage D — Quality gates (vấn đề 5 / 7 / 8) trên output pipeline")

code(
    r'''import cv2
from difflib import SequenceMatcher

# --- Vấn đề 5: ASR quality gate ---

def validate_audio_asr(audio_res: dict, expected_lang: str | None = None, min_conf: float = 0.6) -> dict:
    segs = audio_res.get("segments") or []
    text = (audio_res.get("text") or "").strip()
    report = {
        "status": "HIGH",
        "avg_conf": 1.0,
        "low_conf": 0,
        "n_segments": len(segs),
        "visual_dominant": False,
        "warnings": [],
    }
    if not segs or not text:
        report.update(status="FAILED", visual_dominant=True)
        report["warnings"].append("Empty ASR transcript")
        return report
    confs = []
    for seg in segs:
        words = seg.get("words") or []
        c = (sum(w.get("probability", 0.85) for w in words) / len(words)) if words else 0.85
        confs.append(c)
        if c < min_conf:
            report["low_conf"] += 1
    report["avg_conf"] = float(np.mean(confs))
    low_ratio = report["low_conf"] / max(1, len(segs))
    if expected_lang and audio_res.get("language") != expected_lang:
        report["warnings"].append(f"lang mismatch: {audio_res.get('language')} != {expected_lang}")
    if low_ratio > 0.4 or report["avg_conf"] < 0.65:
        report["status"] = "LOW_QUALITY_WARNING"
        report["visual_dominant"] = True
    elif low_ratio > 0.15:
        report["status"] = "MEDIUM"
    return report


# --- Vấn đề 7: blur + OCR-aware dedup ---

def smart_visual_gate(scenes: list, min_blur: float = 30.0, cos_th: float = 0.88, ocr_diff: float = 0.30) -> dict:
    kept = []
    blur_n = ocr_saved = dup_n = 0
    for sc in scenes:
        path = sc.get("keyframe_path")
        blur = 100.0
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur = float(cv2.Laplacian(img, cv2.CV_64F).var())
        sc["blur_score"] = round(blur, 2)
        ocr = (sc.get("ocr_text") or "").strip()
        if blur < min_blur and len(ocr) < 8:
            blur_n += 1
            continue
        emb = sc.get("embedding")
        is_dup = False
        for k in kept:
            k_emb = k.get("embedding")
            vis = 0.0
            if emb is not None and k_emb is not None:
                vis = float(np.dot(emb, k_emb) / (np.linalg.norm(emb) * np.linalg.norm(k_emb) + 1e-8))
            ocr_sim = SequenceMatcher(None, ocr.lower(), (k.get("ocr_text") or "").lower()).ratio()
            if vis > cos_th:
                if ocr and (k.get("ocr_text") or "").strip() and ocr_sim < (1 - ocr_diff):
                    ocr_saved += 1
                    continue
                is_dup = True
                dup_n += 1
                break
        if not is_dup:
            kept.append(sc)
    return {"input": len(scenes), "kept": len(kept), "blur_dropped": blur_n, "dup_dropped": dup_n, "ocr_saved": ocr_saved, "scenes": kept}


# --- Vấn đề 8: caption grounding ---

def verify_captions(scenes: list, min_ground: float = 0.22) -> dict:
    hallu = 0
    for sc in scenes:
        cap = (sc.get("caption") or "").strip()
        ocr = (sc.get("ocr_text") or "").strip()
        cap_words = set(re.findall(r"\w+", cap.lower())) - {"a", "an", "the", "of", "and", "slide", "text", "image"}
        ocr_words = set(re.findall(r"\w+", ocr.lower()))
        score = len(cap_words & ocr_words) / max(1, len(cap_words))
        sc["grounding_score"] = round(score, 2)
        if cap_words and score < min_ground and sc.get("caption_source") != "ocr_fallback":
            hallu += 1
            if ocr:
                sc["verified_caption"] = f"Slide text: {ocr[:120]}"
            else:
                sc["verified_caption"] = cap
        else:
            sc["verified_caption"] = cap
    return {"n": len(scenes), "hallucination_suspects": hallu}


audio_for_gate = best_audio or {"segments": [], "text": "", "language": None}
gate5 = validate_audio_asr(audio_for_gate, expected_lang=None)
gate7 = smart_visual_gate(slides)
gate8 = verify_captions(gate7["scenes"])

print("=== GATE 5 (ASR) ===", gate5)
print("=== GATE 7 (Visual) ===", {k: v for k, v in gate7.items() if k != "scenes"})
print("=== GATE 8 (Caption) ===", gate8)
'''
)

md("## Stage E — Timeline + Summarizer smoke (fusion)")

code(
    r'''print("=== Speaker (stub) ===")
diarizer = SpeakerDiarizer()
wav_path = str(VIDEO_PATH).rsplit(".", 1)[0] + ".wav"
utterances = diarizer.process(wav_path, (best_audio or {}).get("segments", []))
print(f"utterances={len(utterances)}")

print("=== Timeline ===")
timeline = TimelineBuilder().process(utterances, gate7["scenes"], gate7["scenes"])
chapters = timeline.get("chapters") or []
print(f"aligned={len(timeline.get('aligned_segments') or [])} chapters={len(chapters)}")

print("=== Summarizer ===")
try:
    text_result = Summarizer().process(utterances, gate7["scenes"], chapters)
    print("title:", text_result.get("video_title"))
    print("model:", text_result.get("model_used"))
    print("summary:", (text_result.get("summary") or "")[:400])
except Exception as e:
    print(f"Summarizer failed (API key/network?): {e}")
    text_result = {}
'''
)

md(
    """## Summary — khuyến nghị tối ưu

| Stage | Điểm yếu cũ | Hướng cải tiến đã làm / đề xuất |
|------|-------------|-------------------------------|
| ASR | `WHISPERX_MODEL=large-v3` không wired; default `base.en` | Benchmark Stage A rồi wire `model_name` từ config nếu cần chất lượng |
| Scene | threshold 27 miss slide | Retry 27→15→10→5 + fallback 1 scene |
| CLIP | filter coi **slide = junk** | Lecture-aware prompts: giữ slide, bỏ logo/black |
| Florence | crash / env pin chặt | Guard pixel_values + **OCR caption fallback** |
| Gates 5/7/8 | chỉ có ở notebook | Đo ở Stage D; cân nhắc port vào `tasks.py` |

Chạy lại notebook sau khi đổi code production để xác nhận số liệu Stage B/C cải thiện.
"""
)

code(
    r'''summary = {
    "job_id": JOB_ID,
    "video": str(VIDEO_PATH),
    "asr": asr_df.to_dict(orient="records"),
    "scene_threshold_sweep": scene_df.to_dict(orient="records"),
    "production_scenes": len(visual_result.get("scenes", [])),
    "semantic": semantic_row,
    "gate5_status": gate5.get("status"),
    "gate7_kept": gate7.get("kept"),
    "gate8_hallu": gate8.get("hallucination_suspects"),
    "chapters": len(chapters),
    "summary_title": text_result.get("video_title"),
}
out = OUTPUT_DIR / "optimize_report.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out}")
print(json.dumps(summary, ensure_ascii=False, indent=2))
'''
)

out_path = Path(r"C:\Users\admin\multimodal-lecture-summarizer\experiments\notebooks\test_optimize_model.ipynb")
out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {out_path} with {len(nb['cells'])} cells")

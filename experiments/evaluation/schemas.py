"""Ground-truth annotation schemas for pipeline evaluation tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def reference_dir(root: Path, video_id: str) -> Path:
    return root / "references" / video_id


def expected_files(video_id: str) -> dict[str, str]:
    """Relative paths under benchmarks/references/<video_id>/."""
    base = video_id
    return {
        "transcript": f"{base}/transcript.txt",
        "speech_segments": f"{base}/speech_segments.json",
        "scenes": f"{base}/scenes.json",
        "keyframes": f"{base}/keyframes.json",
        "ocr": f"{base}/ocr.json",
        "captions": f"{base}/captions.json",
        "alignments": f"{base}/alignments.json",
        "chapters": f"{base}/chapters.json",
        "summary": f"{base}/summary.md",
        "qa": f"{base}/qa.json",
    }


# Example shapes (documented for annotators):
#
# speech_segments.json:
#   [{"start": 1.2, "end": 4.5}, ...]
#
# scenes.json:
#   [{"start": 0.0, "end": 12.5}, ...]   # slide-change intervals OR
#   {"boundaries": [12.5, 40.1, ...]}    # cut times
#
# keyframes.json:
#   {"must_keep": ["slide_01", "slide_03"], "optional": ["slide_02"]}
#   OR [{"t": 12.5, "label": "must_keep"}, ...]
#
# ocr.json:
#   [{"image_id": "slide_01", "text": "Chapter 1 Introduction"}, ...]
#
# captions.json:
#   [{"keyframe_id": "kf_01", "content_ok": true, "hallucination": false, "score_1_5": 4}, ...]
#
# alignments.json:
#   [{"utterance_id": "u1", "start": 10.0, "end": 14.0, "slide_id": "slide_02"}, ...]
#
# chapters.json:
#   [{"title": "Intro", "start": 0.0, "end": 45.0}, ...]
#   OR {"boundaries": [45.0, 120.0], "titles": ["Intro", "Methods"]}
#
# qa.json:
#   [{
#     "question": "...",
#     "answer": "...",
#     "timestamp": 83.5,
#     "gold_chunk_ids": ["ch2_seg3"]
#   }, ...]


def normalize_intervals(payload: Any) -> list[tuple[float, float]]:
    if payload is None:
        return []
    if isinstance(payload, dict) and "segments" in payload:
        payload = payload["segments"]
    if isinstance(payload, dict) and "boundaries" in payload:
        bounds = sorted(float(x) for x in payload["boundaries"])
        # convert cut list to intervals only if duration provided
        duration = float(payload.get("duration", 0.0) or 0.0)
        if duration <= 0 and bounds:
            duration = bounds[-1]
        starts = [0.0] + bounds
        ends = bounds + ([duration] if duration > 0 else [bounds[-1]])
        return [(s, e) for s, e in zip(starts, ends) if e > s]
    out: list[tuple[float, float]] = []
    for item in payload or []:
        if isinstance(item, dict):
            s = float(item.get("start", item.get("start_seconds", 0.0)))
            e = float(item.get("end", item.get("end_seconds", s)))
            if e > s:
                out.append((s, e))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            s, e = float(item[0]), float(item[1])
            if e > s:
                out.append((s, e))
    return out


def normalize_boundaries(payload: Any) -> list[float]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "boundaries" in payload:
            return sorted(float(x) for x in payload["boundaries"])
        if "chapters" in payload:
            payload = payload["chapters"]
        elif "segments" in payload:
            payload = payload["segments"]
    bounds: list[float] = []
    for item in payload or []:
        if isinstance(item, (int, float)):
            bounds.append(float(item))
        elif isinstance(item, dict):
            raw = item.get("start", item.get("start_seconds", item.get("startTime")))
            if raw is not None and float(raw) > 0:
                bounds.append(float(raw))
    return sorted(set(bounds))

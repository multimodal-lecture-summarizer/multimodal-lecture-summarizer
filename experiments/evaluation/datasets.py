"""Resolve TED-LIUM / TVSum paths and build stage-appropriate evaluation items.

Mapping (TTTN/DATN):
  ASR, VAD              -> TED-LIUM (wav + official transcripts / speech intervals)
  Scene, Keyframe       -> TVSum (videos + frame importance GT)
  OCR, Caption,
  Timeline, Chapter     -> TED lecture videos (closest to the lecture pipeline)
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]

# Single evaluation label for all TED-LIUM audio + TED talk video stages (TTTN/DATN).
TED_DATASET = "TED"

TED_ROOT = Path(r"D:\datasets\TEDLIUM")
TED_AUDIO = TED_ROOT / "audio"
TED_VIDEO = TED_ROOT / "videos"
TED_META = TED_ROOT / "metadata.csv"

TVSUM_ROOT = Path(r"D:\datasets\tvsum")
TVSUM_ANNO = TVSUM_ROOT / "data" / "ydata-tvsum50-anno.tsv"
TVSUM_INFO = TVSUM_ROOT / "data" / "ydata-tvsum50-info.tsv"
TVSUM_ZIP = TVSUM_ROOT / "ydata-tvsum50-v1_1" / "ydata-tvsum50-video.zip"
TVSUM_EXTRACT = ROOT / "experiments" / "datasets" / "tvsum_extracted" / "tvsum_videos" / "video"

IGNORE_TEXT = "ignore_time_segment_in_scoring"
_TS_RE = re.compile(r"-(\d+\.\d+)-(\d+\.\d+)-")


def tedlium_available() -> bool:
    return TED_META.exists() and TED_AUDIO.exists()


def tvsum_available() -> bool:
    return TVSUM_ANNO.exists() and (TVSUM_ZIP.exists() or TVSUM_EXTRACT.exists())


def load_tedlium_rows() -> list[dict[str, Any]]:
    if not TED_META.exists():
        return []
    rows: list[dict[str, Any]] = []
    with TED_META.open("r", encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            text = (rec.get("text") or "").strip()
            wav = Path(rec.get("wav_path") or "")
            if text == IGNORE_TEXT or not wav.exists():
                continue
            start, end = _parse_times(rec.get("id") or wav.name)
            rows.append(
                {
                    "id": rec.get("id") or wav.stem,
                    "speaker_id": rec.get("speaker_id") or "",
                    "duration": float(rec.get("duration") or 0.0),
                    "wav_path": wav,
                    "text": text.replace(" '", "'"),
                    "start": start,
                    "end": end,
                }
            )
    return rows


def _parse_times(name: str) -> tuple[float | None, float | None]:
    m = _TS_RE.search(name)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def pick_tedlium_asr_items(*, limit: int = 6) -> list[dict[str, Any]]:
    """Diverse, short TED-LIUM clips with official transcripts."""
    rows = [r for r in load_tedlium_rows() if 4.0 <= r["duration"] <= 16.0 and len(r["text"].split()) >= 8]
    by_spk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda x: x["duration"]):
        by_spk[r["speaker_id"]].append(r)
    picked: list[dict[str, Any]] = []
    speakers = list(by_spk.keys())
    idx = 0
    while len(picked) < limit and speakers:
        spk = speakers[idx % len(speakers)]
        bucket = by_spk[spk]
        if bucket:
            picked.append(bucket.pop(0))
        if not bucket:
            speakers.remove(spk)
            if not speakers:
                break
            idx = idx % len(speakers)
            continue
        idx += 1
    return picked


def ted_video_for_speaker(speaker_id: str) -> Path | None:
    if not TED_VIDEO.exists():
        return None
    candidates = [
        TED_VIDEO / f"{speaker_id}.mp4",
        TED_VIDEO / f"{speaker_id}.wav",
    ]
    for p in candidates:
        if p.exists():
            return p
    low = speaker_id.lower().replace("_", " ")
    for p in TED_VIDEO.glob("*.mp4"):
        if speaker_id.lower() in p.stem.lower() or low in p.stem.lower():
            return p
    return None


def ted_full_audio_for_speaker(speaker_id: str) -> Path | None:
    wav = TED_VIDEO / f"{speaker_id}.wav"
    if wav.exists():
        return wav
    mp4 = ted_video_for_speaker(speaker_id)
    return mp4


def tedlium_speech_intervals(speaker_id: str) -> list[tuple[float, float]]:
    iv: list[tuple[float, float]] = []
    for r in load_tedlium_rows():
        if r["speaker_id"] != speaker_id:
            continue
        if r["start"] is None or r["end"] is None:
            continue
        if r["end"] > r["start"]:
            iv.append((float(r["start"]), float(r["end"])))
    iv.sort()
    return iv


def pick_ted_vad_item() -> dict[str, Any] | None:
    items = pick_ted_vad_items(limit=1)
    return items[0] if items else None


def pick_ted_vad_items(*, limit: int = 2) -> list[dict[str, Any]]:
    """TED talks that have both full audio and dense speech GT."""
    counts: dict[str, int] = defaultdict(int)
    for r in load_tedlium_rows():
        counts[r["speaker_id"]] += 1
    out: list[dict[str, Any]] = []
    for speaker, _n in sorted(counts.items(), key=lambda kv: -kv[1]):
        media = ted_full_audio_for_speaker(speaker)
        intervals = tedlium_speech_intervals(speaker)
        if media and len(intervals) >= 5:
            out.append(
                {
                    "video_id": speaker,
                    "media_path": media,
                    "intervals": intervals,
                    "dataset": "TED-LIUM",
                }
            )
        if len(out) >= limit:
            break
    return out


def pick_ted_unified_talks(
    *,
    limit: int = 2,
    asr_clips_per_talk: int = 3,
) -> list[dict[str, Any]]:
    """One eval unit per TED talk: video + utterances + short ASR clips (same dataset label)."""
    videos = pick_ted_lecture_videos(limit=limit)
    rows = load_tedlium_rows()
    talks: list[dict[str, Any]] = []
    for vp in videos:
        speaker = vp.stem
        utts = [
            {"start": r["start"], "end": r["end"], "text": r["text"]}
            for r in rows
            if r["speaker_id"] == speaker and r["start"] is not None and r["end"] is not None
        ]
        if len(utts) < 3:
            utts = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in rows
                if r["speaker_id"].lower() in speaker.lower() and r["start"] is not None
            ]
        clips = [
            r
            for r in load_tedlium_rows()
            if r["speaker_id"] == speaker and 4.0 <= r["duration"] <= 16.0 and len(r["text"].split()) >= 8
        ]
        clips.sort(key=lambda x: x["duration"])
        talks.append(
            {
                "dataset": TED_DATASET,
                "talk_id": speaker,
                "video_path": vp,
                "utterances": utts,
                "asr_clips": clips[: max(1, asr_clips_per_talk)],
            }
        )
    return talks


def pick_ted_lecture_videos(*, limit: int = 2) -> list[Path]:
    """Shorter TED mp4s first — lecture-like talks for visual/timeline stages."""
    if not TED_VIDEO.exists():
        return []
    preferred = ["Blaise_Agueray_Arcas.mp4", "Barry_Schwartz.mp4", "S76.mp4", "S44.mp4"]
    out: list[Path] = []
    for name in preferred:
        p = TED_VIDEO / name
        if p.exists():
            out.append(p)
        if len(out) >= limit:
            return out
    for p in sorted(TED_VIDEO.glob("*.mp4"), key=lambda x: x.stat().st_size):
        if p not in out:
            out.append(p)
        if len(out) >= limit:
            break
    return out


def load_tvsum_info() -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    if not TVSUM_INFO.exists():
        return info
    with TVSUM_INFO.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for rec in reader:
            vid = (rec.get("video_id") or "").strip()
            if vid:
                info[vid] = rec
    return info


def load_tvsum_scores(video_id: str) -> list[float]:
    """Mean frame-level importance (1–5) across annotators."""
    if not TVSUM_ANNO.exists():
        return []
    rows: list[list[float]] = []
    with TVSUM_ANNO.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] != video_id:
                continue
            scores = [float(x) for x in parts[2].split(",") if x]
            if scores:
                rows.append(scores)
    if not rows:
        return []
    n = min(len(r) for r in rows)
    return [sum(r[i] for r in rows) / len(rows) for i in range(n)]


def parse_tvsum_length_sec(text: str | None) -> float:
    if not text:
        return 0.0
    parts = [int(x) for x in text.strip().split(":") if x.isdigit() or x]
    try:
        bits = [int(x) for x in text.strip().split(":")]
    except Exception:
        return 0.0
    if len(bits) == 2:
        return bits[0] * 60 + bits[1]
    if len(bits) == 3:
        return bits[0] * 3600 + bits[1] * 60 + bits[2]
    return 0.0


def ensure_tvsum_videos(video_ids: Iterable[str]) -> dict[str, Path]:
    TVSUM_EXTRACT.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    wanted = list(video_ids)
    for vid in wanted:
        local = TVSUM_EXTRACT / f"{vid}.mp4"
        if local.exists():
            found[vid] = local
    missing = [v for v in wanted if v not in found]
    if missing and TVSUM_ZIP.exists():
        with ZipFile(TVSUM_ZIP) as zf:
            names = {Path(n).name: n for n in zf.namelist() if n.lower().endswith(".mp4")}
            for vid in missing:
                key = f"{vid}.mp4"
                if key not in names:
                    continue
                target = TVSUM_EXTRACT / key
                if not target.exists():
                    print(f"[TVSum] Extracting {key} ...")
                    with zf.open(names[key]) as src, target.open("wb") as dst:
                        dst.write(src.read())
                if target.exists():
                    found[vid] = target
    return found


def pick_tvsum_videos(*, limit: int = 2) -> list[dict[str, Any]]:
    """Prefer short TVSum videos that already exist or can be extracted."""
    info = load_tvsum_info()
    ranked = sorted(
        info.items(),
        key=lambda kv: parse_tvsum_length_sec(kv[1].get("length")),
    )
    # Prefer already-extracted, then shortest
    existing = []
    rest = []
    for vid, rec in ranked:
        p = TVSUM_EXTRACT / f"{vid}.mp4"
        item = {"video_id": vid, "info": rec, "length_sec": parse_tvsum_length_sec(rec.get("length"))}
        if p.exists():
            existing.append(item)
        else:
            rest.append(item)
    chosen = (existing + rest)[: max(limit, 1)]
    paths = ensure_tvsum_videos([c["video_id"] for c in chosen])
    out = []
    for c in chosen:
        p = paths.get(c["video_id"])
        if not p:
            continue
        scores = load_tvsum_scores(c["video_id"])
        if not scores:
            continue
        c["video_path"] = p
        c["scores"] = scores
        c["dataset"] = "TVSum"
        out.append(c)
        if len(out) >= limit:
            break
    return out


def tvsum_important_windows(
    scores: list[float],
    *,
    fps: float,
    threshold: float | None = None,
    min_dur: float = 1.0,
) -> list[tuple[float, float]]:
    if not scores or fps <= 0:
        return []
    thr = threshold if threshold is not None else (sum(scores) / len(scores) + 0.4)
    windows: list[tuple[float, float]] = []
    start = None
    for i, s in enumerate(scores):
        if s >= thr:
            if start is None:
                start = i
        elif start is not None:
            windows.append((start / fps, i / fps))
            start = None
    if start is not None:
        windows.append((start / fps, len(scores) / fps))
    return [(a, b) for a, b in windows if (b - a) >= min_dur]


def tvsum_scene_boundaries(
    scores: list[float],
    *,
    fps: float,
    drop: float = 0.25,
    min_gap_sec: float = 2.0,
) -> list[float]:
    """Treat sharp importance drops as pseudo scene cuts (TVSum has no shot labels)."""
    if len(scores) < 8 or fps <= 0:
        return []
    win = max(3, int(fps * 0.5))
    smooth = []
    for i in range(len(scores)):
        a = max(0, i - win)
        b = min(len(scores), i + win + 1)
        smooth.append(sum(scores[a:b]) / (b - a))
    # TVSum scores are piecewise-constant (copied across frames in a shot).
    # Consecutive-frame deltas after smoothing are tiny, so use plateau edges.
    bounds: list[float] = []
    last_t = -1e9
    last_level = smooth[0]
    for i in range(1, len(smooth)):
        if abs(smooth[i] - last_level) >= drop:
            t = i / fps
            if t - last_t >= min_gap_sec:
                bounds.append(t)
                last_t = t
                last_level = smooth[i]
    if not bounds:
        for a, _b in tvsum_important_windows(scores, fps=fps):
            if a >= min_gap_sec:
                bounds.append(a)
    return bounds

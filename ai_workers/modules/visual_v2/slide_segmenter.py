"""Slide-aware scene segmentation.

PySceneDetect's ContentDetector scores the whole frame, so a slide that changes inside a fixed
layout (speaker video next to the slide) barely moves the score and the cut is missed. This module
compares small grid blocks only in the *static* part of the frame:

    sequential decode -> 2 samples/s -> gray 160x90 -> 16x9 block grid
    -> drop blocks that change all the time (speaker region)
    -> fraction of static blocks that changed vs k samples earlier
    -> boundary when it exceeds ``min_fraction`` and the new picture then stays put

Only OpenCV and NumPy are used. ``find_boundaries`` is pure (no video I/O) so it can be tested with
synthetic frames. ``SlideSegmenter.detect`` returns the same scene dicts as ``SceneDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

FRAME_W, FRAME_H = 160, 90
GRID_COLS, GRID_ROWS = 16, 9


@dataclass(frozen=True)
class SegmenterParams:
    sample_fps: float = 2.0
    # A block counts as changed when its mean absolute gray difference exceeds this (0-255 scale).
    block_threshold: float = 4.0
    # Fraction of static blocks that must change for a candidate boundary. Progressive bullet
    # reveals stay below it.
    min_fraction: float = 0.22
    # Samples the new picture must stay stable; also the look-back distance, so dissolves are
    # compared with a picture from before they started, and two changes closer than this merge.
    k: int = 3
    # A block that changes in more than this fraction of consecutive steps belongs to the speaker
    # (or any moving overlay) and is excluded.
    volatile_fraction: float = 0.1
    # Fewer static blocks than this fraction of the grid -> too little slide area to judge.
    min_static_fraction: float = 0.20
    # Consecutive-sample change below this fraction counts as "calm" while settling.
    settle_fraction: float = 0.03


@dataclass
class BoundaryResult:
    # Times (seconds) of the last sample that still shows the old slide.
    boundaries: list[float] = field(default_factory=list)
    static_fraction: float = 0.0
    usable: bool = True


def _block_means(frame: np.ndarray) -> np.ndarray:
    """(H, W) uint8/float -> (GRID_ROWS, GRID_COLS) block means."""
    h, w = frame.shape
    bh, bw = h // GRID_ROWS, w // GRID_COLS
    cropped = frame[: bh * GRID_ROWS, : bw * GRID_COLS].astype(np.float32)
    return cropped.reshape(GRID_ROWS, bh, GRID_COLS, bw).mean(axis=(1, 3))


def _changed_blocks(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean grid: blocks whose mean absolute pixel difference exceeds threshold."""
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return _block_means(diff) > threshold


def find_boundaries(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    params: SegmenterParams | None = None,
) -> BoundaryResult:
    """Find slide change times in a sequence of gray frames sampled at ``params.sample_fps``."""
    p = params or SegmenterParams()
    n = len(frames)
    if n != len(times):
        raise ValueError("frames and times must have the same length")
    if n < 2:
        return BoundaryResult(usable=n > 0)

    # Per-block volatility over consecutive samples.
    step_changed = np.stack(
        [_changed_blocks(frames[i - 1], frames[i], p.block_threshold) for i in range(1, n)]
    )
    volatility = step_changed.mean(axis=0)
    static_mask = volatility <= p.volatile_fraction
    static_count = int(static_mask.sum())
    static_fraction = static_count / float(GRID_ROWS * GRID_COLS)
    if static_fraction < p.min_static_fraction:
        return BoundaryResult(static_fraction=static_fraction, usable=False)

    def frac(i: int, j: int) -> float:
        changed = _changed_blocks(frames[i], frames[j], p.block_threshold)
        return float(changed[static_mask].sum()) / static_count

    k = max(1, p.k)
    boundaries: list[float] = []
    i = k
    while i < n:
        if frac(i, i - k) <= p.min_fraction:
            i += 1
            continue

        # Last sample that still matches the look-back sample: a hard cut ends at i-1, a dissolve
        # at the last sample before the blend drifts away from the old picture.
        base = i - k
        old = base
        while old + 1 < i and frac(old + 1, base) <= p.min_fraction / 4.0:
            old += 1

        # Settle: first index whose next k-1 steps are all calm. Near the end, use what is left.
        s = i
        while s < n - 1:
            window_end = min(n - 1, s + k - 1)
            if all(frac(j, j - 1) <= p.settle_fraction for j in range(s + 1, window_end + 1)):
                break
            s += 1

        if frac(s, old) > p.min_fraction:
            t = float(times[old])
            # Two changes closer than k samples (slide -> black -> slide) keep the first mark.
            if not boundaries or (t - boundaries[-1]) * p.sample_fps >= k:
                boundaries.append(t)
        i = s + k  # the new picture needs k samples before the next comparison is meaningful
    return BoundaryResult(boundaries=boundaries, static_fraction=static_fraction)


def _timecode(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def build_scenes(
    boundaries: Sequence[float], duration: float, fps: float
) -> list[dict[str, Any]]:
    """Turn boundary times into contiguous scene dicts using the SceneDetector contract."""
    cuts = [0.0] + [b for b in sorted(boundaries) if 0.0 < b < duration] + [duration]
    scenes: list[dict[str, Any]] = []
    for idx in range(len(cuts) - 1):
        start, end = cuts[idx], cuts[idx + 1]
        if end <= start:
            continue
        scenes.append(
            {
                "scene_index": len(scenes),
                "start_seconds": start,
                "end_seconds": end,
                "start_timecode": _timecode(start),
                "end_timecode": _timecode(end),
                "start_frame": int(round(start * fps)),
                "end_frame": int(round(end * fps)),
            }
        )
    return scenes


class SlideSegmentError(RuntimeError):
    """The slide segmenter cannot judge this video; the caller should fall back."""


class SlideSegmenter:
    def __init__(self, params: SegmenterParams | None = None):
        self.params = params or SegmenterParams()

    def read_samples(self, video_path: str) -> tuple[list[np.ndarray], list[float], float, float]:
        """Decode sequentially (no seeking) and keep ``sample_fps`` gray thumbnails."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise SlideSegmentError(f"cannot open video: {video_path}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                raise SlideSegmentError("video reports no frame rate")
            stride = max(1, int(round(fps / self.params.sample_fps)))
            frames: list[np.ndarray] = []
            times: list[float] = []
            index = 0
            while cap.grab():
                if index % stride == 0:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        frames.append(
                            cv2.resize(gray, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA)
                        )
                        times.append(index / fps)
                index += 1
            duration = index / fps
        finally:
            cap.release()
        # Sample spacing is the stride in seconds, which can differ slightly from 1/sample_fps.
        self._effective_fps = fps / stride
        return frames, times, duration, fps

    def detect(self, video_path: str, min_scenes_for_long_video: int = 2) -> list[dict[str, Any]]:
        frames, times, duration, fps = self.read_samples(video_path)
        if len(frames) < 2:
            raise SlideSegmentError("too few frames sampled")
        params = self.params
        if abs(self._effective_fps - params.sample_fps) > 1e-6:
            params = SegmenterParams(**{**params.__dict__, "sample_fps": self._effective_fps})
        result = find_boundaries(frames, times, params)
        if not result.usable:
            raise SlideSegmentError(
                f"static slide area too small ({result.static_fraction:.0%} of the frame)"
            )
        scenes = build_scenes(result.boundaries, duration, fps)
        if duration > 120.0 and len(scenes) < min_scenes_for_long_video:
            raise SlideSegmentError(
                f"only {len(scenes)} segment(s) in a {duration:.0f}s video"
            )
        return scenes

"""Slide segmenter: synthetic-frame tests for find_boundaries, scene contract, config wiring."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_workers.core.checkpoint import build_config_fingerprint
from ai_workers.modules.fusion.quality_postprocess import resolve_stack
from ai_workers.modules.visual_v2.slide_segmenter import (
    FRAME_H,
    FRAME_W,
    SegmenterParams,
    SlideSegmenter,
    build_scenes,
    find_boundaries,
)

FPS = 2.0
P = SegmenterParams(sample_fps=FPS)


def slide(seed: int, bullets: int = 99) -> np.ndarray:
    """A white slide with `bullets` dark text-like bars; each seed is a different layout."""
    rng = np.random.RandomState(seed)
    img = np.full((FRAME_H, FRAME_W), 235, np.uint8)
    for i in range(min(bullets, 8)):
        y = 8 + i * 10
        x0 = int(rng.randint(5, 30))
        x1 = int(rng.randint(70, 150))
        img[y : y + 5, x0:x1] = int(rng.randint(10, 60))
    return img


def sequence(parts: list[tuple[np.ndarray, int]]):
    frames, times = [], []
    for img, count in parts:
        for _ in range(count):
            frames.append(img.copy())
            times.append(len(times) / FPS)
    return frames, times


def with_speaker(frames, seed=0):
    """Paint a noisy moving box (the speaker) in the bottom-right corner of every frame."""
    rng = np.random.RandomState(seed)
    out = []
    for f in frames:
        g = f.copy()
        g[50:90, 110:160] = rng.randint(0, 255, (40, 50), dtype=np.uint8)
        out.append(g)
    return out


def test_hard_cut_marks_last_old_sample():
    frames, times = sequence([(slide(1), 10), (slide(2), 10)])
    result = find_boundaries(frames, times, P)
    assert result.boundaries == [pytest.approx(times[9])]


def test_unchanged_video_has_no_boundary():
    frames, times = sequence([(slide(1), 30)])
    assert find_boundaries(frames, times, P).boundaries == []


def test_multiple_cuts():
    frames, times = sequence([(slide(1), 8), (slide(2), 8), (slide(3), 8)])
    assert find_boundaries(frames, times, P).boundaries == [
        pytest.approx(times[7]),
        pytest.approx(times[15]),
    ]


def bullet_slide(count: int) -> np.ndarray:
    """One thin text line per bullet, each inside a single grid row (a realistic reveal step)."""
    img = np.full((FRAME_H, FRAME_W), 235, np.uint8)
    for i in range(count):
        y = 10 * i + 3
        img[y : y + 4, 10:70] = 30
    return img


def test_progressive_bullets_are_not_boundaries():
    parts = [(bullet_slide(n), 6) for n in range(1, 9)]
    frames, times = sequence(parts)
    assert find_boundaries(frames, times, P).boundaries == []


def test_speaker_motion_is_ignored_and_cut_still_found():
    frames, times = sequence([(slide(1), 10), (slide(2), 10)])
    result = find_boundaries(with_speaker(frames), times, P)
    assert result.boundaries == [pytest.approx(times[9])]
    assert result.static_fraction < 1.0


def test_speaker_motion_alone_has_no_boundary():
    frames, times = sequence([(slide(1), 30)])
    assert find_boundaries(with_speaker(frames), times, P).boundaries == []


def test_dissolve_gives_one_boundary_near_start_of_blend():
    a, b = slide(1), slide(2)
    blend = [
        cv2.addWeighted(a, 1 - w, b, w, 0) for w in (0.2, 0.4, 0.6, 0.8)
    ]
    # Long enough that the blend steps are a small share of all steps (not "volatile" blocks).
    parts = [(a, 40)] + [(f, 1) for f in blend] + [(b, 40)]
    frames, times = sequence(parts)
    result = find_boundaries(frames, times, P)
    assert len(result.boundaries) == 1
    assert times[39] <= result.boundaries[0] <= times[41]


def test_black_interstitial_merges_to_single_boundary():
    black = np.zeros((FRAME_H, FRAME_W), np.uint8)
    frames, times = sequence([(slide(1), 40), (black, 2), (slide(2), 40)])
    result = find_boundaries(frames, times, P)
    assert len(result.boundaries) == 1
    assert result.boundaries[0] == pytest.approx(times[39])


def test_mostly_moving_frame_is_unusable():
    rng = np.random.RandomState(3)
    frames = [rng.randint(0, 255, (FRAME_H, FRAME_W), dtype=np.uint8) for _ in range(20)]
    times = [i / FPS for i in range(20)]
    result = find_boundaries(frames, times, P)
    assert result.usable is False


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        find_boundaries([slide(1)], [0.0, 1.0], P)


def test_build_scenes_keeps_scene_contract_and_is_contiguous():
    scenes = build_scenes([4.5, 10.0], duration=20.0, fps=25.0)
    assert [s["scene_index"] for s in scenes] == [0, 1, 2]
    keys = {
        "scene_index", "start_seconds", "end_seconds", "start_timecode",
        "end_timecode", "start_frame", "end_frame",
    }
    assert all(set(s) == keys for s in scenes)
    assert scenes[0]["start_seconds"] == 0.0 and scenes[-1]["end_seconds"] == 20.0
    assert scenes[0]["end_seconds"] == scenes[1]["start_seconds"]
    assert scenes[1]["start_frame"] == 112 or scenes[1]["start_frame"] == 113
    assert scenes[2]["end_timecode"] == "00:00:20.000"


def _write_video(path, frames_per_slide=40, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    for seed in (1, 2, 3):
        img = cv2.resize(slide(seed), (320, 180), interpolation=cv2.INTER_NEAREST)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for _ in range(frames_per_slide):
            writer.write(bgr)
    writer.release()


def test_detect_on_real_video_file(tmp_path):
    video = tmp_path / "slides.mp4"
    _write_video(video)
    scenes = SlideSegmenter().detect(str(video))
    assert len(scenes) == 3
    for s, expected in zip(scenes[1:], (4.0, 8.0)):
        assert abs(s["start_seconds"] - expected) <= 0.5


def test_detect_raises_for_unreadable_video(tmp_path):
    from ai_workers.modules.visual_v2.slide_segmenter import SlideSegmentError

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(SlideSegmentError):
        SlideSegmenter().detect(str(bad))


def test_scene_detector_falls_back_to_content(tmp_path, monkeypatch):
    from ai_workers.modules.visual_v2 import scene_detector as sd

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    sentinel = [{"scene_index": 0}]

    def boom(self, *_a, **_k):
        raise RuntimeError("no")

    monkeypatch.setattr("ai_workers.modules.visual_v2.slide_segmenter.SlideSegmenter.detect", boom)
    monkeypatch.setattr(sd.SceneDetector, "_detect_content", lambda self, p: sentinel)
    det = sd.SceneDetector({"segmenter": "slide"})
    assert det.detect_scenes(str(video)) == sentinel
    assert det.segmenter_used == "content"


def test_explicit_threshold_means_content_even_if_env_says_slide(monkeypatch):
    from ai_workers.core.config import worker_settings
    from ai_workers.modules.visual_v2 import scene_detector as sd

    monkeypatch.setattr(worker_settings, "SCENE_SEGMENTER", "slide")
    assert sd.SceneDetector({"threshold": 20.0}).segmenter == "content"
    assert sd.SceneDetector().segmenter == "slide"


def test_content_segmenter_never_calls_slide_path(tmp_path, monkeypatch):
    from ai_workers.modules.visual_v2 import scene_detector as sd

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(sd.SceneDetector, "_detect_content", lambda self, p: [])

    def fail(self, *_a, **_k):
        raise AssertionError("slide path used")

    monkeypatch.setattr("ai_workers.modules.visual_v2.slide_segmenter.SlideSegmenter.detect", fail)
    det = sd.SceneDetector({"segmenter": "content"})
    assert det.detect_scenes(str(video)) == []


def test_fingerprint_unchanged_for_content_and_differs_for_slide():
    base = dict(asr_model="m", source_language="en", florence_device="cpu", sprint_stack="recommended")
    legacy = build_config_fingerprint(**base)
    assert build_config_fingerprint(**base, scene_segmenter="content") == legacy
    assert build_config_fingerprint(**base, scene_segmenter="slide") != legacy


def test_slides_stack_preset_skips_density_prune():
    assert "sprint4_v2" not in resolve_stack("slides")
    assert "sprint4_v2" in resolve_stack("recommended")

"""filter_slide_segments: slide-mode filter keeps every segment except near-duplicate neighbours."""

from __future__ import annotations

import numpy as np
import pytest

from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer


def make_scenes(tmp_path, spans):
    scenes = []
    for idx, (start, end) in enumerate(spans):
        img = tmp_path / f"kf{idx}.png"
        img.write_bytes(b"png")
        scenes.append(
            {
                "scene_index": idx,
                "start_seconds": float(start),
                "end_seconds": float(end),
                "end_timecode": f"t{end}",
                "end_frame": int(end * 10),
                "keyframe_path": str(img),
            }
        )
    return scenes


def analyzer_with(monkeypatch, embeddings):
    analyzer = SemanticAnalyzer()
    monkeypatch.setattr(analyzer, "_embed_keyframes", lambda scenes: np.array(embeddings[: len(scenes)]))
    return analyzer


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


def test_keeps_all_distinct_segments_beyond_old_keyframe_cap(tmp_path, monkeypatch):
    n = 30  # the content filter caps at ~15
    scenes = make_scenes(tmp_path, [(i * 10, i * 10 + 10) for i in range(n)])
    emb = [np.eye(n)[i] for i in range(n)]
    out = analyzer_with(monkeypatch, emb).filter_slide_segments(scenes, max_segments=80)
    assert len(out) == n
    assert all("importanceScore" in s for s in out)


def test_merges_near_duplicate_neighbour_and_extends_range(tmp_path, monkeypatch):
    scenes = make_scenes(tmp_path, [(0, 10), (10, 20), (20, 30)])
    emb = [unit(1, 0, 0), unit(1, 0.01, 0), unit(0, 1, 0)]
    out = analyzer_with(monkeypatch, emb).filter_slide_segments(scenes, max_segments=80)
    assert [s["scene_index"] for s in out] == [0, 2]
    assert out[0]["end_seconds"] == 20.0 and out[0]["end_frame"] == 200


def test_non_adjacent_repeat_is_kept(tmp_path, monkeypatch):
    scenes = make_scenes(tmp_path, [(0, 10), (10, 20), (20, 30)])
    emb = [unit(1, 0, 0), unit(0, 1, 0), unit(1, 0, 0)]
    out = analyzer_with(monkeypatch, emb).filter_slide_segments(scenes, max_segments=80)
    assert len(out) == 3


def test_cap_keeps_longest_in_time_order(tmp_path, monkeypatch):
    spans = [(0, 5), (5, 30), (30, 33), (33, 60), (60, 62)]
    scenes = make_scenes(tmp_path, spans)
    emb = [np.eye(5)[i] for i in range(5)]
    out = analyzer_with(monkeypatch, emb).filter_slide_segments(scenes, max_segments=2)
    assert [s["scene_index"] for s in out] == [1, 3]
    # No gaps: head segment joins the first kept, dropped middle/tail join the previous kept.
    assert (out[0]["start_seconds"], out[0]["end_seconds"]) == (0.0, 33.0)
    assert (out[1]["start_seconds"], out[1]["end_seconds"]) == (33.0, 62.0)


def test_missing_keyframes_are_dropped_and_empty_is_empty(tmp_path, monkeypatch):
    scenes = make_scenes(tmp_path, [(0, 10)])
    scenes[0]["keyframe_path"] = str(tmp_path / "missing.png")
    analyzer = analyzer_with(monkeypatch, [unit(1, 0)])
    assert analyzer.filter_slide_segments(scenes, max_segments=80) == []
    assert analyzer.filter_slide_segments([], max_segments=80) == []


def test_embedding_failure_skips_dedup_but_keeps_segments(tmp_path, monkeypatch):
    scenes = make_scenes(tmp_path, [(0, 10), (10, 20)])
    analyzer = SemanticAnalyzer()

    def boom(_scenes):
        raise RuntimeError("clip unavailable")

    monkeypatch.setattr(analyzer, "_embed_keyframes", boom)
    assert len(analyzer.filter_slide_segments(scenes, max_segments=80)) == 2

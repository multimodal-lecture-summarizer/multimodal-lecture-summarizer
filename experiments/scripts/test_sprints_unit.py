"""Unit tests for sprint functions (no pipeline, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.sprints import (
    sprint1_smooth_chapters,
    sprint3_enrich_captions,
    sprint4_prune_dense_keyframes,
    sprint5_boost_chapter_coverage,
    sprint7_transcript_caption_fallback,
)


def test_sprint1_merges_short_chapter():
    chapters = [
        {"title": "A", "startTime": 0, "endTime": 100},
        {"title": "B", "startTime": 100, "endTime": 101, "summary": "tail"},
    ]
    out, stats = sprint1_smooth_chapters(chapters, min_dur_sec=45)
    assert len(out) == 1, stats
    assert out[0]["endTime"] == 101
    print("PASS sprint1_merges_short_chapter")


def test_sprint3_enriches_generic():
    kfs = [{"description": "Keyframe for Scene 3", "ocr_text": "Introduction to ML"}]
    out, stats = sprint3_enrich_captions(kfs)
    assert stats["enriched_from_ocr"] == 1
    assert "Introduction" in out[0]["description"]
    print("PASS sprint3_enriches_generic")


def test_sprint4_keeps_top_evidence_in_window():
    kfs = [
        {"timestamp": 0, "transcript": "long " * 20, "description": "A", "importanceScore": 0.9},
        {"timestamp": 5, "transcript": "", "description": "B", "importanceScore": 0.5},
        {"timestamp": 10, "transcript": "x", "description": "C", "importanceScore": 0.5},
    ]
    out, stats = sprint4_prune_dense_keyframes(kfs, window_sec=45, max_per_window=2)
    assert stats["output"] == 2
    assert all((k.get("transcript") or "").strip() for k in out)
    print("PASS sprint4_keeps_top_evidence")


def test_sprint7_enriches_from_transcript():
    kfs = [{
        "description": "Keyframe for Scene 21",
        "transcript": "This is an entire book. So this is an example of non-image data.",
    }]
    out, stats = sprint7_transcript_caption_fallback(kfs)
    assert stats["enriched_from_transcript"] == 1
    assert "Slide context:" in out[0]["description"]
    print("PASS sprint7_transcript_fallback")


def test_sprint4_v2_restores_chapter_gap():
    from experiments.pipeline.sprints import sprint4_v2_chapter_aware_prune

    chapters = [
        {"startTime": 0, "endTime": 100},
        {"startTime": 100, "endTime": 200},
    ]
    kfs = [
        {"timestamp": 10, "transcript": "a" * 50, "description": "A", "importanceScore": 0.9},
        {"timestamp": 15, "transcript": "b" * 50, "description": "B", "importanceScore": 0.8},
        {"timestamp": 110, "transcript": "only in ch2", "description": "C", "importanceScore": 0.7},
    ]
    kept, stats = sprint4_v2_chapter_aware_prune(chapters, kfs, window_sec=50, max_per_window=1)
    assert any(float(k["timestamp"]) == 110 for k in kept)
    print("PASS sprint4_v2_chapter_restore")


def test_sprint5_boosts_in_chapter():
    chapters = [{"title": "Ch1", "startTime": 0, "endTime": 60}]
    kfs = [{"timestamp": 30, "transcript": "hello", "description": "Slide", "importanceScore": 0.5}]
    _, out, stats = sprint5_boost_chapter_coverage(chapters, kfs)
    assert stats["keyframes_boosted"] == 1
    assert out[0]["importanceScore"] > 0.5
    print("PASS sprint5_boosts")


def main():
    tests = [
        test_sprint1_merges_short_chapter,
        test_sprint3_enriches_generic,
        test_sprint4_keeps_top_evidence_in_window,
        test_sprint7_enriches_from_transcript,
        test_sprint4_v2_restores_chapter_gap,
        test_sprint5_boosts_in_chapter,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"RESULT: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

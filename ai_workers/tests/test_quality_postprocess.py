"""Smoke tests for production quality post-process (sprint stack)."""

from __future__ import annotations

import unittest

from ai_workers.modules.fusion.quality_postprocess import (
    RECOMMENDED_STACK,
    apply_quality_postprocess,
    is_generic_caption,
    resolve_stack,
    sprint1_smooth_chapters,
    sprint3_enrich_captions,
)


class QualityPostprocessTests(unittest.TestCase):
    def test_resolve_stack_presets(self):
        self.assertEqual(resolve_stack("recommended"), RECOMMENDED_STACK)
        self.assertIn("sprint10", resolve_stack("sprint10"))
        self.assertEqual(resolve_stack("unknown"), RECOMMENDED_STACK)

    def test_sprint1_merges_short_chapters(self):
        chapters = [
            {"title": "A", "startTime": 0.0, "endTime": 10.0, "summary": "a"},
            {"title": "B", "startTime": 10.0, "endTime": 20.0, "summary": "b"},
            {"title": "C", "startTime": 20.0, "endTime": 100.0, "summary": "c"},
        ]
        out, stats = sprint1_smooth_chapters(chapters, min_dur_sec=45.0)
        self.assertEqual(stats["input"], 3)
        self.assertLess(stats["output"], 3)
        # Short A+B merge into one prefix; long C stays (or absorbs if still short)
        self.assertGreaterEqual(max(c["endTime"] - c["startTime"] for c in out), 45.0)

    def test_sprint3_enriches_generic_from_ocr(self):
        keyframes = [
            {
                "timestamp": 1.0,
                "description": "Keyframe for Scene 1",
                "ocr_text": "Neural Networks Overview",
                "transcript": "",
            }
        ]
        out, stats = sprint3_enrich_captions(keyframes)
        self.assertEqual(stats["enriched_from_ocr"], 1)
        self.assertTrue(out[0]["description"].startswith("Slide Text:"))
        self.assertFalse(is_generic_caption(out[0]["description"]))

    def test_recommended_stack_end_to_end(self):
        chapters = [
            {"title": "Intro", "startTime": 0.0, "endTime": 8.0, "summary": "hi"},
            {"title": "Main", "startTime": 8.0, "endTime": 90.0, "summary": "body"},
        ]
        keyframes = [
            {
                "timestamp": 5.0,
                "description": "Keyframe for Scene 1",
                "ocr_text": "Agenda",
                "transcript": "welcome everyone",
                "importanceScore": 0.5,
            },
            {
                "timestamp": 12.0,
                "description": "Keyframe for Scene 2",
                "ocr_text": "",
                "transcript": "deep learning basics",
                "importanceScore": 0.6,
            },
            {
                "timestamp": 20.0,
                "description": "Keyframe for Scene 3",
                "ocr_text": "loss function",
                "transcript": "we minimize the loss",
                "importanceScore": 0.7,
            },
        ]
        result = apply_quality_postprocess(
            chapters=chapters,
            keyframes=keyframes,
            stack_name="recommended",
            min_chapter_sec=45.0,
        )
        self.assertEqual(result["stack"], RECOMMENDED_STACK)
        self.assertIn("sprint1", result["sprint_stats"])
        self.assertIn("sprint3", result["sprint_stats"])
        self.assertIn("sprint4_v2", result["sprint_stats"])
        self.assertIn("sprint7", result["sprint_stats"])
        self.assertGreaterEqual(len(result["chapters"]), 1)
        self.assertGreaterEqual(len(result["keyframes"]), 1)
        # After S3/S7, generic captions should be reduced
        generic = sum(
            1 for k in result["keyframes"] if is_generic_caption(k.get("description") or "")
        )
        self.assertLess(generic, 3)


if __name__ == "__main__":
    unittest.main()

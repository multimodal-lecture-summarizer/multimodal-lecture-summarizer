from __future__ import annotations

import builtins
import unittest
from unittest.mock import patch

from ai_workers.modules.fusion.timeline import TimelineBuilder


class TimelineChapteringTests(unittest.TestCase):
    def test_segment_chapters_does_not_load_sentence_transformer(self):
        utterances = [
            {"start": 0.0, "end": 25.0, "text": "intro agenda lecture goals"},
            {"start": 35.0, "end": 55.0, "text": "intro course outline topics"},
            {"start": 70.0, "end": 95.0, "text": "gradient descent loss optimization"},
            {"start": 110.0, "end": 135.0, "text": "gradient learning rate convergence"},
            {"start": 155.0, "end": 185.0, "text": "evaluation metrics accuracy precision"},
            {"start": 205.0, "end": 235.0, "text": "confusion matrix recall validation"},
        ]
        slides = [
            {"start_seconds": 0.0, "end_seconds": 70.0},
            {"start_seconds": 70.0, "end_seconds": 155.0},
            {"start_seconds": 155.0, "end_seconds": 235.0},
        ]

        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise AssertionError("Timeline chaptering must not load SentenceTransformer")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            chapters = TimelineBuilder().segment_chapters(utterances, slides)

        self.assertGreaterEqual(len(chapters), 1)
        self.assertEqual(chapters[0]["startTime"], 0.0)
        self.assertGreater(chapters[-1]["endTime"], chapters[0]["startTime"])


if __name__ == "__main__":
    unittest.main()

"""Unit and integration tests for Summary Length & Cognitive Depth Re-summarize endpoint (Phase 1)."""

from __future__ import annotations

import os
import sys
import uuid
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
REPO_ROOT = BACKEND_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.summary import Summary
from app.models.video import Video
from app.schemas.summary import SummaryDTO, ResummarizeRequest
from app.api.v1.summaries import format_summary_dto, resummarize_video


class SummaryResummarizeTests(unittest.TestCase):
    def setUp(self):
        self.video_id = uuid.uuid4()
        self.user_id = uuid.uuid4()
        self.mock_user = MagicMock(user_id=self.user_id, output_language="vi")
        self.mock_video = MagicMock(
            video_id=self.video_id,
            user_id=self.user_id,
            duration=120.0,
            title="Video Test Bài Giảng",
        )
        self.mock_summary = Summary(
            summary_id=uuid.uuid4(),
            video_id=self.video_id,
            summary_text="Bản tóm tắt tiêu chuẩn ban đầu với khoảng mười từ cơ bản.",
            summary_length="medium",
            summary_variants_json={
                "medium": {
                    "summary_text": "Bản tóm tắt tiêu chuẩn ban đầu với khoảng mười từ cơ bản.",
                    "key_takeaways_json": [{"timestamp": 0.0, "title": "Khởi đầu", "description": "Ý chính"}],
                    "word_count": 12,
                    "reading_time_sec": 4,
                    "model_used": "Groq Llama 3.1 8B",
                    "generated_at": "2026-10-25T10:00:00",
                }
            },
            chapters_json=[
                {"title": "Chương 1", "startTime": 0.0, "endTime": 60.0, "summary": "Nội dung chương 1"}
            ],
            keyframes_json=[
                {"timestamp": 10.0, "imageUrl": "http://img.png", "description": "Slide 1", "importanceScore": 0.9}
            ],
            key_takeaways_json=[{"timestamp": 0.0, "title": "Khởi đầu", "description": "Ý chính"}],
            objects_summary_json={"unique_classes": ["motorcycle"], "total_tracked_objects": 1},
            transcript_text="Xin chào các bạn, đây là bài giảng hướng dẫn kỹ thuật thực hành.",
            model_used="Groq Llama 3.1 8B",
            processing_time=15.2,
        )

    def test_schema_resummarize_request(self):
        req1 = ResummarizeRequest(target_length="short", force_regenerate=True)
        self.assertEqual(req1.target_length, "short")
        self.assertTrue(req1.force_regenerate)

        # CamelCase alias test
        req2 = ResummarizeRequest.model_validate({"targetLength": "detailed", "forceRegenerate": False})
        self.assertEqual(req2.target_length, "detailed")
        self.assertFalse(req2.force_regenerate)

    def test_format_summary_dto_fields(self):
        dto = format_summary_dto(self.mock_video, self.mock_summary)
        self.assertIsInstance(dto, SummaryDTO)
        self.assertEqual(dto.summary_length, "medium")
        self.assertIn("medium", dto.summary_variants_json)
        self.assertTrue(dto.word_count > 0)
        self.assertTrue(dto.reading_time_sec > 0)
        self.assertEqual(len(dto.chapters), 1)
        self.assertEqual(len(dto.keyframes), 1)

    def test_cache_hit_fast_path(self):
        # Add 'short' variant to cache
        self.mock_summary.summary_variants_json["short"] = {
            "summary_text": "Tóm tắt ngắn gọn lướt nhanh TL;DR.",
            "key_takeaways_json": [{"timestamp": 0.0, "title": "TLDR", "description": "Takeaway ngắn"}],
            "word_count": 8,
            "reading_time_sec": 2,
            "model_used": "Groq Llama 3.1 8B",
            "generated_at": "2026-10-25T10:05:00",
        }

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            self.mock_video,     # video check
            self.mock_summary,   # summary check
        ]

        payload = ResummarizeRequest(target_length="short", force_regenerate=False)
        with patch("ai_workers.modules.fusion.summarizer.Summarizer") as mock_summarizer:
            resp = resummarize_video(
                video_id=self.video_id,
                payload=payload,
                db=mock_db,
                current_user=self.mock_user,
            )
            mock_summarizer.assert_not_called()

        self.assertTrue(resp.success)
        self.assertEqual(resp.data.summary_length, "short")
        self.assertEqual(resp.data.summary_text, "Tóm tắt ngắn gọn lướt nhanh TL;DR.")
        self.assertIn("Cache Hit", resp.message)

    def test_cache_miss_on_demand_generation(self):
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            self.mock_video,     # video check
            self.mock_summary,   # summary check
            None,                # job check (DB fallback)
        ]
        mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = []

        payload = ResummarizeRequest(target_length="detailed", force_regenerate=False)

        mock_generated_result = {
            "video_title": "Bài giảng chuyên sâu",
            "summary": "### 1. Phân tích Chi tiết\nNội dung giải thích cặn kẽ từng bước kỹ thuật và nguyên lý.",
            "key_takeaways": [{"timestamp": 5.0, "title": "Chi tiết 1", "description": "Mô tả chi tiết"}],
            "chapters": [{"title": "Chương 1", "startTime": 0.0, "endTime": 60.0, "summary": "Chi tiết"}],
            "word_count": 22,
            "reading_time_sec": 7,
            "model_used": "OpenRouter qwen-2.5-7b",
        }

        with patch("ai_workers.modules.fusion.summarizer.Summarizer.process", return_value=mock_generated_result) as mock_process:
            resp = resummarize_video(
                video_id=self.video_id,
                payload=payload,
                db=mock_db,
                current_user=self.mock_user,
            )
            mock_process.assert_called_once()
            called_kwargs = mock_process.call_args[1]
            self.assertEqual(called_kwargs.get("summary_length"), "detailed")

        self.assertTrue(resp.success)
        self.assertEqual(resp.data.summary_length, "detailed")
        self.assertIn("detailed", self.mock_summary.summary_variants_json)
        self.assertIn("On-Demand Generation", resp.message)
        self.assertEqual(resp.data.summary_text, mock_generated_result["summary"])


if __name__ == "__main__":
    unittest.main()

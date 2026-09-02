"""Unit and Integration tests for Extractive Fallback & Graceful Degradation (Day 3).

Verifies that when LLM providers fail or time out, a deterministic extractive summary
is generated from intermediate data, adhering strictly to schema, chapter time bounding,
redundancy pruning, state transitions (completed_partial), checkpoint resumption, and log sanitization.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import openai

from ai_workers.core.checkpoint import (
    CheckpointManager,
    STAGE_QUALITY_COMPLETE,
    STATUS_INTERMEDIATE_READY,
)
from ai_workers.modules.summarization.errors import LLMTimeoutError
from ai_workers.modules.summarization.extractive_fallback import ExtractiveSummarizer
from ai_workers.modules.summarization.summarizer import Summarizer


class ExtractiveFallbackTests(unittest.TestCase):
    """Day 3 Extractive Fallback Test Suite."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mls_test_fallback_")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        self.sample_utterances = [
            {"start": 0.0, "end": 10.0, "text": "Xin chào các bạn, hôm nay chúng ta tìm hiểu về Trí tuệ Nhân tạo và Machine Learning.", "speaker": "SPEAKER_00"},
            {"start": 10.5, "end": 25.0, "text": "Khái niệm quan trọng là mô hình học có giám sát sử dụng dữ liệu có gán nhãn để huấn luyện.", "speaker": "SPEAKER_00"},
            {"start": 26.0, "end": 45.0, "text": "Ví dụ tiêu biểu gồm có hồi quy tuyến tính và phân loại hình ảnh.", "speaker": "SPEAKER_00"},
            {"start": 46.0, "end": 65.0, "text": "Tóm lại, thuật toán cần tối thiểu hàm mất mát và tối ưu trọng số gradient descent.", "speaker": "SPEAKER_00"},
            {"start": 66.0, "end": 90.0, "text": "Trong phần tiếp theo, chúng ta chuyển sang mạng nơ-ron nhân tạo và Deep Learning.", "speaker": "SPEAKER_00"},
        ]
        self.sample_slides = [
            {
                "start_seconds": 0.0,
                "end_seconds": 45.0,
                "start_timecode": "00:00",
                "caption": "Slide mở đầu tổng quan Trí tuệ Nhân tạo",
                "ocr_text": "TRÍ TUỆ NHÂN TẠO\nHọc có giám sát & Không giám sát",
                "importanceScore": 0.9,
                "keyframe_url": "http://img.r2/slide1.png",
            },
            {
                "start_seconds": 46.0,
                "end_seconds": 90.0,
                "start_timecode": "00:46",
                "caption": "Slide tối ưu hóa và Gradient Descent",
                "ocr_text": "TỐI ƯU HÓA HÀM MẤT MÁT\nGradient Descent",
                "importanceScore": 0.85,
                "keyframe_url": "http://img.r2/slide2.png",
            }
        ]
        self.sample_chapters = [
            {"title": "Chương 1: Khái niệm Machine Learning", "startTime": 0.0, "endTime": 45.0, "summary": "Mở đầu"},
            {"title": "Chương 2: Tối ưu hóa và Deep Learning", "startTime": 46.0, "endTime": 90.0, "summary": "Nâng cao"},
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Case 1: LLM Success -> Fallback is NOT called
    # -------------------------------------------------------------------------
    def test_case_1_llm_success_does_not_invoke_fallback(self):
        summarizer = Summarizer(llm_timeout=10.0, overall_timeout=90.0)
        summarizer.openrouter_client._client = MagicMock()

        mock_llm_json = {
            "video_title": "Bài giảng Machine Learning",
            "summary": "Tóm tắt từ LLM OpenRouter thành công.",
            "key_takeaways": ["Ý chính 1", "Ý chính 2"],
            "chapters": [
                {"title": "Chương 1", "summary": "Tóm tắt chương 1", "startTime": 0.0, "endTime": 45.0},
                {"title": "Chương 2", "summary": "Tóm tắt chương 2", "startTime": 46.0, "endTime": 90.0},
            ]
        }
        summarizer.openrouter_client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=json.dumps(mock_llm_json)))]
        )

        with patch("ai_workers.modules.summarization.extractive_fallback.ExtractiveSummarizer.generate_fallback") as mock_fallback:
            result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-success")
            mock_fallback.assert_not_called()

        self.assertEqual(result["fallback_used"], False)
        self.assertEqual(result["summary_method"], "llm")
        self.assertIn("OpenRouter", result["model_used"])
        self.assertEqual(result["video_title"], "Bài giảng Machine Learning")

    # -------------------------------------------------------------------------
    # Case 2: All LLM Candidates Fail -> Extractive Fallback creates non-empty summary
    # -------------------------------------------------------------------------
    def test_case_2_all_llms_fail_activates_extractive_fallback(self):
        summarizer = Summarizer(llm_timeout=5.0, overall_timeout=10.0)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        # All LLM calls raise Timeout
        summarizer.openrouter_client._client.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())
        summarizer.groq_client._client.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())

        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-fallback-1")

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["summary_method"], "extractive_fallback")
        self.assertIn("Extractive Fallback", result["model_used"])
        self.assertTrue(len(result["summary"]) > 20)
        self.assertTrue(len(result["key_takeaways"]) >= 2)
        self.assertEqual(len(result["chapters"]), 2)
        self.assertIsNotNone(result.get("llm_error"))

    # -------------------------------------------------------------------------
    # Case 3: Fallback Output matches exact LLM schema
    # -------------------------------------------------------------------------
    def test_case_3_fallback_output_schema_compatibility(self):
        extractive = ExtractiveSummarizer()
        result = extractive.generate_fallback(
            self.sample_utterances,
            self.sample_slides,
            self.sample_chapters,
            llm_error={"error_code": "LLM_TIMEOUT", "message": "Timed out"},
            job_id="job-schema",
        )

        # Validate top-level keys
        required_keys = ["video_title", "summary", "key_takeaways", "chapters", "model_used", "fallback_used", "summary_method", "llm_error"]
        for k in required_keys:
            self.assertIn(k, result, f"Missing key {k} in fallback output")

        self.assertIsInstance(result["video_title"], str)
        self.assertIsInstance(result["summary"], str)
        self.assertIsInstance(result["key_takeaways"], list)
        self.assertIsInstance(result["chapters"], list)

        # Validate chapter schema
        for c in result["chapters"]:
            self.assertIn("title", c)
            self.assertIn("startTime", c)
            self.assertIn("endTime", c)
            self.assertIn("summary", c)
            self.assertTrue(len(c["summary"]) > 0)

    # -------------------------------------------------------------------------
    # Case 4: State Transition - completed_partial in process_video
    # -------------------------------------------------------------------------
    def test_case_4_task_sets_completed_partial_on_fallback(self):
        from ai_workers import tasks

        dummy_audio = {"text": "Xin chào...", "segments": self.sample_utterances}
        dummy_visual = {"scenes": self.sample_slides, "keyframes": []}
        dummy_timeline = {"chapters": self.sample_chapters, "aligned_segments": []}

        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_mod.AudioTranscriber = MagicMock()
        mock_audio_mod.AudioTranscriber.return_value.process.return_value = dummy_audio

        mock_speaker_mod = ModuleType("ai_workers.modules.audio_v2.speaker")
        mock_speaker_mod.SpeakerDiarizer = MagicMock()
        mock_speaker_mod.SpeakerDiarizer.return_value.process.return_value = self.sample_utterances

        mock_scene_mod = ModuleType("ai_workers.modules.visual_v2.scene_detector")
        mock_scene_mod.SceneDetector = MagicMock()
        mock_scene_mod.SceneDetector.return_value.process.return_value = dummy_visual

        mock_semantic_mod = ModuleType("ai_workers.modules.visual_v2.semantic")
        mock_semantic_mod.SemanticAnalyzer = MagicMock()
        mock_semantic_mod.SemanticAnalyzer.return_value.filter_scenes_clip.return_value = dummy_visual["scenes"]

        mock_timeline_mod = ModuleType("ai_workers.modules.fusion.timeline")
        mock_timeline_mod.TimelineBuilder = MagicMock()
        mock_timeline_mod.TimelineBuilder.return_value.process.return_value = dummy_timeline

        mock_quality_mod = ModuleType("ai_workers.modules.fusion.quality_postprocess")
        mock_quality_mod.apply_quality_postprocess = MagicMock(return_value={"chapters": self.sample_chapters, "keyframes": []})

        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        # Summarizer returns fallback result
        mock_summarizer_class.return_value.process.return_value = {
            "video_title": "Tóm tắt trích xuất: Trí tuệ Nhân tạo",
            "summary": "### Tổng quan bài giảng\n\nNội dung bài giảng được trích xuất tự động.",
            "key_takeaways": ["Machine learning có giám sát"],
            "chapters": self.sample_chapters,
            "model_used": "Extractive Fallback (TF-IDF + Multimodal)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": {"error_code": "ALL_PROVIDERS_FAILED", "message": "All LLMs failed"},
        }
        mock_summarizer_mod.Summarizer = mock_summarizer_class

        mock_modules = {
            "ai_workers.modules.audio_v2.transcriber": mock_audio_mod,
            "ai_workers.modules.audio_v2.speaker": mock_speaker_mod,
            "ai_workers.modules.visual_v2.scene_detector": mock_scene_mod,
            "ai_workers.modules.visual_v2.semantic": mock_semantic_mod,
            "ai_workers.modules.fusion.timeline": mock_timeline_mod,
            "ai_workers.modules.fusion.summarizer": mock_summarizer_mod,
            "ai_workers.modules.fusion.quality_postprocess": mock_quality_mod,
        }

        job_id = "job-partial-state"
        from ai_workers.core.config import worker_settings

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            result = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            self.assertEqual(result["status"], "done")
            self.assertEqual(result["stage"], "completed_partial")
            self.assertEqual(result["fallback_used"], True)
            self.assertEqual(result["summary_method"], "extractive_fallback")
            self.assertIn("llm_error", result)

    # -------------------------------------------------------------------------
    # Case 5: Chapter Summary Bounded to Chapter Time Range
    # -------------------------------------------------------------------------
    def test_case_5_chapter_summary_bounded_strictly_to_timeframe(self):
        extractive = ExtractiveSummarizer()
        result = extractive.generate_fallback(
            self.sample_utterances,
            self.sample_slides,
            self.sample_chapters,
            job_id="job-chapter-bounds",
        )

        chap1 = result["chapters"][0]
        chap2 = result["chapters"][1]

        # Chap 1 ([0, 45s]) should contain sentences about Machine Learning / Học có giám sát
        self.assertTrue("Machine Learning" in chap1["summary"] or "học có giám sát" in chap1["summary"] or "Trí tuệ Nhân tạo" in chap1["summary"])
        # Chap 1 should NOT contain Chapter 2's deep learning / gradient descent discussion from t=66-90s
        self.assertNotIn("mạng nơ-ron nhân tạo", chap1["summary"])

        # Chap 2 ([46, 90s]) should contain gradient descent / tối ưu hóa / mạng nơ-ron
        self.assertTrue("gradient descent" in chap2["summary"] or "mạng nơ-ron" in chap2["summary"] or "tối ưu" in chap2["summary"])

    # -------------------------------------------------------------------------
    # Case 6: Duplicate & Near-Duplicate Sentence Pruning
    # -------------------------------------------------------------------------
    def test_case_6_redundancy_and_duplicate_pruning(self):
        extractive = ExtractiveSummarizer(similarity_threshold=0.5)
        redundant_utterances = [
            {"start": 0.0, "end": 10.0, "text": "Học máy sử dụng thuật toán gradient descent để tối ưu hàm mất mát."},
            {"start": 10.0, "end": 20.0, "text": "Thuật toán gradient descent được học máy sử dụng nhằm tối ưu hàm mất mát."},  # Near-duplicate
            {"start": 20.0, "end": 30.0, "text": "Học máy dùng gradient descent để tối ưu hóa các hàm mất mát."},  # Near-duplicate
            {"start": 30.0, "end": 40.0, "text": "Mạng nơ-ron tích chập CNN rất hiệu quả trong phân loại ảnh y tế."},  # Distinct
        ]

        result = extractive.generate_fallback(
            redundant_utterances,
            slides=[],
            chapters=[{"title": "Chương 1", "startTime": 0.0, "endTime": 40.0, "summary": "Ch1"}],
        )

        exec_summary = result["summary"]
        # Ensure gradient descent is mentioned, but CNN is also included rather than 3 repeating gradient sentences
        self.assertIn("gradient descent", exec_summary.lower())
        self.assertIn("cnn", exec_summary.lower())

    # -------------------------------------------------------------------------
    # Case 7: Sparse / Minimal / Audio-only / Slide-only Inputs do not crash
    # -------------------------------------------------------------------------
    def test_case_7_sparse_inputs_handled_gracefully(self):
        extractive = ExtractiveSummarizer()

        # 1. Single short utterance
        res1 = extractive.generate_fallback(
            [{"start": 0.0, "end": 5.0, "text": "Đây là bài kiểm tra ngắn về lập trình Python."}],
            slides=[],
            chapters=[],
        )
        self.assertEqual(res1["status"], "done")
        self.assertTrue(len(res1["summary"]) > 0)

        # 2. Slide only (0 audio utterances)
        res2 = extractive.generate_fallback(
            utterances=[],
            slides=[{"start_seconds": 0.0, "end_seconds": 30.0, "ocr_text": "SLIDE TIÊU ĐỀ\nGiới thiệu Khoa học Dữ liệu", "caption": "Slide mở đầu"}],
            chapters=[{"title": "Mở đầu", "startTime": 0.0, "endTime": 30.0}],
        )
        self.assertEqual(res2["status"], "done")
        self.assertIn("Khoa học Dữ liệu", res2["summary"])

    # -------------------------------------------------------------------------
    # Case 8: Resume from Checkpoint + LLM Failure -> Fallback skips heavy stages
    # -------------------------------------------------------------------------
    def test_case_8_resume_from_checkpoint_runs_fallback_without_heavy_stages(self):
        from ai_workers import tasks

        job_id = "job-resume-fallback"

        # Pre-populate Checkpoint (Day 2)
        self.checkpoint_manager.save_checkpoint(
            job_id=job_id,
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data={
                "utterances": self.sample_utterances,
                "audio_result": {"text": "Full text", "segments": self.sample_utterances},
                "visual_result": {"scenes": self.sample_slides},
                "timeline_result": {"chapters": self.sample_chapters},
                "chapters": self.sample_chapters,
                "keyframes": [],
                "sprint_stats": {"sprint1": 1},
                "export_meta": {"version": 2},
                "video_file_url": "http://mock.mp4",
            }
        )

        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_mod.AudioTranscriber = MagicMock()

        mock_speaker_mod = ModuleType("ai_workers.modules.audio_v2.speaker")
        mock_speaker_mod.SpeakerDiarizer = MagicMock()

        mock_scene_mod = ModuleType("ai_workers.modules.visual_v2.scene_detector")
        mock_scene_mod.SceneDetector = MagicMock()

        mock_semantic_mod = ModuleType("ai_workers.modules.visual_v2.semantic")
        mock_semantic_mod.SemanticAnalyzer = MagicMock()

        mock_timeline_mod = ModuleType("ai_workers.modules.fusion.timeline")
        mock_timeline_mod.TimelineBuilder = MagicMock()

        mock_quality_mod = ModuleType("ai_workers.modules.fusion.quality_postprocess")
        mock_quality_mod.apply_quality_postprocess = MagicMock()

        # Summarizer fails with LLM timeout and returns extractive fallback
        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        mock_summarizer_class.return_value.process.return_value = {
            "video_title": "Tóm tắt: Machine Learning",
            "summary": "Tóm tắt trích xuất sau khi resume.",
            "key_takeaways": ["Ý 1"],
            "chapters": self.sample_chapters,
            "model_used": "Extractive Fallback (TF-IDF + Multimodal)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": {"error_code": "LLM_TIMEOUT", "message": "Timed out"},
        }
        mock_summarizer_mod.Summarizer = mock_summarizer_class

        mock_modules = {
            "ai_workers.modules.audio_v2.transcriber": mock_audio_mod,
            "ai_workers.modules.audio_v2.speaker": mock_speaker_mod,
            "ai_workers.modules.visual_v2.scene_detector": mock_scene_mod,
            "ai_workers.modules.visual_v2.semantic": mock_semantic_mod,
            "ai_workers.modules.fusion.timeline": mock_timeline_mod,
            "ai_workers.modules.fusion.summarizer": mock_summarizer_mod,
            "ai_workers.modules.fusion.quality_postprocess": mock_quality_mod,
        }

        from ai_workers.core.config import worker_settings

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            result = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            # Check heavy stages were SKIPPED
            mock_audio_mod.AudioTranscriber.assert_not_called()
            mock_speaker_mod.SpeakerDiarizer.assert_not_called()
            mock_scene_mod.SceneDetector.assert_not_called()
            mock_semantic_mod.SemanticAnalyzer.assert_not_called()
            mock_timeline_mod.TimelineBuilder.assert_not_called()
            mock_quality_mod.apply_quality_postprocess.assert_not_called()

            # Check status was completed_partial
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["stage"], "completed_partial")
            self.assertEqual(result["fallback_used"], True)

    # -------------------------------------------------------------------------
    # Case 9: LLM + Fallback both fail (0 input data) -> status = failed
    # -------------------------------------------------------------------------
    def test_case_9_total_failure_sets_job_failed(self):
        from ai_workers import tasks

        job_id = "job-total-fail"
        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        # Fallback returns failure because there was no content
        mock_summarizer_class.return_value.process.return_value = {
            "status": "failed",
            "video_title": "Không có dữ liệu bài giảng",
            "summary": "Không thể trích xuất nội dung bài giảng do thiếu bản ghi.",
            "chapters": [],
            "model_used": "Extractive Fallback (Failed: No Content)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": {"error_code": "NO_CONTENT", "message": "No utterances"},
        }
        mock_summarizer_mod.Summarizer = mock_summarizer_class

        # Checkpoint with empty data
        self.checkpoint_manager.save_checkpoint(
            job_id=job_id,
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data={"utterances": [], "chapters": [], "keyframes": []},
        )

        mock_modules = {
            "ai_workers.modules.audio_v2.transcriber": MagicMock(),
            "ai_workers.modules.audio_v2.speaker": MagicMock(),
            "ai_workers.modules.visual_v2.scene_detector": MagicMock(),
            "ai_workers.modules.visual_v2.semantic": MagicMock(),
            "ai_workers.modules.fusion.timeline": MagicMock(),
            "ai_workers.modules.fusion.summarizer": mock_summarizer_mod,
            "ai_workers.modules.fusion.quality_postprocess": MagicMock(),
        }

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            result = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "failed")
            self.assertNotEqual(result["status"], "done")
            self.assertNotEqual(result["stage"], "completed")

    # -------------------------------------------------------------------------
    # Case 10: Log Sanitization during Fallback
    # -------------------------------------------------------------------------
    def test_case_10_log_sanitization_during_fallback(self):
        summarizer = Summarizer()
        secret_key = "sk-openrouter-secret-key-123456789"
        auth_error = openai.AuthenticationError(
            message=f"Invalid API Key: Bearer {secret_key}",
            response=MagicMock(status_code=401),
            body=None,
        )

        summarizer.openrouter_client._client = MagicMock()
        summarizer.openrouter_client._client.chat.completions.create.side_effect = auth_error
        summarizer.groq_client._client = MagicMock()
        summarizer.groq_client._client.chat.completions.create.side_effect = auth_error

        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-sanitize")

        # Secret key MUST NOT be present anywhere in result payload or llm_error
        dumped_json = json.dumps(result)
        self.assertNotIn(secret_key, dumped_json)


if __name__ == "__main__":
    unittest.main()

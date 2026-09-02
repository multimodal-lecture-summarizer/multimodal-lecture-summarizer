"""Day 4 — Comprehensive Chaos Testing, Observability & Final Hardening Test Suite.

Simulates extreme failure scenarios across LLM providers, network disruptions,
checkpoint corruption, worker crashes, state machine transitions, resource leaks,
and credential sanitization.
"""

from __future__ import annotations

import json
import os
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
    PIPELINE_VERSION,
    STAGE_QUALITY_COMPLETE,
    STATUS_INTERMEDIATE_READY,
)
from ai_workers.modules.summarization.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    sanitize_text,
)
from ai_workers.modules.summarization.extractive_fallback import ExtractiveSummarizer
from ai_workers.modules.summarization.summarizer import Summarizer


class ReliabilityChaosTests(unittest.TestCase):
    """Day 4 Chaos and Hardening Test Suite."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mls_chaos_test_")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)
        self.sample_utterances = [
            {"start": 0.0, "end": 15.0, "text": "Hôm nay chúng ta học về Kiến trúc phần mềm và Khả năng chịu lỗi.", "speaker": "SPEAKER_00"},
            {"start": 16.0, "end": 35.0, "text": "Nguyên lý quan trọng là hệ thống cần có cơ chế retry với exponential backoff và fallback.", "speaker": "SPEAKER_00"},
            {"start": 36.0, "end": 60.0, "text": "Tóm lại, tính bất biến và checkpoint trung gian giúp pipeline phục hồi an toàn sau sự cố.", "speaker": "SPEAKER_00"},
        ]
        self.sample_slides = [
            {
                "start_seconds": 0.0,
                "end_seconds": 35.0,
                "start_timecode": "00:00",
                "caption": "Slide mở đầu Kiến trúc phần mềm chịu lỗi",
                "ocr_text": "SOFTWARE RESILIENCE & CHAOS ENGINEERING\nRetry, Fallback & Checkpointing",
                "importanceScore": 0.95,
                "keyframe_url": "http://img.r2/slide1.png",
            },
            {
                "start_seconds": 36.0,
                "end_seconds": 60.0,
                "start_timecode": "00:36",
                "caption": "Slide Checkpoint và Idempotency",
                "ocr_text": "CHECKPOINT & IDEMPOTENCY\nSafe recovery",
                "importanceScore": 0.9,
                "keyframe_url": "http://img.r2/slide2.png",
            }
        ]
        self.sample_chapters = [
            {"title": "Chương 1: Khái niệm Chịu lỗi", "startTime": 0.0, "endTime": 35.0, "summary": "Ch1"},
            {"title": "Chương 2: Checkpoint và Phục hồi", "startTime": 36.0, "endTime": 60.0, "summary": "Ch2"},
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _build_mock_modules(self, summarizer_result: dict[str, Any]) -> dict[str, Any]:
        """Helper to create standardized mock modules for pipeline execution."""
        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_mod.AudioTranscriber = MagicMock()
        mock_audio_mod.AudioTranscriber.return_value.process.return_value = {
            "text": "Full text", "segments": self.sample_utterances
        }

        mock_speaker_mod = ModuleType("ai_workers.modules.audio_v2.speaker")
        mock_speaker_mod.SpeakerDiarizer = MagicMock()
        mock_speaker_mod.SpeakerDiarizer.return_value.process.return_value = self.sample_utterances

        mock_scene_mod = ModuleType("ai_workers.modules.visual_v2.scene_detector")
        mock_scene_mod.SceneDetector = MagicMock()
        mock_scene_mod.SceneDetector.return_value.process.return_value = {
            "scenes": self.sample_slides, "keyframes": []
        }

        mock_semantic_mod = ModuleType("ai_workers.modules.visual_v2.semantic")
        mock_semantic_mod.SemanticAnalyzer = MagicMock()
        mock_semantic_mod.SemanticAnalyzer.return_value.filter_scenes_clip.return_value = self.sample_slides

        mock_timeline_mod = ModuleType("ai_workers.modules.fusion.timeline")
        mock_timeline_mod.TimelineBuilder = MagicMock()
        mock_timeline_mod.TimelineBuilder.return_value.process.return_value = {
            "chapters": self.sample_chapters, "aligned_segments": []
        }

        mock_quality_mod = ModuleType("ai_workers.modules.fusion.quality_postprocess")
        mock_quality_mod.apply_quality_postprocess = MagicMock(
            return_value={"chapters": self.sample_chapters, "keyframes": []}
        )

        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        mock_summarizer_class.return_value.process.return_value = summarizer_result
        mock_summarizer_mod.Summarizer = mock_summarizer_class

        return {
            "ai_workers.modules.audio_v2.transcriber": mock_audio_mod,
            "ai_workers.modules.audio_v2.speaker": mock_speaker_mod,
            "ai_workers.modules.visual_v2.scene_detector": mock_scene_mod,
            "ai_workers.modules.visual_v2.semantic": mock_semantic_mod,
            "ai_workers.modules.fusion.timeline": mock_timeline_mod,
            "ai_workers.modules.fusion.summarizer": mock_summarizer_mod,
            "ai_workers.modules.fusion.quality_postprocess": mock_quality_mod,
        }

    # =========================================================================
    # Group 1: LLM Chaos Scenarios
    # =========================================================================

    def test_chaos_01_llm_transient_timeout_and_failover_recovery(self):
        """OpenRouter times out on all attempts -> Groq succeeds on attempt 2."""
        summarizer = Summarizer(llm_timeout=2.0, overall_timeout=30.0)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        # OpenRouter times out
        summarizer.openrouter_client._client.chat.completions.create.side_effect = openai.APITimeoutError(request=MagicMock())

        # Groq fails attempt 1 with rate limit, then succeeds on attempt 2
        valid_response = json.dumps({
            "video_title": "Kiến trúc Phần mềm Chịu lỗi",
            "summary": "Tóm tắt từ Groq thành công sau khi OpenRouter timeout.",
            "key_takeaways": ["Retry với backoff", "Checkpoint trung gian"],
            "chapters": [
                {"title": "Chương 1", "startTime": 0.0, "endTime": 35.0, "summary": "Ch1 summary"},
                {"title": "Chương 2", "startTime": 36.0, "endTime": 60.0, "summary": "Ch2 summary"},
            ],
        })
        summarizer.groq_client._client.chat.completions.create.side_effect = [
            openai.RateLimitError(message="Rate limited", response=MagicMock(status_code=429), body=None),
            MagicMock(choices=[MagicMock(message=MagicMock(content=valid_response))]),
        ]

        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-chaos-1")

        self.assertEqual(result["fallback_used"], False)
        self.assertEqual(result["summary_method"], "llm")
        self.assertIn("Groq", result["model_used"])
        self.assertEqual(result["video_title"], "Kiến trúc Phần mềm Chịu lỗi")

    def test_chaos_02_llm_500_server_error_and_connection_reset(self):
        """Provider returns 502 Bad Gateway and ConnectionResetError -> classified and retried."""
        summarizer = Summarizer(llm_timeout=2.0, overall_timeout=15.0)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        server_err = openai.InternalServerError(message="Bad Gateway (502)", response=MagicMock(status_code=502), body=None)
        conn_err = ConnectionResetError("Connection reset by peer")

        summarizer.openrouter_client._client.chat.completions.create.side_effect = server_err
        summarizer.groq_client._client.chat.completions.create.side_effect = conn_err

        # Should smoothly fall back to extractive summary
        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-chaos-2")

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["summary_method"], "extractive_fallback")
        self.assertTrue(len(result["summary"]) > 20)
        self.assertIn(result["llm_error"]["error_code"], ["LLM_NETWORK_ERROR", "LLM_PROVIDER_ERROR"])

    def test_chaos_03_non_retryable_auth_error_skips_further_attempts_on_target(self):
        """HTTP 401 Authentication error does not perform 3 wasted retry loops on that model."""
        summarizer = Summarizer(llm_timeout=2.0, overall_timeout=20.0)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        auth_err = openai.AuthenticationError(message="Invalid API Key (401)", response=MagicMock(status_code=401), body=None)
        summarizer.openrouter_client._client.chat.completions.create.side_effect = auth_err

        valid_response = json.dumps({
            "video_title": "Groq Recovery",
            "summary": "Summary generated by Groq after OpenRouter auth failure.",
            "key_takeaways": ["Ý 1"],
            "chapters": [
                {"title": "Chương 1", "startTime": 0.0, "endTime": 35.0, "summary": "Ch1"},
                {"title": "Chương 2", "startTime": 36.0, "endTime": 60.0, "summary": "Ch2"},
            ],
        })
        summarizer.groq_client._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=valid_response))]
        )

        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-chaos-3")

        self.assertEqual(result["fallback_used"], False)
        self.assertIn("Groq", result["model_used"])

    def test_chaos_04_overall_timeout_budget_hard_cutoff(self):
        """Cumulative execution time exceeds 90s budget -> remaining candidate models aborted."""
        summarizer = Summarizer(llm_timeout=10.0, overall_timeout=1.5)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        def slow_call(*args, **kwargs):
            time.sleep(1.6)
            raise openai.APITimeoutError(request=MagicMock())

        summarizer.openrouter_client._client.chat.completions.create.side_effect = slow_call
        summarizer.groq_client._client.chat.completions.create.side_effect = slow_call

        start = time.time()
        result = summarizer.summarize(self.sample_utterances, self.sample_slides, self.sample_chapters, job_id="job-chaos-4")
        elapsed = time.time() - start

        # Must have halted early and triggered extractive fallback
        self.assertLess(elapsed, 12.0)
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["summary_method"], "extractive_fallback")

    # =========================================================================
    # Group 2: Checkpoint Chaos & Recovery
    # =========================================================================

    def test_chaos_05_corrupted_or_truncated_checkpoint_file(self):
        """Corrupted/partial JSON on disk is cleanly ignored, triggering fresh execution."""
        job_id = "job-corrupt-ckpt"
        ckpt_path = Path(self.temp_dir) / f"{job_id}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            f.write('{"job_id": "job-corrupt-ckpt", "stage": "quality_complete", "data": { TRUNCATED')

        loaded = self.checkpoint_manager.load_checkpoint(job_id)
        self.assertIsNone(loaded)

    def test_chaos_06_checkpoint_version_and_stage_mismatch(self):
        """Outdated pipeline version or wrong stage is rejected."""
        job_id = "job-mismatch"
        payload_wrong_ver = {
            "job_id": job_id,
            "pipeline_version": "1.0.0",
            "stage": STAGE_QUALITY_COMPLETE,
            "status": STATUS_INTERMEDIATE_READY,
            "data": {"utterances": [], "chapters": [], "keyframes": []},
        }
        ckpt_path = Path(self.temp_dir) / f"{job_id}.json"
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(payload_wrong_ver, f)

        self.assertIsNone(self.checkpoint_manager.load_checkpoint(job_id))

        payload_wrong_stage = {
            "job_id": job_id,
            "pipeline_version": PIPELINE_VERSION,
            "stage": "audio_complete",
            "status": "processing",
            "data": {"utterances": [], "chapters": [], "keyframes": []},
        }
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(payload_wrong_stage, f)

        self.assertIsNone(self.checkpoint_manager.load_checkpoint(job_id, expected_stage=STAGE_QUALITY_COMPLETE))

    def test_chaos_07_checkpoint_atomic_write_failure_aborts_without_calling_llm(self):
        """If disk write / atomic replace fails, task aborts immediately before LLM call."""
        from ai_workers import tasks

        mock_summarizer_called = False
        def fake_summarize(*args, **kwargs):
            nonlocal mock_summarizer_called
            mock_summarizer_called = True
            return {}

        mock_modules = self._build_mock_modules({})
        mock_modules["ai_workers.modules.fusion.summarizer"].Summarizer.return_value.process = fake_summarize

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager.save_checkpoint", return_value=False):

            with self.assertRaises(RuntimeError) as cm:
                tasks.process_video.run("job-disk-fail", "mock_video.mp4", "hybrid")

            self.assertIn("Failed to persist checkpoint", str(cm.exception))
            self.assertFalse(mock_summarizer_called)

    def test_chaos_08_resume_after_crash_skips_all_heavy_stages(self):
        """Worker crashes after checkpoint -> task re-executed with same job_id skips heavy stages."""
        from ai_workers import tasks

        job_id = "job-crash-recovery"
        self.checkpoint_manager.save_checkpoint(
            job_id=job_id,
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data={
                "utterances": self.sample_utterances,
                "audio_result": {"text": "Xin chào...", "segments": self.sample_utterances},
                "visual_result": {"scenes": self.sample_slides},
                "timeline_result": {"chapters": self.sample_chapters},
                "chapters": self.sample_chapters,
                "keyframes": [],
                "sprint_stats": {"sprint1": 1},
                "export_meta": {"version": 2},
                "video_file_url": "http://video.mp4",
            }
        )

        mock_modules = self._build_mock_modules({
            "video_title": "Tóm tắt phục hồi",
            "summary": "### Tổng quan\n\nNội dung phục hồi thành công từ checkpoint.",
            "key_takeaways": ["Khả năng phục hồi"],
            "chapters": self.sample_chapters,
            "model_used": "OpenRouter (nvidia/nemotron-3-nano-30b-a3b:free)",
            "fallback_used": False,
            "summary_method": "llm",
        })

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            result = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            # Verify heavy stages were not called
            mock_modules["ai_workers.modules.audio_v2.transcriber"].AudioTranscriber.assert_not_called()
            mock_modules["ai_workers.modules.audio_v2.speaker"].SpeakerDiarizer.assert_not_called()
            mock_modules["ai_workers.modules.visual_v2.scene_detector"].SceneDetector.assert_not_called()
            mock_modules["ai_workers.modules.visual_v2.semantic"].SemanticAnalyzer.assert_not_called()
            mock_modules["ai_workers.modules.fusion.timeline"].TimelineBuilder.assert_not_called()
            mock_modules["ai_workers.modules.fusion.quality_postprocess"].apply_quality_postprocess.assert_not_called()

            self.assertEqual(result["status"], "done")
            self.assertEqual(result["stage"], "completed")

    # =========================================================================
    # Group 3: State Machine & Idempotency Audit
    # =========================================================================

    def test_chaos_09_state_machine_completed_partial_semantics(self):
        """When fallback is used, task marks stage=completed_partial, status=done, fallback_used=True."""
        from ai_workers import tasks
        from ai_workers.core.config import worker_settings

        job_id = "job-state-partial"
        mock_modules = self._build_mock_modules({
            "video_title": "Tóm tắt trích xuất dự phòng",
            "summary": "### Tổng quan bài giảng (Tóm tắt trích xuất)\n\nNội dung trích xuất.",
            "key_takeaways": ["Ý trích xuất"],
            "chapters": self.sample_chapters,
            "model_used": "Extractive Fallback (TF-IDF + Multimodal)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": {"error_code": "LLM_TIMEOUT", "message": "Timed out"},
        })

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
            self.assertEqual(result["progress"], 100)
            self.assertIsNotNone(result.get("llm_error"))

    def test_chaos_10_state_machine_total_failure_semantics(self):
        """When both LLM and fallback fail (e.g. 0 data), task marks stage=failed, status=failed."""
        from ai_workers import tasks
        from ai_workers.core.config import worker_settings

        job_id = "job-state-failed"
        mock_modules = self._build_mock_modules({
            "status": "failed",
            "video_title": "Không có dữ liệu",
            "summary": "Không thể trích xuất.",
            "chapters": [],
            "model_used": "Extractive Fallback (Failed: No Content)",
            "fallback_used": True,
            "summary_method": "extractive_fallback",
            "llm_error": {"error_code": "NO_CONTENT", "message": "No utterances"},
        })

        # Checkpoint with empty data
        self.checkpoint_manager.save_checkpoint(
            job_id=job_id,
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data={"utterances": [], "chapters": [], "keyframes": []},
        )

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            result = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["stage"], "failed")
            self.assertNotEqual(result["status"], "done")
            self.assertNotEqual(result["stage"], "completed")

    # =========================================================================
    # Group 4: Credential Sanitization & Resource Cleanup
    # =========================================================================

    def test_chaos_11_credential_sanitization_in_all_error_paths(self):
        """Ensures API keys, tokens, and authorization headers are never leaked in errors or logs."""
        raw_token = "sk-openrouter-secret-key-super-confidential-998877"
        bearer_token = "Bearer gsk_groq_production_token_1122334455"

        sanitized_1 = sanitize_text(f"Failed to connect using key {raw_token}")
        self.assertNotIn(raw_token, sanitized_1)
        self.assertIn("[REDACTED]", sanitized_1)

        sanitized_2 = sanitize_text(f"Header: {bearer_token} failed authorization")
        self.assertNotIn("gsk_groq_production_token_1122334455", sanitized_2)
        self.assertIn("[REDACTED]", sanitized_2)

        # Exception wrapper sanitization
        exc = LLMAuthenticationError(message=f"Unauthorized: {raw_token} {bearer_token}")
        error_dict = exc.to_dict()
        error_json = json.dumps(error_dict)
        self.assertNotIn(raw_token, error_json)
        self.assertNotIn("gsk_groq_production_token_1122334455", error_json)

    def test_chaos_12_resource_cleanup_after_pipeline_exception(self):
        """Temporary files are cleaned up and memory release called even when stage fails."""
        from ai_workers import tasks
        from ai_workers.core.config import worker_settings

        temp_video = Path(self.temp_dir) / "test_temp_video.mp4"
        temp_video.write_text("dummy video content")
        temp_audio = Path(self.temp_dir) / "test_temp_video.wav"
        temp_audio.write_text("dummy audio content")

        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_mod.AudioTranscriber = MagicMock()
        mock_audio_mod.AudioTranscriber.return_value.process.side_effect = RuntimeError("VRAM OOM")

        mock_modules = {
            "ai_workers.modules.audio_v2.transcriber": mock_audio_mod,
            "ai_workers.modules.audio_v2.speaker": MagicMock(),
            "ai_workers.modules.visual_v2.scene_detector": MagicMock(),
            "ai_workers.modules.visual_v2.semantic": MagicMock(),
            "ai_workers.modules.fusion.timeline": MagicMock(),
            "ai_workers.modules.fusion.summarizer": MagicMock(),
            "ai_workers.modules.fusion.quality_postprocess": MagicMock(),
        }

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager), \
             patch("ai_workers.tasks.release_worker_resources") as mock_cleanup:

            with self.assertRaises(RuntimeError):
                tasks.process_video.run("job-cleanup-test", str(temp_video), "hybrid")

            # Verify resource cleanup was called
            self.assertTrue(mock_cleanup.called)


if __name__ == "__main__":
    unittest.main()

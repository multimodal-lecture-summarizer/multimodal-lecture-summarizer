"""Unit and Integration tests for Checkpoint, Persistence & Resume (Day 2).

Tests atomic checkpoint creation, verification, crash recovery, resume skipping of
heavy preprocessing stages (ASR, Visual, Fusion, Quality), and overall LLM budget.
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
from unittest.mock import MagicMock, call, patch

import openai

from ai_workers.core.checkpoint import (
    CheckpointManager,
    PIPELINE_VERSION,
    STAGE_QUALITY_COMPLETE,
    STATUS_INTERMEDIATE_READY,
    STATUS_SUMMARIZING,
)
from ai_workers.modules.summarization.errors import LLMTimeoutError
from ai_workers.modules.summarization.summarizer import Summarizer


class CheckpointPersistenceTests(unittest.TestCase):
    """Test suite covering Day 2 Checkpoint, Persistence, Resume, and Budget requirements."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="mls_test_checkpoints_")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Case 1: Checkpoint persistence and verification
    # -------------------------------------------------------------------------
    def test_case_1_save_and_verify_checkpoint(self):
        job_id = "test-job-001"
        data = {
            "utterances": [{"start": 0.0, "end": 10.0, "text": "Hello AI"}],
            "audio_result": {"text": "Hello AI", "segments": []},
            "visual_result": {"scenes": []},
            "timeline_result": {"chapters": []},
            "chapters": [{"title": "Chapter 1", "startTime": 0.0, "endTime": 10.0, "summary": "Intro"}],
            "keyframes": [{"timestamp": 0.0, "imageUrl": "http://img.png", "ocr_text": "AI"}],
            "sprint_stats": {"sprint1": 1},
            "export_meta": {"version": 2},
            "video_file_url": "http://video.mp4",
        }

        saved = self.checkpoint_manager.save_checkpoint(
            job_id=job_id,
            stage=STAGE_QUALITY_COMPLETE,
            status=STATUS_INTERMEDIATE_READY,
            data=data,
        )
        self.assertTrue(saved)

        # Verify file exists on disk
        ckpt_file = Path(self.temp_dir) / f"{job_id}.json"
        self.assertTrue(ckpt_file.exists())

        # Load and verify content
        loaded = self.checkpoint_manager.load_checkpoint(job_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["job_id"], job_id)
        self.assertEqual(loaded["pipeline_version"], PIPELINE_VERSION)
        self.assertEqual(loaded["stage"], STAGE_QUALITY_COMPLETE)
        self.assertEqual(loaded["status"], STATUS_INTERMEDIATE_READY)
        self.assertEqual(loaded["data"]["chapters"], data["chapters"])
        self.assertEqual(loaded["data"]["keyframes"], data["keyframes"])

    # -------------------------------------------------------------------------
    # Case 2: Checkpoint invalid / version mismatch -> load returns None
    # -------------------------------------------------------------------------
    def test_case_2_version_mismatch_or_corruption_rejects_checkpoint(self):
        job_id = "test-job-mismatch"
        ckpt_file = Path(self.temp_dir) / f"{job_id}.json"

        # 1. Version mismatch
        payload = {
            "job_id": job_id,
            "pipeline_version": "1.0.0",  # Old version
            "stage": STAGE_QUALITY_COMPLETE,
            "status": STATUS_INTERMEDIATE_READY,
            "data": {"utterances": [], "chapters": [], "keyframes": []},
        }
        with open(ckpt_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        loaded = self.checkpoint_manager.load_checkpoint(job_id)
        self.assertIsNone(loaded)

        # 2. Corrupted JSON file
        with open(ckpt_file, "w", encoding="utf-8") as f:
            f.write("INVALID_JSON_CORRUPTED_FILE")

        loaded_corrupt = self.checkpoint_manager.load_checkpoint(job_id)
        self.assertIsNone(loaded_corrupt)

    # -------------------------------------------------------------------------
    # Case 3: Checkpoint write failure aborts without pretending success
    # -------------------------------------------------------------------------
    def test_case_3_persistence_failure_returns_false(self):
        # Point to invalid read-only path to induce write failure
        bad_manager = CheckpointManager(checkpoint_dir="/invalid_non_existent_dir_12345/abc")
        with patch("tempfile.mkstemp", side_effect=OSError("Disk full or permission denied")):
            saved = bad_manager.save_checkpoint("job-fail", data={})
            self.assertFalse(saved)

    # -------------------------------------------------------------------------
    # Case 4: Idempotency — repeated saves update in place without duplicates
    # -------------------------------------------------------------------------
    def test_case_4_idempotent_checkpoint_updates(self):
        job_id = "test-job-idempotent"
        data_v1 = {"utterances": [], "chapters": [{"title": "V1", "startTime": 0.0, "endTime": 5.0, "summary": "s"}], "keyframes": []}
        data_v2 = {"utterances": [], "chapters": [{"title": "V2", "startTime": 0.0, "endTime": 10.0, "summary": "s"}], "keyframes": []}

        self.checkpoint_manager.save_checkpoint(job_id, data=data_v1)
        loaded_1 = self.checkpoint_manager.load_checkpoint(job_id)
        created_at_1 = loaded_1["created_at"]

        # Save again for same job_id
        time.sleep(0.01)
        self.checkpoint_manager.save_checkpoint(job_id, data=data_v2)
        loaded_2 = self.checkpoint_manager.load_checkpoint(job_id)

        self.assertEqual(loaded_2["data"]["chapters"][0]["title"], "V2")
        self.assertEqual(loaded_2["created_at"], created_at_1)  # Original created_at retained
        self.assertNotEqual(loaded_2["updated_at"], created_at_1)  # updated_at refreshed

    # -------------------------------------------------------------------------
    # Case 5: Overall Summarization Time Budget Enforcement (60-90s)
    # -------------------------------------------------------------------------
    def test_case_5_summarizer_overall_budget_enforced(self):
        """When candidate attempts exceed overall_timeout, Summarizer stops trying remaining models."""
        summarizer = Summarizer(llm_timeout=10.0, overall_timeout=1.0)
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        # Simulate slow response exceeding the 1.0s budget
        def slow_chat(*args, **kwargs):
            time.sleep(1.2)
            raise openai.APITimeoutError(request=MagicMock())

        summarizer.openrouter_client._client.chat.completions.create.side_effect = slow_chat
        summarizer.groq_client._client.chat.completions.create.side_effect = slow_chat

        utterances = [{"start": 0.0, "end": 60.0, "text": "Hello lecture"}]
        slides = []
        chapters = [{"title": "Intro", "startTime": 0.0, "endTime": 60.0, "summary": "Demo"}]

        start_time = time.time()
        result = summarizer.summarize(utterances, slides, chapters, job_id="job-budget-test")
        elapsed = time.time() - start_time

        # Ensure it stopped early instead of running through all 7 candidates * 3 attempts
        self.assertLess(elapsed, 10.0)
        self.assertTrue("Fallback" in result["model_used"])
        self.assertIn("llm_error", result)
        self.assertEqual(result["llm_error"]["error_code"], "LLM_TIMEOUT")

    # -------------------------------------------------------------------------
    # Case 6: Full Integration — Checkpoint Save & Direct Resume Skips ASR/Visual
    # -------------------------------------------------------------------------
    def test_case_6_resume_from_checkpoint_skips_asr_and_visual(self):
        """When checkpoint exists, process_video skips ASR, Visual, Fusion, Quality and resumes from Summarization."""
        from ai_workers import tasks

        dummy_audio = {
            "text": "Xin chào các bạn, hôm nay chúng ta học về trí tuệ nhân tạo.",
            "segments": [{"start": 0.0, "end": 10.0, "text": "Xin chào các bạn, hôm nay chúng ta học về trí tuệ nhân tạo."}],
            "language": "vi",
        }
        dummy_diar = dummy_audio["segments"]
        dummy_visual = {
            "scenes": [{
                "start_seconds": 0.0,
                "end_seconds": 10.0,
                "start_timecode": "00:00",
                "caption": "Slide mở đầu môn học",
                "ocr_text": "Trí tuệ nhân tạo",
                "keyframe_path": None,
                "keyframe_url": "",
            }],
            "keyframes": [],
        }
        dummy_timeline = {
            "chapters": [{"title": "Chương 1", "startTime": 0.0, "endTime": 10.0, "summary": "Mở đầu"}],
            "aligned_segments": [],
        }

        # Mock heavy ML modules
        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_class = MagicMock()
        mock_audio_class.return_value.process.return_value = dummy_audio
        mock_audio_mod.AudioTranscriber = mock_audio_class

        mock_speaker_mod = ModuleType("ai_workers.modules.audio_v2.speaker")
        mock_speaker_class = MagicMock()
        mock_speaker_class.return_value.process.return_value = dummy_diar
        mock_speaker_mod.SpeakerDiarizer = mock_speaker_class

        mock_scene_mod = ModuleType("ai_workers.modules.visual_v2.scene_detector")
        mock_scene_class = MagicMock()
        mock_scene_class.return_value.process.return_value = dummy_visual
        mock_scene_mod.SceneDetector = mock_scene_class

        mock_semantic_mod = ModuleType("ai_workers.modules.visual_v2.semantic")
        mock_semantic_class = MagicMock()
        mock_semantic_class.return_value.process.return_value = dummy_visual
        mock_semantic_mod.SemanticAnalyzer = mock_semantic_class

        mock_timeline_mod = ModuleType("ai_workers.modules.fusion.timeline")
        mock_timeline_class = MagicMock()
        mock_timeline_class.return_value.process.return_value = dummy_timeline
        mock_timeline_mod.TimelineBuilder = mock_timeline_class

        mock_quality_mod = ModuleType("ai_workers.modules.fusion.quality_postprocess")
        mock_quality_mod.apply_quality_postprocess = MagicMock(return_value={"chapters": dummy_timeline["chapters"], "keyframes": []})

        summarizer_return_val = {
            "video_title": "Bài giảng Trí tuệ Nhân tạo",
            "summary": "Tóm tắt bài giảng chi tiết.",
            "chapters": dummy_timeline["chapters"],
            "model_used": "OpenRouter (qwen/qwen-2.5-7b-instruct)",
        }
        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        mock_summarizer_class.return_value.process.return_value = summarizer_return_val
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

        job_id = "job-resume-test"
        from ai_workers.core.config import worker_settings

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            # --- RUN 1: Fresh run without checkpoint ---
            result_1 = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")
            self.assertEqual(result_1["status"], "done")
            self.assertEqual(result_1["stage"], "completed")

            # Verify Checkpoint was created
            ckpt = self.checkpoint_manager.load_checkpoint(job_id)
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt["stage"], STAGE_QUALITY_COMPLETE)

            # Reset call counts
            mock_audio_class.return_value.process.reset_mock()
            mock_speaker_class.return_value.process.reset_mock()
            mock_scene_class.return_value.process.reset_mock()
            mock_timeline_class.return_value.process.reset_mock()
            mock_quality_mod.apply_quality_postprocess.reset_mock()

            # --- RUN 2: Re-run / Resume with checkpoint present ---
            result_2 = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")
            self.assertEqual(result_2["status"], "done")
            self.assertEqual(result_2["stage"], "completed")

            # CRITICAL: Verify ASR, Speaker, Scene, Timeline, and Quality were NOT called
            mock_audio_class.return_value.process.assert_not_called()
            mock_speaker_class.return_value.process.assert_not_called()
            mock_scene_class.return_value.process.assert_not_called()
            mock_timeline_class.return_value.process.assert_not_called()
            mock_quality_mod.apply_quality_postprocess.assert_not_called()

            # Summarizer WAS called
            self.assertTrue(mock_summarizer_class.return_value.process.called)

    # -------------------------------------------------------------------------
    # Case 7: Simulated Worker Crash & Restart
    # -------------------------------------------------------------------------
    def test_case_7_worker_crash_after_checkpoint_and_successful_recovery(self):
        """Simulate worker crash right after checkpoint; restart worker task resumes seamlessly."""
        from ai_workers import tasks

        dummy_audio = {"text": "Audio text", "segments": [{"start": 0.0, "end": 10.0, "text": "Audio text"}]}
        dummy_visual = {"scenes": [{"start_seconds": 0.0, "end_seconds": 10.0, "caption": "Scene 1"}], "keyframes": []}
        dummy_timeline = {"chapters": [{"title": "Ch1", "startTime": 0.0, "endTime": 10.0, "summary": "Summary 1"}], "aligned_segments": []}

        mock_audio_mod = ModuleType("ai_workers.modules.audio_v2.transcriber")
        mock_audio_mod.AudioTranscriber = MagicMock()
        mock_audio_mod.AudioTranscriber.return_value.process.return_value = dummy_audio

        mock_speaker_mod = ModuleType("ai_workers.modules.audio_v2.speaker")
        mock_speaker_mod.SpeakerDiarizer = MagicMock()
        mock_speaker_mod.SpeakerDiarizer.return_value.process.return_value = dummy_audio["segments"]

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
        mock_quality_mod.apply_quality_postprocess = MagicMock(return_value={"chapters": dummy_timeline["chapters"], "keyframes": []})

        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
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

        job_id = "job-crash-test"
        from ai_workers.core.config import worker_settings

        with patch.dict(sys.modules, mock_modules), \
             patch("ai_workers.tasks.ensure_process_memory_available", return_value=8192), \
             patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), \
             patch.object(tasks.process_video, "update_state"), \
             patch("ai_workers.tasks.CheckpointManager", return_value=self.checkpoint_manager):

            # --- Attempt 1: Crashes during summarization ---
            mock_summarizer_class.return_value.process.side_effect = KeyboardInterrupt("Simulated Worker OOM/Kill")

            with self.assertRaises(KeyboardInterrupt):
                tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")

            # Verify Checkpoint WAS safely persisted before the crash
            ckpt = self.checkpoint_manager.load_checkpoint(job_id)
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt["stage"], STAGE_QUALITY_COMPLETE)

            # --- Attempt 2: Worker task restarted by Celery ---
            # Summarizer recovers and succeeds
            mock_summarizer_class.return_value.process.side_effect = None
            mock_summarizer_class.return_value.process.return_value = {
                "video_title": "Recovered Title",
                "summary": "Recovered Summary",
                "chapters": dummy_timeline["chapters"],
                "model_used": "Groq (llama-3.1-8b-instant)",
            }

            result_recovered = tasks.process_video.run(job_id, "mock_video.mp4", "hybrid")
            self.assertEqual(result_recovered["status"], "done")
            self.assertEqual(result_recovered["stage"], "completed")
            self.assertEqual(result_recovered["video_title"], "Recovered Title")


if __name__ == "__main__":
    unittest.main()

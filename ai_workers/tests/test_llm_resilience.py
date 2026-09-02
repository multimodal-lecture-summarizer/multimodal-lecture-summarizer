"""Unit tests for LLM Resilience Layer (Day 1).

Tests Tenacity exponential retry, HTTP timeout enforcement, error classification,
structured logging sanitization, and fallback schemas.
"""

from __future__ import annotations

import io
import json
import logging
import unittest
from unittest.mock import MagicMock, patch
from tenacity import wait_none

import openai

from ai_workers.modules.summarization.errors import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMBaseError,
    LLMNetworkError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseParsingError,
    LLMTimeoutError,
    map_exception_to_llm_error,
)
from ai_workers.modules.summarization.llm_client import (
    LLMClient,
    is_retryable_error,
    sanitize_text,
)
from ai_workers.modules.summarization.summarizer import Summarizer
from ai_workers.modules.fusion.summarizer import Summarizer as FusionSummarizer


def make_mock_response(content_str: str) -> MagicMock:
    """Create mock chat completion response object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content_str
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def make_openai_status_error(status_code: int, message: str = "error") -> openai.APIStatusError:
    """Create mock OpenAI APIStatusError with given status code."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = {}
    return openai.APIStatusError(
        message=message,
        response=mock_response,
        body={"error": {"message": message, "code": status_code}},
    )


class LLMResilienceTests(unittest.TestCase):
    """Test suite covering the 10 required LLM resilience scenarios."""

    def setUp(self):
        self.log_stream = io.StringIO()
        self.log_handler = logging.StreamHandler(self.log_stream)
        self.logger = logging.getLogger("ai_workers.summarization.llm_client")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(self.log_handler)

    def tearDown(self):
        self.logger.removeHandler(self.log_handler)

    # -------------------------------------------------------------------------
    # Case 1: LLM success immediately on first attempt -> no retry
    # -------------------------------------------------------------------------
    def test_case_1_llm_success_first_attempt_no_retry(self):
        client = LLMClient(
            provider="OpenRouter",
            api_key="sk-or-v1-fake-test-key",
            base_url="https://openrouter.ai/api/v1",
            custom_wait=wait_none(),
        )

        mock_resp = make_mock_response('{"video_title": "Test Title", "summary": "Test", "chapters": []}')
        with patch.object(client._client.chat.completions, "create", return_value=mock_resp) as mock_create:
            result = client.generate_chat_completion(
                model="qwen/qwen-2.5-7b-instruct",
                prompt="test prompt",
                job_id="job-c1",
            )
            self.assertEqual(mock_create.call_count, 1)
            self.assertIn("Test Title", result)

        log_output = self.log_stream.getvalue()
        self.assertIn("LLM request succeeded", log_output)
        self.assertIn("job_id=job-c1", log_output)
        self.assertIn("attempt=1", log_output)

    # -------------------------------------------------------------------------
    # Case 2: Timeout on 1st attempt -> retry -> success on 2nd attempt
    # -------------------------------------------------------------------------
    def test_case_2_timeout_first_attempt_retry_success(self):
        client = LLMClient(
            provider="Groq",
            api_key="gsk_fake_groq_key",
            base_url="https://api.groq.com/openai/v1",
            custom_wait=wait_none(),
        )

        timeout_err = openai.APITimeoutError(request=MagicMock())
        mock_success = make_mock_response('{"video_title": "Recovered Title", "summary": "OK", "chapters": []}')

        with patch.object(
            client._client.chat.completions, "create", side_effect=[timeout_err, mock_success]
        ) as mock_create:
            result = client.generate_chat_completion(
                model="llama-3.1-8b-instant",
                prompt="test prompt",
                job_id="job-c2",
            )
            self.assertEqual(mock_create.call_count, 2)
            self.assertIn("Recovered Title", result)

        log_output = self.log_stream.getvalue()
        self.assertIn("error=LLM_TIMEOUT", log_output)
        self.assertIn("attempt=1", log_output)
        self.assertIn("attempt=2", log_output)

    # -------------------------------------------------------------------------
    # Case 3: HTTP 500 on 1st attempt -> retry -> success on 2nd attempt
    # -------------------------------------------------------------------------
    def test_case_3_http_500_first_attempt_retry_success(self):
        client = LLMClient(
            provider="OpenRouter",
            api_key="sk-or-v1-fake-key",
            base_url="https://openrouter.ai/api/v1",
            custom_wait=wait_none(),
        )

        server_err = make_openai_status_error(500, "Internal Server Error")
        mock_success = make_mock_response('{"video_title": "Server Recovered", "summary": "OK", "chapters": []}')

        with patch.object(
            client._client.chat.completions, "create", side_effect=[server_err, mock_success]
        ) as mock_create:
            result = client.generate_chat_completion(
                model="qwen/qwen-2.5-7b-instruct",
                prompt="test prompt",
                job_id="job-c3",
            )
            self.assertEqual(mock_create.call_count, 2)
            self.assertIn("Server Recovered", result)

        log_output = self.log_stream.getvalue()
        self.assertIn("error=LLM_PROVIDER_ERROR", log_output)

    # -------------------------------------------------------------------------
    # Case 4: HTTP 429 -> retry
    # -------------------------------------------------------------------------
    def test_case_4_http_429_retry(self):
        client = LLMClient(
            provider="Groq",
            api_key="gsk_fake_groq_key",
            base_url="https://api.groq.com/openai/v1",
            custom_wait=wait_none(),
        )

        rate_limit_err = make_openai_status_error(429, "Rate limit reached")
        mock_success = make_mock_response('{"video_title": "Rate Limit Handled", "summary": "OK", "chapters": []}')

        with patch.object(
            client._client.chat.completions, "create", side_effect=[rate_limit_err, mock_success]
        ) as mock_create:
            result = client.generate_chat_completion(
                model="llama-3.1-8b-instant",
                prompt="test prompt",
                job_id="job-c4",
            )
            self.assertEqual(mock_create.call_count, 2)
            self.assertIn("Rate Limit Handled", result)

        log_output = self.log_stream.getvalue()
        self.assertIn("error=LLM_RATE_LIMIT", log_output)

    # -------------------------------------------------------------------------
    # Case 5: Timeout all 3 attempts -> raises structured LLM failure
    # -------------------------------------------------------------------------
    def test_case_5_timeout_all_3_attempts_structured_failure(self):
        client = LLMClient(
            provider="OpenRouter",
            api_key="sk-or-v1-fake-key",
            base_url="https://openrouter.ai/api/v1",
            max_attempts=3,
            custom_wait=wait_none(),
        )

        timeout_err = openai.APITimeoutError(request=MagicMock())

        with patch.object(client._client.chat.completions, "create", side_effect=timeout_err) as mock_create:
            with self.assertRaises(LLMTimeoutError) as ctx:
                client.generate_chat_completion(
                    model="qwen/qwen-2.5-7b-instruct",
                    prompt="test prompt",
                    job_id="job-c5",
                )
            self.assertEqual(mock_create.call_count, 3)
            self.assertEqual(ctx.exception.error_code, "LLM_TIMEOUT")
            self.assertTrue(ctx.exception.is_retryable)

    # -------------------------------------------------------------------------
    # Case 6: HTTP 401 -> fails fast, does NOT retry 3 times
    # -------------------------------------------------------------------------
    def test_case_6_http_401_no_retry(self):
        client = LLMClient(
            provider="OpenRouter",
            api_key="sk-or-v1-invalid-key",
            base_url="https://openrouter.ai/api/v1",
            max_attempts=3,
            custom_wait=wait_none(),
        )

        auth_err = make_openai_status_error(401, "Invalid API key")

        with patch.object(client._client.chat.completions, "create", side_effect=auth_err) as mock_create:
            with self.assertRaises(LLMAuthenticationError) as ctx:
                client.generate_chat_completion(
                    model="qwen/qwen-2.5-7b-instruct",
                    prompt="test prompt",
                    job_id="job-c6",
                )
            # CRITICAL: Attempt count must be exactly 1 (no retries for auth errors)
            self.assertEqual(mock_create.call_count, 1)
            self.assertEqual(ctx.exception.error_code, "LLM_AUTH_ERROR")
            self.assertFalse(ctx.exception.is_retryable)

    # -------------------------------------------------------------------------
    # Case 7: HTTP 400 -> fails fast, does NOT retry 3 times
    # -------------------------------------------------------------------------
    def test_case_7_http_400_no_retry(self):
        client = LLMClient(
            provider="Groq",
            api_key="gsk_fake_groq_key",
            base_url="https://api.groq.com/openai/v1",
            max_attempts=3,
            custom_wait=wait_none(),
        )

        bad_req_err = make_openai_status_error(400, "Invalid model parameter")

        with patch.object(client._client.chat.completions, "create", side_effect=bad_req_err) as mock_create:
            with self.assertRaises(LLMBadRequestError) as ctx:
                client.generate_chat_completion(
                    model="invalid-model",
                    prompt="test prompt",
                    job_id="job-c7",
                )
            # CRITICAL: Attempt count must be exactly 1 (no retries for 400 bad request)
            self.assertEqual(mock_create.call_count, 1)
            self.assertEqual(ctx.exception.error_code, "LLM_BAD_REQUEST")
            self.assertFalse(ctx.exception.is_retryable)

    # -------------------------------------------------------------------------
    # Case 8: Connection error -> retry
    # -------------------------------------------------------------------------
    def test_case_8_connection_error_retry(self):
        client = LLMClient(
            provider="Groq",
            api_key="gsk_fake_groq_key",
            base_url="https://api.groq.com/openai/v1",
            custom_wait=wait_none(),
        )

        conn_err = openai.APIConnectionError(request=MagicMock())
        mock_success = make_mock_response('{"video_title": "Network Recovered", "summary": "OK", "chapters": []}')

        with patch.object(
            client._client.chat.completions, "create", side_effect=[conn_err, mock_success]
        ) as mock_create:
            result = client.generate_chat_completion(
                model="llama-3.1-8b-instant",
                prompt="test prompt",
                job_id="job-c8",
            )
            self.assertEqual(mock_create.call_count, 2)
            self.assertIn("Network Recovered", result)

        log_output = self.log_stream.getvalue()
        self.assertIn("error=LLM_NETWORK_ERROR", log_output)

    # -------------------------------------------------------------------------
    # Case 9: Verify configured timeout is passed to underlying OpenAI client
    # -------------------------------------------------------------------------
    def test_case_9_timeout_configuration_enforced(self):
        custom_timeout = 18.5
        client = LLMClient(
            provider="OpenRouter",
            api_key="sk-or-v1-fake-key",
            base_url="https://openrouter.ai/api/v1",
            timeout=custom_timeout,
        )
        self.assertEqual(client.timeout, 18.5)
        self.assertEqual(client._client.timeout, 18.5)

    # -------------------------------------------------------------------------
    # Case 10: Verify API key / Authorization token NEVER appear in logs
    # -------------------------------------------------------------------------
    def test_case_10_api_key_sanitization_in_logs(self):
        secret_or_key = "sk-or-v1-abcdef1234567890secretkeyvalue"
        secret_groq_key = "gsk_abcdef1234567890secretkeyvalue"

        # Direct test on sanitize_text function
        raw_text_1 = f"Calling OpenRouter with key {secret_or_key} and Authorization: Bearer {secret_or_key}"
        sanitized_1 = sanitize_text(raw_text_1)
        self.assertNotIn(secret_or_key, sanitized_1)
        self.assertIn("[REDACTED]", sanitized_1)

        raw_text_2 = f"Groq error with key {secret_groq_key}"
        sanitized_2 = sanitize_text(raw_text_2)
        self.assertNotIn(secret_groq_key, sanitized_2)

        # Verify client execution does not leak key to log stream
        client = LLMClient(
            provider="OpenRouter",
            api_key=secret_or_key,
            base_url="https://openrouter.ai/api/v1",
            max_attempts=1,
            custom_wait=wait_none(),
        )

        auth_err = make_openai_status_error(401, f"Invalid token {secret_or_key}")
        with patch.object(client._client.chat.completions, "create", side_effect=auth_err):
            try:
                client.generate_chat_completion(
                    model="qwen/qwen-2.5-7b-instruct",
                    prompt="test prompt",
                    job_id="job-c10",
                )
            except LLMAuthenticationError:
                pass

        log_output = self.log_stream.getvalue()
        self.assertNotIn(secret_or_key, log_output)

    # -------------------------------------------------------------------------
    # Summarizer Service Fallback & Backward Compatibility Tests
    # -------------------------------------------------------------------------
    def test_summarizer_full_fallback_on_all_failures(self):
        """When all LLM targets fail, summarizer returns structured fallback without crashing."""
        summarizer = Summarizer(custom_wait=wait_none())
        summarizer.openrouter_client._client = MagicMock()
        summarizer.groq_client._client = MagicMock()

        timeout_err = openai.APITimeoutError(request=MagicMock())
        summarizer.openrouter_client._client.chat.completions.create.side_effect = timeout_err
        summarizer.groq_client._client.chat.completions.create.side_effect = timeout_err

        utterances = [{"start": 0.0, "end": 60.0, "text": "Hello world"}]
        slides = [{"start_seconds": 0.0, "end_seconds": 60.0, "caption": "Intro slide", "ocr_text": "AI"}]
        chapters = [{"title": "Introduction", "startTime": 0.0, "endTime": 60.0, "summary": "Intro"}]

        result = summarizer.summarize(utterances, slides, chapters, job_id="job-fallback")

        self.assertIn("video_title", result)
        self.assertIn("summary", result)
        self.assertIn("chapters", result)
        self.assertTrue("Fallback" in result["model_used"])
        self.assertIn("llm_error", result)
        self.assertEqual(result["llm_error"]["error_code"], "LLM_TIMEOUT")

    def test_summarizer_success_with_chapters(self):
        """When LLM returns valid schema, summarizer parses and formats chapters correctly."""
        summarizer = Summarizer(custom_wait=wait_none())
        summarizer.openrouter_client._client = MagicMock()

        json_response = json.dumps({
            "analysis": "Analysis CoT...",
            "video_title": "Bài giảng Trí tuệ Nhân tạo",
            "summary": "Tóm tắt bài giảng chi tiết về AI.",
            "chapters": [
                {
                    "title": "Giới thiệu AI",
                    "startTime": 0.0,
                    "endTime": 60.0,
                    "summary": "Tổng quan AI và ứng dụng.",
                }
            ],
        })

        mock_resp = make_mock_response(json_response)
        summarizer.openrouter_client._client.chat.completions.create.return_value = mock_resp

        utterances = [{"start": 0.0, "end": 60.0, "text": "Hello AI world"}]
        slides = []
        chapters = [{"title": "Chương 1", "startTime": 0.0, "endTime": 60.0, "summary": "Demo"}]

        result = summarizer.summarize(utterances, slides, chapters, job_id="job-success")

        self.assertEqual(result["video_title"], "Bài giảng Trí tuệ Nhân tạo")
        self.assertEqual(result["summary"], "Tóm tắt bài giảng chi tiết về AI.")
        self.assertEqual(len(result["chapters"]), 1)
        self.assertEqual(result["chapters"][0]["title"], "Giới thiệu AI")

    def test_fusion_summarizer_backward_compatibility(self):
        """Verify ai_workers.modules.fusion.summarizer.Summarizer is identical to new Summarizer."""
        self.assertIs(FusionSummarizer, Summarizer)

    # -------------------------------------------------------------------------
    # Integration Test: process_video Celery task does not crash on LLM failure
    # -------------------------------------------------------------------------
    def test_process_video_integration_llm_failure_does_not_crash_task(self):
        """Integration test: when LLM fails, process_video completes with status='done' and fallback summary."""
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

        import sys
        from types import ModuleType

        # Create mock module stubs so test is 100% pure orchestration and decoupled from GPU/PyTorch/Transformers
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

        fallback_llm_result = {
            "video_title": "Bài giảng chưa đặt tên (Lỗi AI)",
            "summary": "### Tóm tắt bài giảng\n\nNội dung bài giảng đã trích xuất thành công. (Gặp lỗi khi tạo tóm tắt bằng AI: [LLM_TIMEOUT] Request timed out)",
            "chapters": dummy_timeline["chapters"],
            "model_used": "Offline Fallback (Error: LLM_TIMEOUT)",
            "llm_error": {"error_code": "LLM_TIMEOUT", "message": "Request timed out", "is_retryable": True},
        }

        mock_summarizer_mod = ModuleType("ai_workers.modules.fusion.summarizer")
        mock_summarizer_class = MagicMock()
        mock_summarizer_class.return_value.process.return_value = fallback_llm_result
        mock_summarizer_mod.Summarizer = mock_summarizer_class

        mock_quality_mod = ModuleType("ai_workers.modules.fusion.quality_postprocess")
        mock_quality_mod.apply_quality_postprocess = MagicMock(return_value={"chapters": dummy_timeline["chapters"], "keyframes": []})

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
             patch.object(tasks.process_video, "update_state"):

            # Execute full process_video pipeline
            result = tasks.process_video.run(
                "job-integration-fail",
                "mock_input_video.mp4",
                "hybrid",
            )

            # Assert task does not crash and returns structured result
            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "done")
            self.assertEqual(result["stage"], "completed")
            self.assertEqual(result["progress"], 100)
            self.assertIn("Offline Fallback", result["model_used"])
            self.assertIn("summary", result)
            self.assertIn("chapters", result)
            self.assertEqual(len(result["chapters"]), 1)


if __name__ == "__main__":
    unittest.main()

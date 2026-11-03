import threading
import time
from unittest.mock import patch

import numpy as np

from ai_workers.core import model_prefetch
from ai_workers.modules.audio_v2.qwen_transcriber import SAMPLE_RATE, QwenEnglishTranscriber


def test_prefetch_auto_follows_colab_environment(monkeypatch):
    monkeypatch.setattr(model_prefetch.worker_settings, "PREFETCH_MODELS", "auto")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    assert model_prefetch.prefetch_enabled() is False
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    assert model_prefetch.prefetch_enabled() is True


def test_prefetch_explicit_flag_overrides_environment(monkeypatch):
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-colab")
    monkeypatch.setattr(model_prefetch.worker_settings, "PREFETCH_MODELS", "false")
    assert model_prefetch.prefetch_enabled() is False
    monkeypatch.setattr(model_prefetch.worker_settings, "PREFETCH_MODELS", "true")
    assert model_prefetch.prefetch_enabled() is True


def test_start_prefetch_thread_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(model_prefetch.worker_settings, "PREFETCH_MODELS", "false")
    assert model_prefetch.start_prefetch_thread() is None


def test_failed_prefetch_task_does_not_stop_the_others():
    def boom():
        raise RuntimeError("network down")

    done = []
    tasks = [("bad", boom), ("good", lambda: done.append(1))]
    logs = []
    with patch.object(model_prefetch, "prefetch_tasks", return_value=tasks):
        outcome = model_prefetch.run_prefetch(logs.append)

    assert outcome == {"bad": False, "good": True}
    assert done == [1]
    assert any("bad skipped" in line for line in logs)


def test_task_list_follows_backend_settings(monkeypatch):
    monkeypatch.setattr(model_prefetch.worker_settings, "ASR_ENGLISH_BACKEND", "whisper")
    monkeypatch.setattr(model_prefetch.worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "llm")
    names = [name for name, _ in model_prefetch.prefetch_tasks()]
    assert "qwen3-asr" not in names and "led" not in names
    assert {"whisper", "clip", "paddleocr"} <= set(names)


def test_denoise_and_model_load_overlap():
    both_running = threading.Barrier(2, timeout=5)

    def fake_denoise(self, _path):
        both_running.wait()  # only passes when the model is loading at the same time
        return np.zeros(SAMPLE_RATE, dtype=np.float32)

    def fake_load(self):
        both_running.wait()
        time.sleep(0.01)
        return object()

    with patch.object(QwenEnglishTranscriber, "reduce_noise", fake_denoise), \
         patch.object(QwenEnglishTranscriber, "_load_model", fake_load), \
         patch("ai_workers.modules.audio_v2.qwen_transcriber.split_audio_chunks", return_value=[]), \
         patch("ai_workers.modules.audio_v2.qwen_transcriber.validate_words"):
        result = QwenEnglishTranscriber()._transcribe_qwen("x.wav")

    assert result["language"] == "en"

import threading
from types import SimpleNamespace
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


def test_hub_downloads_never_overlap():
    import time

    active, overlaps = [], []

    def fake_snapshot_download(repo_id, **kwargs):
        active.append(repo_id)
        if len(active) > 1:
            overlaps.append(repo_id)
        time.sleep(0.03)
        active.remove(repo_id)

    threads = [threading.Thread(target=model_prefetch._snapshot, args=(f"repo/{i}",)) for i in range(4)]
    with patch("huggingface_hub.snapshot_download", fake_snapshot_download):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert overlaps == []


def test_task_list_follows_backend_settings(monkeypatch):
    monkeypatch.setattr(model_prefetch.worker_settings, "ASR_ENGLISH_BACKEND", "whisper")
    monkeypatch.setattr(model_prefetch.worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "llm")
    names = [name for name, _ in model_prefetch.prefetch_tasks()]
    assert "qwen3-asr" not in names and "led" not in names
    assert {"whisper", "clip", "paddleocr"} <= set(names)


def _empty_result():
    return SimpleNamespace(text="", time_stamps=[])


def test_denoise_and_model_load_overlap():
    both_running = threading.Barrier(2, timeout=5)
    one_second = np.zeros(SAMPLE_RATE, dtype=np.float32)

    def fake_denoise(self, samples):
        both_running.wait()  # only passes when the model is loading at the same time
        return samples

    def fake_load(self):
        both_running.wait()
        return SimpleNamespace(transcribe=lambda **kw: [_empty_result()])

    with patch.object(QwenEnglishTranscriber, "load_samples", return_value=one_second), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", fake_denoise), \
         patch.object(QwenEnglishTranscriber, "remember_denoised"), \
         patch.object(QwenEnglishTranscriber, "_load_model", fake_load), \
         patch("ai_workers.modules.audio_v2.qwen_transcriber.split_audio_chunks", return_value=[(0.0, one_second)]):
        result = QwenEnglishTranscriber()._transcribe_qwen("x.wav")

    assert result["language"] == "en"


def test_asr_starts_before_last_chunk_is_denoised():
    chunk = np.zeros(SAMPLE_RATE, dtype=np.float32)
    chunks = [(float(i) * 240, chunk) for i in range(4)]
    first_batch_transcribed = threading.Event()
    denoised = []

    def fake_denoise(self, samples):
        if len(denoised) == 3:  # the last chunk waits until ASR has already run on the first batch
            assert first_batch_transcribed.wait(timeout=5)
        denoised.append(1)
        return samples

    def fake_transcribe(**kw):
        first_batch_transcribed.set()
        audio = kw["audio"]
        return [_empty_result() for _ in (audio if isinstance(audio, list) else [audio])]

    with patch.object(QwenEnglishTranscriber, "load_samples", return_value=chunk), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", fake_denoise), \
         patch.object(QwenEnglishTranscriber, "remember_denoised") as remember, \
         patch.object(QwenEnglishTranscriber, "_load_model", lambda self: SimpleNamespace(transcribe=fake_transcribe)), \
         patch("ai_workers.modules.audio_v2.qwen_transcriber.worker_settings.QWEN_ASR_BATCH_SIZE", 4), \
         patch("ai_workers.modules.audio_v2.qwen_transcriber.split_audio_chunks", return_value=chunks):
        result = QwenEnglishTranscriber()._transcribe_qwen("x.wav")

    assert result["language"] == "en"
    assert len(denoised) == 4
    remember.assert_called_once()


def test_pipeline_batch_size_always_leaves_two_batches():
    from ai_workers.modules.audio_v2.qwen_transcriber import pipeline_batch_size

    with patch("ai_workers.modules.audio_v2.qwen_transcriber.worker_settings.QWEN_ASR_BATCH_SIZE", 4):
        assert pipeline_batch_size(4) == 2
        assert pipeline_batch_size(7) == 4
        assert pipeline_batch_size(1) == 1
    with patch("ai_workers.modules.audio_v2.qwen_transcriber.worker_settings.QWEN_ASR_BATCH_SIZE", 1):
        assert pipeline_batch_size(4) == 1

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ai_workers.core.config import flag_enabled
from ai_workers.modules.audio_v2 import qwen_transcriber as qt
from ai_workers.modules.audio_v2.qwen_transcriber import QwenEnglishTranscriber
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber


@pytest.fixture(autouse=True)
def _clean_resident_model():
    qt.reset_resident_model()
    yield
    qt.reset_resident_model()


def _transcriber(device="cuda"):
    with patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", device)):
        return QwenEnglishTranscriber()


def test_resident_model_is_loaded_once_and_reused():
    loads = []
    with patch.object(qt, "resident_enabled", return_value=True), \
         patch.object(QwenEnglishTranscriber, "_load_model", lambda self: loads.append(1) or object()):
        t = _transcriber()
        first, second = t._acquire_model(), t._acquire_model()

    assert first is second
    assert len(loads) == 1


def test_model_is_loaded_per_job_when_residency_is_off():
    loads = []
    with patch.object(qt, "resident_enabled", return_value=False), \
         patch.object(QwenEnglishTranscriber, "_load_model", lambda self: loads.append(1) or object()):
        t = _transcriber()
        t._acquire_model()
        t._acquire_model()

    assert len(loads) == 2


def test_cpu_model_is_never_kept():
    loads = []
    with patch.object(qt, "resident_enabled", return_value=True), \
         patch.object(QwenEnglishTranscriber, "_load_model", lambda self: loads.append(1) or object()):
        t = _transcriber(device="cpu")
        t._acquire_model()
        t._acquire_model()

    assert len(loads) == 2


def test_job_waits_for_the_preload_instead_of_loading_twice():
    loads = []

    def slow_load(self):
        loads.append(1)
        time.sleep(0.1)
        return object()

    results = []
    with patch.object(qt, "resident_enabled", return_value=True), \
         patch.object(QwenEnglishTranscriber, "_load_model", slow_load):
        threads = [threading.Thread(target=lambda: results.append(_transcriber()._acquire_model())) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    assert len(loads) == 1
    assert all(r is results[0] for r in results)


def test_failed_transcription_discards_the_kept_model():
    one_second = np.zeros(qt.SAMPLE_RATE, dtype=np.float32)

    def broken(**kw):
        raise ValueError("model state is bad")

    with patch.object(qt, "resident_enabled", return_value=True), \
         patch.object(QwenEnglishTranscriber, "_load_model", lambda self: SimpleNamespace(transcribe=broken)), \
         patch.object(QwenEnglishTranscriber, "load_samples", return_value=one_second), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", side_effect=lambda s: s), \
         patch.object(QwenEnglishTranscriber, "remember_denoised"), \
         patch.object(qt, "split_audio_chunks", return_value=[(0.0, one_second)]):
        with pytest.raises(ValueError):
            _transcriber()._transcribe_qwen("x.wav")

    assert qt._resident_model is None


def test_flag_enabled_values(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    assert flag_enabled("auto") is False
    assert flag_enabled(None) is False
    monkeypatch.setenv("COLAB_GPU", "1")
    assert flag_enabled("auto") is True
    assert flag_enabled("false") is False
    assert flag_enabled("TRUE") is True

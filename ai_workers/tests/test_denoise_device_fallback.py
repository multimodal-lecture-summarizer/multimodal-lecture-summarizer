import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# denoise imports audio libs that are only installed on the GPU worker; the fallback logic
# under test never touches them. Stub them only while importing so other tests are unaffected.
# Only the two stub keys are removed afterwards; restoring all of sys.modules would drop
# unrelated modules imported meanwhile (e.g. cv2) and break later imports.
_stubs = {name: MagicMock() for name in ("torchaudio", "librosa") if name not in sys.modules}
sys.modules.update(_stubs)
try:
    from ai_workers.modules.common import denoise
finally:
    for _name in _stubs:
        sys.modules.pop(_name, None)


class _Model:
    def __init__(self):
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


def _manager(device):
    manager = denoise.DenoiseManager()
    manager.model = _Model().to(device)
    manager.df_state = SimpleNamespace(sr=lambda: 16000)
    manager.device = device
    return manager


def test_cuda_failure_retries_on_cpu_instead_of_raw_audio():
    seen_devices = []

    def fake_enhance(model, _state, audio):
        seen_devices.append(model.device)
        if model.device == "cuda":
            raise RuntimeError("cuDNN error: CUDNN_STATUS_NOT_SUPPORTED")
        return audio * 0.5

    manager = _manager("cuda")
    with patch.object(denoise, "DF_AVAILABLE", True), patch.object(denoise, "_gpu_failed", False), \
         patch.object(denoise, "enhance", fake_enhance, create=True), \
         patch.object(denoise, "load_audio", lambda *_a, **_k: (torch.ones(1, 8), 16000), create=True), \
         patch.object(manager, "initialize"), patch.object(manager, "_read_raw_audio") as raw, \
         patch("torch.cuda.empty_cache"):
        out = manager.enhance_audio("x.wav")

    assert seen_devices == ["cuda", "cpu"]
    assert manager.device == "cpu"
    raw.assert_not_called()
    assert np.allclose(out, 0.5)


def test_same_file_is_denoised_once(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x" * 16)
    calls = []

    def fake_enhance(model, _state, audio):
        calls.append(1)
        return audio * 0.5

    manager = _manager("cpu")
    with patch.object(denoise, "DF_AVAILABLE", True), \
         patch.object(denoise, "enhance", fake_enhance, create=True), \
         patch.object(denoise, "load_audio", lambda *_a, **_k: (torch.ones(1, 8), 16000), create=True), \
         patch.object(manager, "initialize"):
        first = manager.enhance_audio(str(wav))
        second = manager.enhance_audio(str(wav))

    assert len(calls) == 1
    assert first is second


def _timed_chunk(manager, seconds_elapsed):
    samples = np.zeros(16000 * 10, dtype=np.float32)  # a 10 s chunk
    with patch.object(denoise, "DF_AVAILABLE", True), patch.object(denoise, "_gpu_failed", False), \
         patch.object(manager, "initialize"), \
         patch.object(manager, "_enhance_samples", return_value=samples), \
         patch.object(denoise.time, "perf_counter", side_effect=[0.0, seconds_elapsed]), \
         patch("torch.cuda.empty_cache"):
        manager.enhance_array(samples)


def test_gpu_that_is_slower_than_cpu_is_dropped():
    manager = _manager("cuda")
    _timed_chunk(manager, 5.0)  # 0.5x real time
    assert manager.device == "cpu"


def test_fast_gpu_is_kept():
    manager = _manager("cuda")
    _timed_chunk(manager, 0.3)  # 0.03x real time
    assert manager.device == "cuda"


def test_gpu_failure_flag_forces_cpu_device():
    with patch.object(denoise, "_gpu_failed", True):
        assert denoise._denoise_device() == "cpu"

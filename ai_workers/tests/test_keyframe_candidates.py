import numpy as np
import pytest

from ai_workers.modules.visual_v2 import scene_detector as sd


class _FakeCapture:
    """Counts the seeks, which are what make keyframe extraction slow on a real video."""

    def __init__(self, seeks):
        self.seeks = seeks

    def isOpened(self):
        return True

    def set(self, _prop, value):
        self.seeks.append(value)
        return True

    def read(self):
        return True, np.full((8, 8, 3), self.seeks[-1] % 255, dtype=np.uint8)

    def release(self):
        pass


def _extract(monkeypatch, tmp_path, candidates, scene_frames=1000):
    seeks = []
    monkeypatch.chdir(tmp_path)  # keyframes are written under the working directory
    monkeypatch.setattr(sd.cv2, "VideoCapture", lambda _path: _FakeCapture(seeks))
    monkeypatch.setattr(sd.cv2, "imwrite", lambda *_a, **_k: True)
    monkeypatch.setattr(sd.worker_settings, "KEYFRAME_CANDIDATES", candidates)
    scenes = [{"scene_index": 0, "start_frame": 0, "end_frame": scene_frames - 1}]
    sd.SceneDetector().extract_keyframes("video.mp4", scenes, str(tmp_path / "job"))
    return seeks, scenes[0]


@pytest.mark.parametrize("candidates", [7, 3])
def test_number_of_seeks_follows_the_setting(monkeypatch, tmp_path, candidates):
    seeks, scene = _extract(monkeypatch, tmp_path, candidates)
    assert len(seeks) == candidates
    assert seeks == sorted(seeks)  # forward-only, no wasted back-seeks
    assert scene["keyframe_path"]


def test_tiny_scene_uses_a_single_middle_frame(monkeypatch, tmp_path):
    seeks, _scene = _extract(monkeypatch, tmp_path, candidates=7, scene_frames=2)
    assert len(seeks) == 1

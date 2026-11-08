import numpy as np
import pytest

from ai_workers.core import model_prefetch
from ai_workers.modules.visual_v2 import object_tracker
from ai_workers.modules.visual_v2.resident_models import DETECTOR_SLOT


class FakeDetector:
    """Stands in for RtDetrDetector: a large 'person' box that drifts right on every call."""

    built = 0

    def __init__(self, repo="fake/repo", device="cpu"):
        FakeDetector.built += 1
        self.repo, self.device = repo, device
        self.calls = 0
        self.cpu_fallbacks = 0

    def label(self, class_id):
        return {0: "person", 2: "car"}.get(int(class_id), str(class_id))

    def to_cpu(self):
        self.cpu_fallbacks += 1
        self.device = "cpu"

    def detect(self, bgr, min_score):
        self.calls += 1
        h, w = bgr.shape[:2]
        shift = 4 * self.calls
        xyxy = np.array([[0.2 * w + shift, 0.2 * h, 0.6 * w + shift, 0.8 * h]], dtype=float)
        return xyxy, np.array([0.9]), np.array([0])


@pytest.fixture
def fake_detector(monkeypatch):
    FakeDetector.built = 0
    DETECTOR_SLOT.clear()
    monkeypatch.setattr(object_tracker, "build_detector", lambda repo, device: FakeDetector(repo, device))
    monkeypatch.setattr(object_tracker, "warm_up", lambda detector, min_score: None)
    monkeypatch.setattr(object_tracker.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(object_tracker.worker_settings, "ENABLE_OBJECT_TRACKING", True)
    yield FakeDetector
    DETECTOR_SLOT.clear()


def test_object_tracking_is_on_by_default():
    from ai_workers.core.config import worker_settings

    fields = type(worker_settings).model_fields
    assert fields["ENABLE_OBJECT_TRACKING"].default is True


def test_models_are_the_benchmark_selection():
    from ai_workers.core.config import worker_settings

    fields = type(worker_settings).model_fields
    assert fields["OBJECT_DETECTOR_MODEL"].default == "PekingU/rtdetr_r18vd"
    # Published BoT-SORT defaults the benchmark scored (trackers==2.6.1), CMC on, no ReID.
    assert object_tracker.BOTSORT_PARAMS["track_activation_threshold"] == 0.7
    assert object_tracker.BOTSORT_PARAMS["high_conf_det_threshold"] == 0.6
    assert object_tracker.BOTSORT_PARAMS["lost_track_buffer"] == 30
    assert object_tracker.BOTSORT_PARAMS["enable_cmc"] is True
    assert object_tracker.BOTSORT_PARAMS["cmc_method"] == "sparseOptFlow"


def test_botsort_uses_the_sampling_rate_as_frame_rate(monkeypatch):
    import trackers

    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(trackers, "BoTSORTTracker", Recorder)
    object_tracker.build_botsort(3)

    assert seen["frame_rate"] == 3.0
    assert seen["lost_track_buffer"] == 30 and seen["track_activation_threshold"] == 0.7


def test_missing_tracker_package_fails_once_with_install_hint(fake_detector, monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def no_trackers(name, *args, **kwargs):
        if name == "trackers":
            raise ImportError("No module named 'trackers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trackers)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    tracker = object_tracker.ObjectDetectorTracker()

    with pytest.raises(RuntimeError, match="requirements-tracking.txt"):
        tracker.track_video_scenes(str(video), [{"start_frame": 0, "end_frame": 1}])


def test_botsort_is_the_trackers_library_class():
    tracker = object_tracker.build_botsort(3)
    assert type(tracker).__name__ == "BoTSORTTracker"


def test_resident_detector_is_built_once(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker, "visual_resident_enabled", lambda: True)

    first = object_tracker.ObjectDetectorTracker()
    second = object_tracker.ObjectDetectorTracker()

    assert fake_detector.built == 1
    assert first.model is second.model


def test_non_resident_detector_is_rebuilt_per_tracker(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker, "visual_resident_enabled", lambda: False)

    object_tracker.ObjectDetectorTracker()
    object_tracker.ObjectDetectorTracker()

    assert fake_detector.built == 2


def test_preload_keeps_the_model_for_the_job(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker, "visual_resident_enabled", lambda: True)

    object_tracker.preload_detector_model()
    object_tracker.ObjectDetectorTracker()

    assert fake_detector.built == 1
    assert DETECTOR_SLOT.loaded


def test_preload_without_residency_only_downloads(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker, "visual_resident_enabled", lambda: False)

    object_tracker.preload_detector_model()

    assert fake_detector.built == 1
    assert not DETECTOR_SLOT.loaded


def test_preload_is_noop_when_tracking_disabled(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker.worker_settings, "ENABLE_OBJECT_TRACKING", False)

    object_tracker.preload_detector_model()

    assert fake_detector.built == 0


def test_cuda_failure_moves_the_detector_to_cpu_and_retries(fake_detector, monkeypatch):
    monkeypatch.setattr(object_tracker, "visual_resident_enabled", lambda: False)
    tracker = object_tracker.ObjectDetectorTracker()
    tracker.model.device = "cuda"
    real_detect = tracker.model.detect
    state = {"failed": False}

    def flaky(bgr, min_score):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("CUDA out of memory")
        return real_detect(bgr, min_score)

    tracker.model.detect = flaky
    xyxy, scores, _ = tracker._detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert len(scores) == 1
    assert tracker.model.cpu_fallbacks == 1
    assert tracker.device == "cpu"


def test_prefetch_lists_detector_only_when_tracking_enabled(monkeypatch):
    monkeypatch.setattr(model_prefetch.worker_settings, "ENABLE_OBJECT_TRACKING", True)
    assert "detector" in [name for name, _ in model_prefetch.prefetch_tasks()]
    monkeypatch.setattr(model_prefetch.worker_settings, "ENABLE_OBJECT_TRACKING", False)
    assert "detector" not in [name for name, _ in model_prefetch.prefetch_tasks()]


def test_detector_prefetch_downloads_weights_then_preloads(monkeypatch):
    from unittest.mock import patch

    seen = {}
    with patch.object(model_prefetch, "_snapshot", lambda repo, **kw: seen.update(repo=repo)), \
         patch("ai_workers.modules.visual_v2.object_tracker.preload_detector_model") as preload:
        model_prefetch._prefetch_detector()

    assert seen["repo"] == "PekingU/rtdetr_r18vd"
    preload.assert_called_once()

"""Slide-element stage: label mapping, box normalisation, CUDA fallback, scene wiring, config gating."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ai_workers.core import model_prefetch
from ai_workers.modules.visual_v2 import slide_element_detector as sed


class _Arr:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Arr(xyxy), _Arr(conf), _Arr(cls)

    def __len__(self):
        return len(self.conf.numpy())


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYolo:
    # Same names (and order) as the LVVO GT categories the fine-tune wrote into the checkpoint.
    names = {0: "Table", 1: "Chart-Graph", 2: "Photographic-image", 3: "Visual-illustration"}

    def __init__(self, boxes, fail_on_cuda=False):
        self.boxes = boxes
        self.fail_on_cuda = fail_on_cuda
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        if self.fail_on_cuda and kwargs["device"] == "cuda":
            raise RuntimeError("CUDA out of memory")
        return [_Result(self.boxes)]


def make_detector(model, device="cpu"):
    det = object.__new__(sed.SlideElementDetector)
    det.weights, det.device, det.model = "selected.pt", device, model
    det.names = dict(model.names)
    return det


TWO_BOXES = _Boxes([[0, 0, 50, 25], [10, 20, 200, 120]], [0.4, 0.9], [1, 0])


def test_detect_maps_labels_normalises_and_sorts():
    model = FakeYolo(TWO_BOXES)
    out = make_detector(model).detect(np.zeros((100, 100, 3), dtype=np.uint8), 0.25)

    assert [e["label"] for e in out] == ["table", "chart"]
    assert out[0]["confidence"] == 0.9
    assert out[0]["bbox_norm"] == [0.1, 0.2, 1.0, 1.0]  # clipped to the frame
    assert out[1]["bbox_norm"] == [0.0, 0.0, 0.5, 0.25]
    # Benchmark inference settings, threshold passed through.
    assert model.calls[0] == dict(imgsz=640, conf=0.25, iou=0.7, max_det=100, half=False, device="cpu", verbose=False)


def test_detect_handles_no_boxes():
    out = make_detector(FakeYolo(_Boxes(np.zeros((0, 4)), [], []))).detect(np.zeros((8, 8, 3), np.uint8), 0.25)
    assert out == []


def test_cuda_failure_falls_back_to_cpu_once():
    model = FakeYolo(TWO_BOXES, fail_on_cuda=True)
    det = make_detector(model, device="cuda")

    assert len(det.detect(np.zeros((100, 100, 3), np.uint8), 0.25)) == 2
    assert det.device == "cpu"
    assert [c["device"] for c in model.calls] == ["cuda", "cpu"]
    assert model.calls[1]["half"] is False


def test_detect_slide_elements_adds_field_next_to_coco_boxes(tmp_path):
    keyframe = tmp_path / "kf.jpg"
    cv2.imwrite(str(keyframe), np.zeros((100, 100, 3), dtype=np.uint8))
    coco = {"keyframe_bboxes": [{"label": "person"}], "tracks_summary": []}
    scenes = [
        {"keyframe_path": str(keyframe), "detected_objects": coco},
        {"keyframe_path": str(keyframe)},  # tracking off: the field is created
        {"keyframe_path": str(tmp_path / "missing.jpg")},
    ]

    counts = sed.detect_slide_elements(scenes, detector=make_detector(FakeYolo(TWO_BOXES)))

    assert counts == {"table": 2, "chart": 2}
    assert scenes[0]["detected_objects"]["keyframe_bboxes"] == [{"label": "person"}]
    assert len(scenes[0]["detected_objects"]["slide_elements"]) == 2
    assert scenes[1]["detected_objects"]["tracks_summary"] == []
    assert len(scenes[1]["detected_objects"]["slide_elements"]) == 2
    assert "detected_objects" not in scenes[2]


def test_missing_weights_file_fails_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="SLIDE_ELEMENT_DETECTOR_WEIGHTS"):
        sed.SlideElementDetector(str(tmp_path / "selected.pt"), "cpu")


def test_stage_off_by_default_and_prefetch_follows_the_setting(monkeypatch):
    assert sed.worker_settings.model_fields["SLIDE_ELEMENT_DETECTOR_WEIGHTS"].default == ""
    monkeypatch.setattr(model_prefetch.worker_settings, "SLIDE_ELEMENT_DETECTOR_WEIGHTS", "")
    assert not sed.slide_element_detection_enabled()
    assert "slide-elements" not in [name for name, _ in model_prefetch.prefetch_tasks()]

    monkeypatch.setattr(model_prefetch.worker_settings, "SLIDE_ELEMENT_DETECTOR_WEIGHTS", "/drive/selected.pt")
    assert sed.slide_element_detection_enabled()
    assert "slide-elements" in [name for name, _ in model_prefetch.prefetch_tasks()]

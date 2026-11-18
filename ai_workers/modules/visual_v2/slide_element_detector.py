"""Slide-element detector: YOLO11n fine-tuned on LVVO (table, chart, photo, illustration boxes).

Benchmark: protocol lvvo-ft-261115-v1, held-out LVVO test (193 images from 31 unseen video groups),
YOLO11n fine-tune mAP@[.50:.95] 0.519 averaged over 3 seeds, against 0.085 for zero-shot
GroundingDINO (benchmark/docs/BENCHMARK.md). The worker uses the seed-1 checkpoint, the one
with the best validation mAP (0.572); seeds are never picked by test score.

It runs next to the COCO RT-DETR + BoT-SORT tracker, not instead of it: it has no person or object
classes. Only scene keyframes are scored (slides change slowly, and no tracking is needed).
Inference mirrors the benchmark adapter: imgsz 640, NMS IoU 0.7, at most 100 boxes, fp16 on CUDA.
The weights are not downloaded: SLIDE_ELEMENT_DETECTOR_WEIGHTS points at the `selected.pt` file
(on Colab, the Drive copy). An empty value turns the stage off.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

from ai_workers.core.config import worker_settings
from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK
from ai_workers.modules.visual_v2.resident_models import SLIDE_ELEMENT_SLOT, visual_resident_enabled

logger = logging.getLogger("ai_workers.modules.visual_v2.slide_element_detector")

IMGSZ = 640
NMS_IOU = 0.7
MAX_DETECTIONS = 100

# LVVO class names (as stored in the checkpoint) -> the labels reported downstream.
LABELS: Dict[str, str] = {
    "table": "table",
    "chart-graph": "chart",
    "photographic-image": "photo",
    "visual-illustration": "illustration",
}


def slide_element_detection_enabled() -> bool:
    return bool(worker_settings.SLIDE_ELEMENT_DETECTOR_WEIGHTS.strip())


def _resolve_device() -> str:
    import torch

    dev = str(worker_settings.SLIDE_ELEMENT_DEVICE).strip().lower()
    if dev in ("cuda", "gpu", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


class SlideElementDetector:
    """Detect slide elements in a BGR image; boxes come back normalised to [0, 1]."""

    def __init__(self, weights: str, device: str):
        if not os.path.isfile(weights):
            raise FileNotFoundError(
                f"Slide-element weights not found: {weights}. Point SLIDE_ELEMENT_DETECTOR_WEIGHTS at the "
                "fine-tuned selected.pt (on Colab: mount Drive first), or leave it empty to skip this stage."
            )
        from ultralytics import YOLO

        self.weights = weights
        self.device = device
        with MODEL_LOAD_LOCK:
            self.model = YOLO(weights)
        self.names = {int(i): str(n) for i, n in self.model.names.items()}

    def label(self, class_id: int) -> str:
        name = self.names.get(int(class_id), str(class_id))
        return LABELS.get(name.strip().lower(), name.strip().lower())

    def to_cpu(self) -> None:
        """Permanent fallback after a CUDA failure."""
        self.device = "cpu"

    def _predict(self, bgr: np.ndarray, min_score: float) -> Any:
        return self.model.predict(
            bgr,
            imgsz=IMGSZ,
            conf=min_score,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            half=self.device == "cuda",
            device=self.device,
            verbose=False,
        )[0]

    def detect(self, bgr: np.ndarray, min_score: float) -> List[Dict[str, Any]]:
        """[{label, confidence, bbox_norm [x1, y1, x2, y2]}], best first."""
        try:
            result = self._predict(bgr, min_score)
        except Exception as err:  # noqa: BLE001
            if self.device == "cpu":
                raise
            logger.warning(f"[SlideEl] Detection failed on {self.device}: {err}. Falling back to CPU.")
            self.to_cpu()
            result = self._predict(bgr, min_score)

        height, width = bgr.shape[:2]
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        scores = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        classes = boxes.cls.cpu().numpy() if len(boxes) else np.zeros(0)
        elements: List[Dict[str, Any]] = []
        for i in np.argsort(-scores):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            elements.append({
                "label": self.label(int(classes[i])),
                "confidence": round(float(scores[i]), 3),
                "bbox_norm": [
                    round(max(0.0, min(1.0, x1 / max(1, width))), 4),
                    round(max(0.0, min(1.0, y1 / max(1, height))), 4),
                    round(max(0.0, min(1.0, x2 / max(1, width))), 4),
                    round(max(0.0, min(1.0, y2 / max(1, height))), 4),
                ],
            })
        return elements


def build_slide_element_detector(weights: str, device: str) -> SlideElementDetector:
    logger.info(f"[SlideEl] Loading slide-element detector '{weights}' on '{device}'...")
    return SlideElementDetector(weights, device)


def load_slide_element_detector() -> SlideElementDetector:
    """The detector; kept loaded between jobs when visual residency is on."""
    weights = worker_settings.SLIDE_ELEMENT_DETECTOR_WEIGHTS.strip()
    device = _resolve_device()
    if visual_resident_enabled():
        return SLIDE_ELEMENT_SLOT.get_or_load(lambda: build_slide_element_detector(weights, device))
    return build_slide_element_detector(weights, device)


def preload_slide_element_detector() -> None:
    """Load (and warm up) the detector at worker start; no-op when the stage is off or not resident."""
    if not slide_element_detection_enabled() or not visual_resident_enabled():
        return
    detector = load_slide_element_detector()
    try:
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), worker_settings.SLIDE_ELEMENT_CONF_THRESHOLD)
    except Exception as err:  # noqa: BLE001
        logger.warning(f"[SlideEl] Warm-up pass skipped: {err}")


def detect_slide_elements(scenes: List[Dict[str, Any]], detector: Optional[SlideElementDetector] = None) -> Dict[str, int]:
    """Add `detected_objects.slide_elements` to every scene with a keyframe; returns label counts.

    The boxes sit next to the COCO `keyframe_bboxes` and `tracks_summary` so the stored
    `detected_objects_json` carries them without a schema change.
    """
    import cv2

    detector = detector or load_slide_element_detector()
    min_score = worker_settings.SLIDE_ELEMENT_CONF_THRESHOLD
    counts: Dict[str, int] = {}
    for scene in scenes:
        path = scene.get("keyframe_path") or ""
        if not path or not os.path.exists(path):
            continue
        image = cv2.imread(path)
        if image is None:
            continue
        try:
            elements = detector.detect(image, min_score)
        except Exception as err:  # noqa: BLE001
            logger.warning(f"[SlideEl] Failed on keyframe '{path}': {err}")
            continue
        detected = scene.get("detected_objects") or {"keyframe_bboxes": [], "tracks_summary": []}
        detected["slide_elements"] = elements
        scene["detected_objects"] = detected
        for element in elements:
            counts[element["label"]] = counts.get(element["label"], 0) + 1
    return counts

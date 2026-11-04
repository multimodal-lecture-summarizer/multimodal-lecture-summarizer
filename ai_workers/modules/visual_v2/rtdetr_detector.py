"""RT-DETR-R18vd object detector (Hugging Face `PekingU/rtdetr_r18vd`, COCO-only training).

This is the detector the benchmark selected among the closed COCO models (mAP 0.429 on the
03a COCO subset) and the one the BoT-SORT tracking comparison used as its detection source.
Inference mirrors the benchmark adapter: `RTDetrImageProcessor` preprocessing, fp16 on CUDA,
`post_process_object_detection`, at most 100 boxes per frame. RT-DETR is NMS-free, so there is no
IoU threshold. Scores are returned unfiltered above a low floor; the tracker needs the
low-confidence boxes for its second association pass.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK

logger = logging.getLogger("ai_workers.modules.visual_v2.rtdetr_detector")

MAX_DETECTIONS = 100


class RtDetrDetector:
    """Detect COCO objects in BGR frames; `labels` maps class ids to lower-case COCO names."""

    def __init__(self, repo: str, device: str):
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self._torch = torch
        self.repo = repo
        self.device = device
        self.half = device == "cuda"
        # Serialised with the other transformers loads (concurrent loads corrupt meta tensors).
        with MODEL_LOAD_LOCK:
            self.processor = RTDetrImageProcessor.from_pretrained(repo)
            self.model = RTDetrForObjectDetection.from_pretrained(repo)
        self.model = self.model.to(device).eval()
        if self.half:
            self.model = self.model.half()
        self.labels = {int(i): str(name).strip().lower() for i, name in self.model.config.id2label.items()}

    def label(self, class_id: int) -> str:
        return self.labels.get(int(class_id), str(class_id))

    def to_cpu(self) -> None:
        """Permanent fallback after a CUDA failure."""
        self.model = self.model.float().to("cpu")
        self.device = "cpu"
        self.half = False

    def detect(self, bgr: np.ndarray, min_score: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xyxy[n,4] pixels, score[n], class_id[n]) with score >= min_score, best first."""
        torch = self._torch
        height, width = bgr.shape[:2]
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, target_sizes=[(height, width)], threshold=min_score
        )[0]
        scores = result["scores"].float().cpu().numpy()
        order = np.argsort(-scores)[:MAX_DETECTIONS]
        if len(order) == 0:
            return np.zeros((0, 4), dtype=float), np.zeros(0, dtype=float), np.zeros(0, dtype=int)
        xyxy = result["boxes"].float().cpu().numpy()[order].astype(float)
        class_ids = result["labels"].cpu().numpy()[order].astype(int)
        return xyxy, scores[order].astype(float), class_ids


def build_detector(repo: str, device: str) -> RtDetrDetector:
    """Load the detector on `device`, falling back to CPU if CUDA placement fails."""
    logger.info(f"[OD&T] Loading detector '{repo}' on device '{device}'...")
    try:
        return RtDetrDetector(repo, device)
    except Exception as err:  # noqa: BLE001
        if device == "cpu":
            raise
        logger.warning(f"[OD&T] Failed to load detector on {device}: {err}. Falling back to CPU.")
        return RtDetrDetector(repo, "cpu")


def warm_up(detector: RtDetrDetector, min_score: float) -> Optional[Exception]:
    """One tiny pass to initialise kernels; returns the error if it failed (non-fatal)."""
    try:
        detector.detect(np.zeros((64, 64, 3), dtype=np.uint8), min_score)
    except Exception as err:  # noqa: BLE001
        return err
    return None

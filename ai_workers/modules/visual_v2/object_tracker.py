from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import torch
from ai_workers.core.config import worker_settings

logger = logging.getLogger("ai_workers.modules.visual_v2.object_tracker")

# Easily confused outdoor / sports / food classes requiring stricter confidence threshold (0.65+)
HIGH_CONF_CLASSES: Set[str] = {
    "car", "truck", "bus", "baseball bat", "frisbee",
    "skateboard", "snowboard", "surfboard", "tennis racket", "kite",
    # Easily confused food textures on outdoor trees, moss, knots, and holes
    "broccoli", "carrot", "donut", "pizza", "hot dog", "sandwich", "cake", "banana", "apple", "orange"
}


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def compute_center_dist(box1: List[float], box2: List[float]) -> float:
    """Calculate normalized Euclidean distance between the centers of two bounding boxes."""
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


class ObjectDetectorTracker:
    """Object detection and ByteTrack tracking for lecture video scenes."""

    HIGH_CONF_CLASSES = HIGH_CONF_CLASSES

    def __init__(
        self,
        model_name: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or getattr(worker_settings, "YOLO_MODEL", "yolov8n.pt")
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else getattr(worker_settings, "YOLO_CONF_THRESHOLD", 0.40)
        )
        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else getattr(worker_settings, "YOLO_IOU_THRESHOLD", 0.5)
        )
        self.device = self._resolve_device(device)
        self.model = self._load_model()

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        dev = requested_device or getattr(worker_settings, "OBJECT_TRACKING_DEVICE", "auto")
        dev = str(dev).strip().lower()
        if dev == "cpu":
            return "cpu"
        if dev in ("cuda", "gpu", "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        logger.info(f"[OD&T] Loading model '{self.model_name}' on device '{self.device}'...")
        try:
            model = YOLO(self.model_name)
            if self.device == "cuda":
                try:
                    model.to("cuda")
                except Exception as cuda_err:
                    logger.warning(
                        f"[OD&T] Failed to load model on CUDA: {cuda_err}. Falling back to CPU."
                    )
                    self.device = "cpu"
                    model.to("cpu")
            return model
        except Exception as e:
            logger.warning(
                f"[OD&T] Failed to initialize model '{self.model_name}': {e}. "
                "Attempting CPU fallback initialization..."
            )
            self.device = "cpu"
            model = YOLO(self.model_name)
            model.to("cpu")
            return model

    def detect_on_keyframe(self, keyframe_path: str) -> List[Dict[str, Any]]:
        """Run object detection on a single keyframe image to extract bounding boxes.

        Returns:
            List of detected boxes with normalized coordinates [x1, y1, x2, y2].
        """
        if not keyframe_path or not os.path.exists(keyframe_path):
            return []

        try:
            results = self.model.predict(
                source=keyframe_path,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            if not results or len(results) == 0 or results[0].boxes is None:
                return []

            res = results[0]
            boxes = res.boxes
            orig_h, orig_w = res.orig_shape

            detected: List[Dict[str, Any]] = []
            for idx in range(len(boxes)):
                xyxy = boxes.xyxy[idx].tolist()
                conf = float(boxes.conf[idx].item())
                cls_id = int(boxes.cls[idx].item())
                label = self.model.names.get(cls_id, str(cls_id)).strip().lower()

                # Anti-flicker / confidence filtering
                min_c = 0.65 if label in HIGH_CONF_CLASSES else self.conf_threshold
                if conf < min_c:
                    continue

                track_id = int(boxes.id[idx].item()) if boxes.id is not None else None

                x1_norm = max(0.0, min(1.0, xyxy[0] / max(1, orig_w)))
                y1_norm = max(0.0, min(1.0, xyxy[1] / max(1, orig_h)))
                x2_norm = max(0.0, min(1.0, xyxy[2] / max(1, orig_w)))
                y2_norm = max(0.0, min(1.0, xyxy[3] / max(1, orig_h)))

                bbox_area = max(0.0, x2_norm - x1_norm) * max(0.0, y2_norm - y1_norm)
                # Discard boxes occupying less than 0.5% of frame (distant speck noise)
                if bbox_area < 0.005:
                    continue

                detected.append({
                    "track_id": track_id,
                    "label": label,
                    "confidence": round(conf, 3),
                    "bbox_norm": [
                        round(x1_norm, 4),
                        round(y1_norm, 4),
                        round(x2_norm, 4),
                        round(y2_norm, 4),
                    ],
                })
            return detected
        except Exception as err:
            logger.warning(f"[OD&T] Failed detection on keyframe '{keyframe_path}': {err}")
            return []

    def track_video_scenes(
        self,
        video_path: str,
        scenes: List[Dict[str, Any]],
        sample_fps: int = 3,
        revoke_check_fn: Optional[Callable[[], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Track objects across video scenes using ByteTrack with anti-flicker filtering,
        intra-scene track stitching, and object-maximized keyframe re-selection.

        Args:
            video_path: Path to the local video file.
            scenes: List of scene dictionaries (from SceneDetector).
            sample_fps: Sampling rate (frames per second) for tracking (default: 3).
            revoke_check_fn: Optional callback function to check for task cancellation.

        Returns:
            Tuple of (updated_scenes, video_objects_summary).
        """
        empty_summary: Dict[str, Any] = {
            "total_unique_tracks": 0,
            "label_distribution": {},
            "most_frequent_objects": [],
            "sample_fps_used": sample_fps,
        }

        if not video_path or not os.path.exists(video_path):
            logger.warning(f"[OD&T] Video file not found at: {video_path}")
            return scenes, empty_summary

        if not scenes:
            return scenes, empty_summary

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"[OD&T] Could not open video file: {video_path}")
            return scenes, empty_summary

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_stride = max(1, int(round(video_fps / max(1, sample_fps))))

        global_max_concurrent: Dict[str, int] = {}
        global_label_distribution: Dict[str, int] = {}

        try:
            for scene_idx, scene in enumerate(scenes):
                if revoke_check_fn is not None:
                    revoke_check_fn()

                start_frame = int(scene.get("start_frame", 0))
                end_frame = int(scene.get("end_frame", total_frames))
                scene_duration = max(0.1, (end_frame - start_frame) / video_fps)
                keyframe_path = scene.get("keyframe_path", "")

                scene_raw_tracks: Dict[int, Dict[str, Any]] = {}
                keyframe_bboxes: List[Dict[str, Any]] = []

                try:
                    # 1. Detect objects on representative keyframe first and compute baseline score
                    orig_kf_unique_labels: Set[str] = set()
                    orig_kf_score = 0.0
                    orig_kf_sharpness = 0.0
                    if keyframe_path and os.path.exists(keyframe_path):
                        keyframe_bboxes = self.detect_on_keyframe(keyframe_path)
                        orig_kf_unique_labels = {b["label"] for b in keyframe_bboxes}
                        try:
                            kf_img = cv2.imread(keyframe_path)
                            if kf_img is not None:
                                gray_kf = cv2.cvtColor(kf_img, cv2.COLOR_BGR2GRAY)
                                orig_kf_sharpness = float(cv2.Laplacian(gray_kf, cv2.CV_64F).var())
                                obj_area_sum = 0.0
                                for b in keyframe_bboxes:
                                    bb = b.get("bbox_norm", [0, 0, 0, 0])
                                    area = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
                                    is_edge = (bb[0] <= 0.015 or bb[2] >= 0.985 or bb[1] <= 0.015 or bb[3] >= 0.985)
                                    edge_factor = 0.5 if is_edge else 1.0
                                    obj_area_sum += b.get("confidence", 0.5) * math.sqrt(max(0.001, area)) * edge_factor
                                orig_kf_score = (
                                    (len(orig_kf_unique_labels) * 10.0)
                                    + (obj_area_sum * 10.0)
                                    + min(orig_kf_sharpness / 20.0, 30.0)
                                )
                        except Exception:
                            pass

                    # 2. Track sampled frames across the scene and evaluate keyframe candidates
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    curr_frame_idx = start_frame

                    best_sampled_frame = None
                    best_sampled_score = -1.0
                    best_sampled_sharpness = 0.0
                    best_sampled_time_sec = None
                    best_sampled_unique_labels: Set[str] = set()

                    while curr_frame_idx <= end_frame:
                        if revoke_check_fn is not None and (curr_frame_idx - start_frame) % (frame_stride * 10) == 0:
                            revoke_check_fn()

                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break

                        frame_time_sec = round(curr_frame_idx / video_fps, 2)
                        h_f, w_f = frame.shape[:2]

                        try:
                            results = self.model.track(
                                source=frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                conf=self.conf_threshold,
                                iou=self.iou_threshold,
                                device=self.device,
                                verbose=False,
                            )
                        except Exception as track_err:
                            if self.device != "cpu":
                                logger.warning(
                                    f"[OD&T] Tracking failed on {self.device}: {track_err}. "
                                    "Falling back to CPU for remainder of video."
                                )
                                self.device = "cpu"
                                results = self.model.track(
                                    source=frame,
                                    persist=True,
                                    tracker="bytetrack.yaml",
                                    conf=self.conf_threshold,
                                    iou=self.iou_threshold,
                                    device="cpu",
                                    verbose=False,
                                )
                            else:
                                raise track_err

                        # Compute object presence and sharpness for this candidate frame
                        gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        f_sharpness = float(cv2.Laplacian(gray_f, cv2.CV_64F).var())
                        f_obj_score = 0.0
                        frame_unique_labels: Set[str] = set()

                        if results and len(results) > 0 and results[0].boxes is not None:
                            boxes = results[0].boxes
                            for b_i in range(len(boxes)):
                                conf = float(boxes.conf[b_i].item())
                                cls_id = int(boxes.cls[b_i].item())
                                label = self.model.names.get(cls_id, str(cls_id)).strip().lower()

                                # Stricter confidence filter for high-FP outdoor / food classes
                                min_c = 0.65 if label in HIGH_CONF_CLASSES else self.conf_threshold
                                if conf < min_c:
                                    continue

                                xyxy = boxes.xyxy[b_i].tolist()
                                x1_n = max(0.0, min(1.0, xyxy[0] / max(1, w_f)))
                                y1_n = max(0.0, min(1.0, xyxy[1] / max(1, h_f)))
                                x2_n = max(0.0, min(1.0, xyxy[2] / max(1, w_f)))
                                y2_n = max(0.0, min(1.0, xyxy[3] / max(1, h_f)))
                                b_area = max(0.0, x2_n - x1_n) * max(0.0, y2_n - y1_n)

                                # Discard distant noise boxes (<0.5% screen area)
                                if b_area < 0.005:
                                    continue

                                is_edge = (x1_n <= 0.015 or x2_n >= 0.985 or y1_n <= 0.015 or y2_n >= 0.985)
                                edge_penalty = 0.5 if is_edge else 1.0
                                f_obj_score += conf * math.sqrt(max(0.001, b_area)) * edge_penalty
                                frame_unique_labels.add(label)

                                if boxes.id is None:
                                    continue
                                t_id = int(boxes.id[b_i].item())

                                bbox_norm = [
                                    round(x1_n, 4),
                                    round(y1_n, 4),
                                    round(x2_n, 4),
                                    round(y2_n, 4),
                                ]

                                if t_id not in scene_raw_tracks:
                                    scene_raw_tracks[t_id] = {
                                        "track_id": t_id,
                                        "label": label,
                                        "confidences": [conf],
                                        "first_seen_sec": frame_time_sec,
                                        "last_seen_sec": frame_time_sec,
                                        "frames_appeared": 1,
                                        "sample_bbox_norm": bbox_norm,
                                        "max_conf": conf,
                                        "best_timestamp_sec": frame_time_sec,
                                    }
                                else:
                                    st = scene_raw_tracks[t_id]
                                    st["confidences"].append(conf)
                                    st["last_seen_sec"] = frame_time_sec
                                    st["frames_appeared"] += 1
                                    if conf > st["max_conf"]:
                                        st["max_conf"] = conf
                                        st["sample_bbox_norm"] = bbox_norm
                                        st["best_timestamp_sec"] = frame_time_sec

                        # Composite candidate score: heavy weight on count of unique object classes present in frame
                        # Formula: (unique_labels * 10.0) + (f_obj_score * 10.0) + normalized_sharpness
                        f_composite = (
                            (len(frame_unique_labels) * 10.0)
                            + (f_obj_score * 10.0)
                            + min(f_sharpness / 20.0, 30.0)
                        )
                        if f_composite > best_sampled_score:
                            best_sampled_score = f_composite
                            best_sampled_frame = frame.copy()
                            best_sampled_sharpness = f_sharpness
                            best_sampled_time_sec = frame_time_sec
                            best_sampled_unique_labels = set(frame_unique_labels)

                        # Advance cap by frame_stride
                        if frame_stride > 1:
                            for _ in range(frame_stride - 1):
                                if not cap.grab():
                                    break
                            curr_frame_idx += frame_stride
                        else:
                            curr_frame_idx += 1

                    # 3. Object-Maximized Keyframe Selection:
                    # If a sampled frame contains MORE unique object classes (e.g. Person + Motorcycle) or significantly
                    # higher object presence than the original static keyframe, re-select it as the representative keyframe.
                    if best_sampled_frame is not None and keyframe_path:
                        best_n_labels = len(best_sampled_unique_labels)
                        orig_n_labels = len(orig_kf_unique_labels)

                        should_update = (
                            (best_n_labels > orig_n_labels)
                            or (orig_n_labels == 0 and best_n_labels > 0)
                            or (best_sampled_score > (orig_kf_score + 1.0) and best_n_labels >= orig_n_labels)
                        )
                        if should_update:
                            cv2.imwrite(keyframe_path, best_sampled_frame)
                            keyframe_bboxes = self.detect_on_keyframe(keyframe_path)
                            scene["blur_score"] = round(best_sampled_sharpness, 2)
                            if best_sampled_time_sec is not None:
                                scene["keyframe_timestamp"] = best_sampled_time_sec
                            logger.info(
                                f"[OD&T] Scene {scene.get('scene_index', scene_idx)}: Re-selected keyframe at {best_sampled_time_sec}s "
                                f"(composite_score={best_sampled_score:.2f}, labels={best_n_labels}, boxes={len(keyframe_bboxes)})"
                            )

                    # 4. Intra-Scene Track Stitching:
                    # In the same scene, if 2 tracks share the same label, don't overlap in time, have a short gap (<= 2.0s),
                    # and are spatially close (IoU >= 0.20 or center distance < 0.25), merge them into a single continuous track.
                    tracks_by_label: Dict[str, List[Dict[str, Any]]] = {}
                    for st in scene_raw_tracks.values():
                        tracks_by_label.setdefault(st["label"], []).append(st)

                    stitched_tracks: List[Dict[str, Any]] = []
                    for lbl, trk_list in tracks_by_label.items():
                        trk_list.sort(key=lambda x: x["first_seen_sec"])
                        i = 0
                        while i < len(trk_list):
                            current = trk_list[i]
                            j = i + 1
                            while j < len(trk_list):
                                candidate = trk_list[j]
                                time_gap = candidate["first_seen_sec"] - current["last_seen_sec"]
                                if 0.0 <= time_gap <= 2.0:
                                    iou = compute_iou(current["sample_bbox_norm"], candidate["sample_bbox_norm"])
                                    dist = compute_center_dist(current["sample_bbox_norm"], candidate["sample_bbox_norm"])
                                    if iou >= 0.20 or dist < 0.25:
                                        # Merge candidate into current
                                        current["confidences"].extend(candidate["confidences"])
                                        current["first_seen_sec"] = min(current["first_seen_sec"], candidate["first_seen_sec"])
                                        current["last_seen_sec"] = max(current["last_seen_sec"], candidate["last_seen_sec"])
                                        current["frames_appeared"] += candidate["frames_appeared"]
                                        if candidate["max_conf"] > current["max_conf"]:
                                            current["max_conf"] = candidate["max_conf"]
                                            current["sample_bbox_norm"] = candidate["sample_bbox_norm"]
                                            current["best_timestamp_sec"] = candidate["best_timestamp_sec"]
                                        trk_list.pop(j)
                                        continue
                                j += 1
                            stitched_tracks.append(current)
                            i += 1

                    # 5. Minimum Track Persistence (Noise & Flicker Elimination):
                    # Only keep tracks that appeared in >= 2 frames (or confidence >= 0.70 for short scenes < 2.0s).
                    # Eliminates single-frame false positives (e.g. distant dirt dust mistakenly tagged as car).
                    persistent_tracks: List[Dict[str, Any]] = []
                    for st in stitched_tracks:
                        if st["frames_appeared"] >= 2 or (scene_duration < 2.0 and st["max_conf"] >= 0.70):
                            persistent_tracks.append(st)

                    # 6. Assign Clean Sequential Track IDs per scene (e.g. 1, 2, 3...)
                    persistent_tracks.sort(key=lambda x: x["first_seen_sec"])
                    for idx, st in enumerate(persistent_tracks, start=1):
                        st["clean_id"] = idx

                    # 7. Match Keyframe BBoxes with persistent tracks
                    matched_clean_ids: Set[int] = set()
                    for kf_box in keyframe_bboxes:
                        best_st = None
                        best_iou = 0.0
                        for st in persistent_tracks:
                            if st["label"] == kf_box["label"]:
                                iou = compute_iou(kf_box["bbox_norm"], st["sample_bbox_norm"])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_st = st
                        if best_st is not None and best_iou >= 0.10:
                            kf_box["track_id"] = best_st["clean_id"]
                            matched_clean_ids.add(best_st["clean_id"])

                    # If any keyframe box wasn't matched with a sampled track, assign a sequential ID
                    next_id = len(persistent_tracks) + 1
                    for kf_box in keyframe_bboxes:
                        if kf_box.get("track_id") is None:
                            kf_box["track_id"] = next_id
                            matched_clean_ids.add(next_id)
                            next_id += 1

                    # 8. Build final tracks_summary with `in_keyframe` and `best_timestamp_sec`
                    tracks_summary: List[Dict[str, Any]] = []
                    for st in persistent_tracks:
                        avg_c = sum(st["confidences"]) / max(1, len(st["confidences"]))
                        cid = st["clean_id"]
                        is_in_kf = (cid in matched_clean_ids) or any(
                            kb["label"] == st["label"] and compute_iou(kb["bbox_norm"], st["sample_bbox_norm"]) >= 0.15
                            for kb in keyframe_bboxes
                        )
                        best_ts = round(st.get("best_timestamp_sec", st["first_seen_sec"]), 2)

                        tracks_summary.append({
                            "track_id": cid,
                            "label": st["label"],
                            "avg_confidence": round(avg_c, 3),
                            "first_seen_sec": round(st["first_seen_sec"], 2),
                            "last_seen_sec": round(st["last_seen_sec"], 2),
                            "frames_appeared": st["frames_appeared"],
                            "sample_bbox_norm": st["sample_bbox_norm"],
                            "in_keyframe": bool(is_in_kf),
                            "best_timestamp_sec": best_ts,
                        })

                    # Include unmatched keyframe detections into tracks_summary as keyframe presence
                    kf_time = scene.get("keyframe_timestamp", round(start_frame / video_fps, 2))
                    for kf_box in keyframe_bboxes:
                        kid = kf_box.get("track_id")
                        if kid and not any(t["track_id"] == kid for t in tracks_summary):
                            tracks_summary.append({
                                "track_id": kid,
                                "label": kf_box["label"],
                                "avg_confidence": round(float(kf_box.get("confidence", 0.5)), 3),
                                "first_seen_sec": round(kf_time, 2),
                                "last_seen_sec": round(kf_time, 2),
                                "frames_appeared": 1,
                                "sample_bbox_norm": kf_box["bbox_norm"],
                                "in_keyframe": True,
                                "best_timestamp_sec": round(kf_time, 2),
                            })

                    # 9. Compute scene object counts from full tracks_summary
                    object_counts: Dict[str, int] = {}
                    for ts in tracks_summary:
                        lbl = ts["label"]
                        object_counts[lbl] = object_counts.get(lbl, 0) + 1

                    scene["detected_objects"] = {
                        "keyframe_bboxes": keyframe_bboxes,
                        "tracks_summary": tracks_summary,
                    }
                    scene["object_counts"] = object_counts

                    # Track max concurrent objects per label across scenes
                    for lbl, count in object_counts.items():
                        global_max_concurrent[lbl] = max(global_max_concurrent.get(lbl, 0), count)
                        global_label_distribution[lbl] = global_label_distribution.get(lbl, 0) + count

                except Exception as scene_err:
                    logger.warning(
                        f"[OD&T] Error tracking scene {scene.get('scene_index', scene_idx)}: {scene_err}. "
                        "Returning empty detections for this scene."
                    )
                    scene["detected_objects"] = {
                        "keyframe_bboxes": keyframe_bboxes,
                        "tracks_summary": [],
                    }
                    scene["object_counts"] = {}

        finally:
            cap.release()

        # Compute video-level objects summary with realistic deduplicated unique track count
        # Instead of summing unbounded ByteTrack IDs, use sum of maximum concurrent tracks per class
        total_unique_tracks = sum(global_max_concurrent.values())
        if total_unique_tracks == 0 and global_label_distribution:
            total_unique_tracks = len(global_label_distribution)

        sorted_labels = [
            lbl
            for lbl, _ in sorted(
                global_label_distribution.items(), key=lambda item: item[1], reverse=True
            )
        ]
        video_objects_summary: Dict[str, Any] = {
            "total_unique_tracks": total_unique_tracks,
            "label_distribution": global_label_distribution,
            "most_frequent_objects": sorted_labels,
            "sample_fps_used": sample_fps,
        }

        logger.info(
            f"[OD&T] Completed tracking for {len(scenes)} scenes. "
            f"Realistic total unique tracks: {total_unique_tracks}, "
            f"Labels: {dict(global_label_distribution)}"
        )
        return scenes, video_objects_summary


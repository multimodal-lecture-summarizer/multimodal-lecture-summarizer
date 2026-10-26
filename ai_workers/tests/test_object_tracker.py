import os
import tempfile
import cv2
import numpy as np
import pytest

from ai_workers.modules.visual_v2.object_tracker import ObjectDetectorTracker, compute_iou
from ai_workers.modules.summarization.summarizer import Summarizer


def test_compute_iou():
    box1 = [0.1, 0.1, 0.5, 0.5]
    box2 = [0.1, 0.1, 0.5, 0.5]
    assert compute_iou(box1, box2) == 1.0

    box3 = [0.6, 0.6, 0.9, 0.9]
    assert compute_iou(box1, box3) == 0.0

    box4 = [0.3, 0.3, 0.7, 0.7]
    iou = compute_iou(box1, box4)
    assert 0.0 < iou < 1.0


def test_tracker_init():
    tracker = ObjectDetectorTracker(device="cpu")
    assert tracker.device == "cpu"
    assert tracker.model is not None


def test_detect_on_keyframe_nonexistent():
    tracker = ObjectDetectorTracker(device="cpu")
    res = tracker.detect_on_keyframe("nonexistent_path_xyz.jpg")
    assert res == []


def test_detect_on_keyframe_synthetic():
    tracker = ObjectDetectorTracker(device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img_path = f.name

    try:
        # Create a simple synthetic image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle simulating an object
        cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.imwrite(img_path, img)

        res = tracker.detect_on_keyframe(img_path)
        assert isinstance(res, list)
        for item in res:
            assert "track_id" in item
            assert "label" in item
            assert "confidence" in item
            assert "bbox_norm" in item
            bbox = item["bbox_norm"]
            assert len(bbox) == 4
            for coord in bbox:
                assert 0.0 <= coord <= 1.0
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def test_track_video_scenes():
    tracker = ObjectDetectorTracker(device="cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "test_video.mp4")
        keyframe_path = os.path.join(tmpdir, "kf_0.png")

        # Create keyframe
        kf_img = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(kf_img, (160, 120), 40, (0, 255, 0), -1)
        cv2.imwrite(keyframe_path, kf_img)

        # Create 1-second video at 10 fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.circle(frame, (160 + i * 2, 120), 40, (0, 255, 0), -1)
            writer.write(frame)
        writer.release()

        scenes = [
            {
                "scene_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "start_frame": 0,
                "end_frame": 9,
                "keyframe_path": keyframe_path,
            }
        ]

        revoked_checked = []
        def check_rev():
            revoked_checked.append(True)

        updated_scenes, summary = tracker.track_video_scenes(
            video_path=video_path,
            scenes=scenes,
            sample_fps=3,
            revoke_check_fn=check_rev,
        )

        assert len(updated_scenes) == 1
        scene = updated_scenes[0]
        assert "detected_objects" in scene
        assert "object_counts" in scene

        det = scene["detected_objects"]
        assert "keyframe_bboxes" in det
        assert "tracks_summary" in det
        assert isinstance(det["keyframe_bboxes"], list)
        assert isinstance(det["tracks_summary"], list)
        assert isinstance(scene["object_counts"], dict)

        assert "total_unique_tracks" in summary
        assert "label_distribution" in summary
        assert "most_frequent_objects" in summary
        assert summary["sample_fps_used"] == 3
        assert len(revoked_checked) > 0


def test_rag_chunk_includes_objects():
    summarizer = Summarizer()
    video_id = "test_vid_123"
    utterances = [
        {"start": 0.0, "end": 10.0, "text": "Hôm nay chúng ta học về Object Detection."}
    ]
    slides = [
        {
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "ocr_text": "YOLO and ByteTrack",
            "caption": "A slide showing computer vision architecture",
            "object_counts": {"person": 1, "laptop": 2},
        }
    ]

    # Mock chromadb_service to capture added chunks
    captured_chunks = []
    class MockChromaService:
        def add_transcript_chunks(self, vid, chunks, metadatas):
            captured_chunks.extend(chunks)
            return True

    import sys
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend"))
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from unittest.mock import patch
    with patch("app.services.chromadb.chromadb_service", MockChromaService()):
        res = summarizer.build_rag_index(video_id, utterances, slides)
        assert res is True
        assert len(captured_chunks) > 0
        chunk_text = captured_chunks[0]
        assert "Đối tượng xuất hiện:" in chunk_text
        assert "1 person" in chunk_text
        assert "2 laptop" in chunk_text


def test_track_summary_metadata():
    tracker = ObjectDetectorTracker(device="cpu")
    # Verify HIGH_CONF_CLASSES thresholding logic
    assert "car" in tracker.HIGH_CONF_CLASSES
    assert "truck" in tracker.HIGH_CONF_CLASSES
    assert tracker.conf_threshold == 0.40

import json
import os
import unittest
import uuid
from unittest.mock import MagicMock

from app.api.v1.evaluation import (
    get_benchmarks,
    get_object_detection_benchmarks,
    get_evaluation_history,
    evaluate_video,
)
from app.api.v1.stats import get_performance_stats
from app.models.user import User
from app.models.video import Video, VideoScene, VideoMetadata
from app.models.summary import Summary
from app.models.evaluation import EvaluationLog
from app.models.job import Job
from app.core.constants import UserRole, JobStatus, JobType
from app.schemas.evaluation import VideoEvaluationRequest
from app.services.evaluation_service import (
    calculate_iou,
    compute_ap_per_class,
    compute_map_metrics,
    evaluate_object_detection,
    get_benchmark_comparison_data,
)


class TestObjectDetectionEvaluation(unittest.TestCase):
    def test_calculate_iou_exact_overlap(self):
        box_a = [0.1, 0.1, 0.5, 0.5]
        box_b = [0.1, 0.1, 0.5, 0.5]
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 1.0, places=4)

    def test_calculate_iou_disjoint(self):
        box_a = [0.0, 0.0, 0.2, 0.2]
        box_b = [0.5, 0.5, 0.8, 0.8]
        iou = calculate_iou(box_a, box_b)
        self.assertEqual(iou, 0.0)

    def test_calculate_iou_partial_overlap(self):
        # Box A: [0, 0, 10, 10], Area = 100
        # Box B: [5, 5, 15, 15], Area = 100
        # Inter: [5, 5, 10, 10], Area = 25
        # Union: 100 + 100 - 25 = 175
        # IoU: 25 / 175 = 1 / 7 ≈ 0.142857
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [5.0, 5.0, 15.0, 15.0]
        iou = calculate_iou(box_a, box_b)
        self.assertAlmostEqual(iou, 25.0 / 175.0, places=4)

    def test_calculate_iou_inverted_and_degenerate_boxes(self):
        # Inverted box [10, 10, 0, 0] should be normalized safely
        box_inv = [10.0, 10.0, 0.0, 0.0]
        box_norm = [0.0, 0.0, 10.0, 10.0]
        iou = calculate_iou(box_inv, box_norm)
        self.assertAlmostEqual(iou, 1.0, places=4)

        # Zero-area box
        box_zero = [0.5, 0.5, 0.5, 0.5]
        iou_zero = calculate_iou(box_zero, box_norm)
        self.assertEqual(iou_zero, 0.0)

    def test_compute_ap_per_class_perfect_predictions(self):
        gts = [
            {"scene_index": 1, "bbox_norm": [0.1, 0.1, 0.4, 0.4], "label": "person"},
            {"scene_index": 1, "bbox_norm": [0.5, 0.5, 0.9, 0.9], "label": "person"},
        ]
        preds = [
            {"scene_index": 1, "bbox_norm": [0.11, 0.1, 0.4, 0.4], "confidence": 0.95, "label": "person"},
            {"scene_index": 1, "bbox_norm": [0.5, 0.51, 0.9, 0.9], "confidence": 0.88, "label": "person"},
        ]
        res = compute_ap_per_class(preds, gts, iou_threshold=0.50)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fp"], 0)
        self.assertAlmostEqual(res["precision"], 1.0, places=2)
        self.assertAlmostEqual(res["recall"], 1.0, places=2)
        self.assertAlmostEqual(res["ap"], 1.0, places=2)

    def test_compute_ap_per_class_greedy_matching(self):
        # 1 GT, but 2 Predictions overlapping it.
        # Only the highest confidence prediction should be TP; the other must be FP.
        gts = [
            {"scene_index": 1, "bbox_norm": [0.2, 0.2, 0.6, 0.6], "label": "laptop"}
        ]
        preds = [
            {"scene_index": 1, "bbox_norm": [0.2, 0.2, 0.6, 0.6], "confidence": 0.92, "label": "laptop"},
            {"scene_index": 1, "bbox_norm": [0.21, 0.2, 0.6, 0.59], "confidence": 0.85, "label": "laptop"},
        ]
        res = compute_ap_per_class(preds, gts, iou_threshold=0.50)
        self.assertEqual(res["tp"], 1)
        self.assertEqual(res["fp"], 1)
        self.assertAlmostEqual(res["recall"], 1.0, places=2)
        self.assertAlmostEqual(res["precision"], 0.5, places=2)

    def test_compute_ap_per_class_empty_cases(self):
        # Empty predictions
        res_no_pred = compute_ap_per_class([], [{"bbox_norm": [0, 0, 1, 1], "scene_index": 1}], 0.5)
        self.assertEqual(res_no_pred["ap"], 0.0)
        self.assertEqual(res_no_pred["tp"], 0)

        # Empty ground-truth
        res_no_gt = compute_ap_per_class([{"bbox_norm": [0, 0, 1, 1], "confidence": 0.9, "scene_index": 1}], [], 0.5)
        self.assertEqual(res_no_gt["ap"], 0.0)
        self.assertEqual(res_no_gt["fp"], 1)

    def test_compute_map_metrics_multi_class(self):
        preds = {
            "person": [
                {"scene_index": 1, "bbox_norm": [0.1, 0.1, 0.4, 0.4], "confidence": 0.9},
            ],
            "laptop": [
                {"scene_index": 1, "bbox_norm": [0.5, 0.5, 0.8, 0.8], "confidence": 0.85},
            ],
        }
        gts = {
            "person": [
                {"scene_index": 1, "bbox_norm": [0.1, 0.1, 0.4, 0.4]},
            ],
            "laptop": [
                {"scene_index": 1, "bbox_norm": [0.5, 0.5, 0.8, 0.8]},
            ],
        }
        res = compute_map_metrics(preds, gts)
        self.assertAlmostEqual(res["od_map50"], 1.0, places=2)
        self.assertAlmostEqual(res["od_map50_95"], 1.0, places=2)
        self.assertIn("person", res["per_class_metrics"])
        self.assertIn("laptop", res["per_class_metrics"])
        # Academic property: mAP@50 should always be >= mAP@50:95
        self.assertGreaterEqual(res["od_map50"], res["od_map50_95"])

    def test_evaluate_object_detection_mode_1_ground_truth(self):
        mock_scenes = [
            {
                "scene_index": 1,
                "detected_objects_json": {
                    "keyframe_detections": [
                        {"label": "person", "confidence": 0.92, "bbox_norm": [0.3, 0.2, 0.6, 0.8]},
                        {"label": "laptop", "confidence": 0.87, "bbox_norm": [0.2, 0.6, 0.4, 0.8]},
                    ],
                    "tracks_summary": [
                        {"track_id": 1, "label": "person", "avg_confidence": 0.91, "first_seen_sec": 0.0, "last_seen_sec": 12.0},
                        {"track_id": 2, "label": "laptop", "avg_confidence": 0.85, "first_seen_sec": 0.0, "last_seen_sec": 8.5},
                    ],
                },
            }
        ]

        # Reference objects provided
        reference_objects = [
            {
                "scene_index": 1,
                "objects": [
                    {"label": "person", "bbox_norm": [0.31, 0.21, 0.6, 0.8]},
                    {"label": "laptop", "bbox_norm": [0.2, 0.6, 0.41, 0.8]},
                ],
            }
        ]

        res = evaluate_object_detection(mock_scenes, reference_objects)
        self.assertEqual(res["evaluation_mode"], "ground_truth")
        self.assertAlmostEqual(res["od_map50"], 1.0, places=2)
        self.assertEqual(res["tracking_metrics"]["total_unique_tracks"], 2)
        self.assertGreater(res["tracking_metrics"]["avg_track_duration_sec"], 5.0)

    def test_evaluate_object_detection_mode_2_heuristic_fallback(self):
        mock_scenes = [
            {
                "scene_index": 1,
                "detected_objects_json": {
                    "keyframe_detections": [
                        {"label": "person", "confidence": 0.88, "bbox_norm": [0.3, 0.2, 0.6, 0.8]},
                    ],
                    "tracks_summary": [
                        {
                            "track_id": 1,
                            "label": "person",
                            "avg_confidence": 0.88,
                            "frames_appeared": 5,
                            "sample_bbox_norm": [0.3, 0.2, 0.6, 0.8],
                            "first_seen_sec": 2.0,
                            "last_seen_sec": 14.0,
                        }
                    ],
                },
            }
        ]

        # No reference objects provided -> should fallback to pseudo_ground_truth_tracking
        res = evaluate_object_detection(mock_scenes, reference_objects=None)
        self.assertEqual(res["evaluation_mode"], "pseudo_ground_truth_tracking")
        self.assertGreaterEqual(res["od_map50"], 0.0)
        self.assertIn("person", res["per_class_metrics"])
        self.assertEqual(res["tracking_metrics"]["total_unique_tracks"], 1)

    def test_benchmarks_data_contains_od_models(self):
        data = get_benchmark_comparison_data()
        self.assertIsNotNone(data.object_detection_benchmarks)
        self.assertEqual(len(data.object_detection_benchmarks), 3)

        models = [m.model for m in data.object_detection_benchmarks]
        self.assertTrue(any("yolov8n" in m for m in models))
        self.assertTrue(any("yolo11n" in m for m in models))
        self.assertTrue(any("yolov8s" in m for m in models))

        prod_models = [m for m in data.object_detection_benchmarks if m.is_production]
        self.assertEqual(len(prod_models), 1)
        self.assertTrue("yolov8n" in prod_models[0].model)

        # Conclusions should have OD entry
        od_concl = [c for c in data.summary_conclusions if "YOLO" in c or "ByteTrack" in c]
        self.assertGreaterEqual(len(od_concl), 1)

    def test_get_od_benchmarks_api(self):
        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        res = get_object_detection_benchmarks(current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertEqual(len(res.data), 3)

    def test_evaluate_video_api_with_od(self):
        video_id = uuid.uuid4()
        mock_video = Video(
            video_id=video_id,
            title="Lecture on Deep Learning Vision",
            file_path="/storage/videos/vision_lecture.mp4",
        )
        mock_scene = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=video_id,
            scene_index=1,
            script="In this slide we discuss object detection with convolutional neural networks.",
            detected_objects_json={
                "keyframe_detections": [
                    {"label": "person", "confidence": 0.94, "bbox_norm": [0.3, 0.2, 0.6, 0.8]},
                ],
                "tracks_summary": [
                    {"track_id": 1, "label": "person", "avg_confidence": 0.94, "first_seen_sec": 0.0, "last_seen_sec": 10.0},
                ],
            },
        )
        mock_summary = Summary(
            summary_id=uuid.uuid4(),
            video_id=video_id,
            summary_text="Discussion of object detection methods using convolutional networks.",
        )

        mock_db = MagicMock()

        def query_side_effect(model_cls):
            q = MagicMock()
            if model_cls == Video:
                q.filter.return_value.first.return_value = mock_video
            elif model_cls == VideoScene:
                q.filter.return_value.order_by.return_value.all.return_value = [mock_scene]
            elif model_cls == Summary:
                q.filter.return_value.first.return_value = mock_summary
            return q

        mock_db.query.side_effect = query_side_effect

        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        req = VideoEvaluationRequest(
            video_id=str(video_id),
            reference_objects=[
                {
                    "scene_index": 1,
                    "label": "person",
                    "bbox_norm": [0.3, 0.2, 0.6, 0.8],
                }
            ],
        )

        res = evaluate_video(request=req, db=mock_db, current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertTrue(res.data.has_object_detection)
        self.assertIsNotNone(res.data.od_map50)
        self.assertAlmostEqual(res.data.od_map50, 1.0, places=2)
        self.assertIsNotNone(res.data.od_details)

    def test_get_evaluation_history_api(self):
        video_id = uuid.uuid4()
        eval_log = EvaluationLog(
            eval_id=uuid.uuid4(),
            video_id=video_id,
            od_precision=0.91,
            od_recall=0.88,
            od_map50=0.89,
            od_map50_95=0.62,
        )
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [eval_log]

        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        res = get_evaluation_history(video_id=str(video_id), db=mock_db, current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertEqual(len(res.data), 1)
        self.assertEqual(res.data[0].od_map50, 0.89)

    def test_ground_truth_sample_file_validity(self):
        # Verify benchmarks/object_detection_gt_sample.json exists and adheres to schema
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        gt_path = os.path.join(base_dir, "benchmarks", "object_detection_gt_sample.json")
        self.assertTrue(os.path.exists(gt_path), f"File {gt_path} not found")

        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("reference_objects", data)
        self.assertGreater(len(data["reference_objects"]), 0)

        # Run evaluate_object_detection using this GT sample against dummy scenes
        dummy_scenes = [
            {
                "scene_index": 1,
                "detected_objects_json": {
                    "keyframe_detections": [
                        {"label": "person", "confidence": 0.95, "bbox_norm": [0.35, 0.15, 0.65, 0.85]},
                        {"label": "laptop", "confidence": 0.90, "bbox_norm": [0.20, 0.60, 0.45, 0.88]},
                    ],
                    "tracks_summary": [
                        {"track_id": 1, "label": "person", "avg_confidence": 0.93, "first_seen_sec": 0, "last_seen_sec": 30},
                    ],
                },
            }
        ]
        eval_res = evaluate_object_detection(dummy_scenes, data["reference_objects"])
        self.assertEqual(eval_res["evaluation_mode"], "ground_truth")
        self.assertIn("person", eval_res["per_class_metrics"])
        self.assertGreater(eval_res["od_map50"], 0.0)

    def test_performance_stats_od_correlation(self):
        # Verify get_performance_stats includes detected_objects_count and avg_object_confidence
        mock_db = MagicMock()
        mock_db.query.return_value.join.return_value.options.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_db.query.return_value.count.return_value = 10
        mock_db.query.return_value.filter.return_value.count.return_value = 0

        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        res = get_performance_stats(db=mock_db, current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertIsNotNone(res.data.summary_metrics.avg_detected_objects)
        self.assertGreater(len(res.data.scatter_data), 0)
        first_pt = res.data.scatter_data[0]
        self.assertIsNotNone(first_pt.detected_objects_count)
        self.assertIsNotNone(first_pt.avg_object_confidence)


if __name__ == "__main__":
    unittest.main()

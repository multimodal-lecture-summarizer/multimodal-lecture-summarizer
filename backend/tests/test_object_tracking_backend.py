"""Unit and integration tests for Phase 2: Object Detection & Tracking backend support.

Tests cover:
- Pydantic DTO validation, normalization, and camelCase serialization.
- sync_job_status mapping of video_objects_summary, detected_objects, and object_counts.
- get_video_scenes filtering by object_label and min_confidence.
- get_video_objects aggregation endpoint.
- SQLAlchemy 2.0 model column definitions.
"""

from __future__ import annotations

import unittest
import uuid
from unittest.mock import MagicMock, patch

from app.models.user import User
from app.models.video import Video, VideoScene
from app.models.summary import Summary
from app.models.evaluation import EvaluationLog
from app.core.constants import UserRole, VideoStatus, JobStatus
from app.schemas.video import (
    KeyframeBBoxDTO,
    ObjectTrackSummaryDTO,
    SceneDetectedObjectsDTO,
    VideoObjectsSummaryDTO,
    VideoSceneDTO,
    VideoObjectsDetailResponseDTO,
)
from app.schemas.summary import SummaryDTO
from app.schemas.evaluation import ODEvalMetricsDTO, EvaluationDTO
from app.api.v1.jobs import sync_job_status
from app.api.v1.videos import get_video_scenes, get_video_objects


class ObjectTrackingBackendTests(unittest.TestCase):
    def setUp(self):
        self.user_id = uuid.uuid4()
        self.video_id = uuid.uuid4()
        self.admin_user = User(
            user_id=self.user_id,
            email="admin@example.com",
            role=UserRole.ADMIN,
            is_active=True,
        )
        self.regular_user = User(
            user_id=self.user_id,
            email="user@example.com",
            role=UserRole.USER,
            is_active=True,
        )

    # -------------------------------------------------------------------------
    # 1. Pydantic Schemas & DTO Validation
    # -------------------------------------------------------------------------
    def test_keyframe_bbox_and_track_summary_dto(self):
        bbox = KeyframeBBoxDTO(
            track_id=1,
            label="person",
            confidence=0.89,
            bbox_norm=[0.1, 0.2, 0.4, 0.8],
        )
        self.assertEqual(bbox.track_id, 1)
        self.assertEqual(bbox.label, "person")
        self.assertEqual(bbox.confidence, 0.89)
        self.assertEqual(bbox.bbox_norm, [0.1, 0.2, 0.4, 0.8])

        track = ObjectTrackSummaryDTO(
            track_id=1,
            label="person",
            avg_confidence=0.88,
            first_seen_sec=5.0,
            last_seen_sec=20.0,
            frames_appeared=45,
            sample_bbox_norm=[0.1, 0.2, 0.4, 0.8],
        )
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.frames_appeared, 45)

        scene_objects = SceneDetectedObjectsDTO(
            keyframe_bboxes=[bbox],
            tracks_summary=[track],
        )
        self.assertEqual(len(scene_objects.keyframe_bboxes), 1)
        self.assertEqual(len(scene_objects.tracks_summary), 1)

    def test_video_scene_dto_normalization(self):
        """Tests that passing raw 'detected_objects' and 'object_counts' normalizes to JSON fields."""
        raw_scene_data = {
            "scene_id": uuid.uuid4(),
            "video_id": self.video_id,
            "scene_index": 1,
            "start_seconds": 0.0,
            "end_seconds": 10.0,
            "start_timecode": "00:00:00.000",
            "end_timecode": "00:00:10.000",
            "start_frame": 0,
            "end_frame": 300,
            "keyframe_path": "keyframes/1.jpg",
            "keyframe_url": "https://pub.dev/1.jpg",
            "caption": "Introduction slide",
            "detected_objects": {
                "keyframe_bboxes": [
                    {"track_id": 1, "label": "laptop", "confidence": 0.95, "bbox_norm": [0.2, 0.3, 0.7, 0.8]}
                ],
                "tracks_summary": [
                    {"track_id": 1, "label": "laptop", "avg_confidence": 0.94, "first_seen_sec": 0.0, "last_seen_sec": 10.0, "frames_appeared": 30}
                ],
            },
            "object_counts": {"laptop": 1},
        }

        dto = VideoSceneDTO.model_validate(raw_scene_data)
        self.assertIsNotNone(dto.detected_objects_json)
        self.assertEqual(len(dto.detected_objects_json.keyframe_bboxes), 1)
        self.assertEqual(dto.detected_objects_json.keyframe_bboxes[0].label, "laptop")
        self.assertEqual(dto.object_counts_json, {"laptop": 1})

        # Test camelCase dump
        dump = dto.model_dump(by_alias=True)
        self.assertIn("detectedObjectsJson", dump)
        self.assertIn("objectCountsJson", dump)

    def test_summary_dto_objects_summary(self):
        summary_data = {
            "summary_id": uuid.uuid4(),
            "video_id": self.video_id,
            "summary_text": "Overview of deep learning",
            "transcript_text": "Welcome to class",
            "model_used": "Groq Llama 3.1 8B",
            "processing_time": 15.2,
            "video_objects_summary": {
                "total_unique_tracks": 4,
                "label_distribution": {"person": 2, "laptop": 2},
                "most_frequent_objects": ["person", "laptop"],
                "sample_fps_used": 3,
            },
        }
        dto = SummaryDTO.model_validate(summary_data)
        self.assertIsNotNone(dto.objects_summary_json)
        self.assertEqual(dto.objects_summary_json.total_unique_tracks, 4)
        self.assertEqual(dto.objects_summary_json.label_distribution["person"], 2)

        dump = dto.model_dump(by_alias=True)
        self.assertIn("objectsSummaryJson", dump)

    def test_evaluation_schema_metrics(self):
        od_metrics = ODEvalMetricsDTO(
            od_precision=92.5,
            od_recall=88.0,
            od_map50=90.2,
            od_map50_95=71.4,
        )
        self.assertEqual(od_metrics.od_precision, 92.5)

        eval_dto = EvaluationDTO(
            eval_id="test-eval-1",
            video_id=str(self.video_id),
            wer=0.08,
            od_precision=92.5,
            od_recall=88.0,
            od_map50=90.2,
            od_map50_95=71.4,
        )
        self.assertEqual(eval_dto.od_map50, 90.2)

    # -------------------------------------------------------------------------
    # 2. Celery Synchronization Flow (sync_job_status)
    # -------------------------------------------------------------------------
    def test_sync_job_status_persists_od_fields(self):
        job = MagicMock()
        job.job_id = uuid.uuid4()
        job.video_id = self.video_id
        job.status = JobStatus.RUNNING

        celery_result = {
            "stage": "completed",
            "progress": 100,
            "duration": 60.0,
            "video_title": "Lecture on Computer Vision",
            "video_file_path": "videos/test.mp4",
            "summary": "CV lecture summary",
            "chapters": [],
            "keyframes": [],
            "key_takeaways": ["Takeaway 1"],
            "transcript_text": "CV transcript",
            "video_objects_summary": {
                "total_unique_tracks": 3,
                "label_distribution": {"person": 2, "book": 1},
                "most_frequent_objects": ["person", "book"],
                "sample_fps_used": 3,
            },
            "scenes": [
                {
                    "scene_index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 30.0,
                    "start_timecode": "00:00:00.000",
                    "end_timecode": "00:00:30.000",
                    "start_frame": 0,
                    "end_frame": 900,
                    "keyframe_path": "keyframes/1.jpg",
                    "keyframe_url": "https://pub.dev/1.jpg",
                    "caption": "Lecturer at whiteboard",
                    "script": "Hello class",
                    "detected_objects": {
                        "keyframe_bboxes": [
                            {"track_id": 1, "label": "person", "confidence": 0.94, "bbox_norm": [0.1, 0.1, 0.5, 0.9]}
                        ],
                        "tracks_summary": [
                            {"track_id": 1, "label": "person", "avg_confidence": 0.92, "first_seen_sec": 0.0, "last_seen_sec": 28.0, "frames_appeared": 80}
                        ],
                    },
                    "object_counts": {"person": 1},
                }
            ],
        }

        mock_async_res = MagicMock()
        mock_async_res.state = "SUCCESS"
        mock_async_res.result = celery_result

        mock_db = MagicMock()
        mock_video = MagicMock()
        mock_video.video_id = self.video_id
        mock_video.rag_status = "PENDING"

        def mock_query(model):
            mock_q = MagicMock()
            if model == Video:
                mock_q.filter.return_value.first.return_value = mock_video
                mock_q.filter.return_value.update.return_value = 1
            elif model == Summary:
                mock_q.filter.return_value.first.return_value = None
            elif model == VideoScene:
                mock_q.filter.return_value.delete.return_value = 1
            return mock_q

        mock_db.query.side_effect = mock_query

        added_entities = []
        mock_db.add.side_effect = lambda entity: added_entities.append(entity)

        with patch("celery.result.AsyncResult", return_value=mock_async_res), \
             patch("app.core.celery_app.celery_app.send_task") as mock_send_task:
            sync_job_status(job, db=mock_db)

        # Check Summary entity
        summaries_added = [e for e in added_entities if isinstance(e, Summary)]
        self.assertEqual(len(summaries_added), 1)
        summary = summaries_added[0]
        self.assertEqual(summary.objects_summary_json, celery_result["video_objects_summary"])

        # Check VideoScene entity
        scenes_added = [e for e in added_entities if isinstance(e, VideoScene)]
        self.assertEqual(len(scenes_added), 1)
        scene = scenes_added[0]
        self.assertEqual(scene.detected_objects_json, celery_result["scenes"][0]["detected_objects"])
        self.assertEqual(scene.object_counts_json, {"person": 1})

    # -------------------------------------------------------------------------
    # 3. Scene Retrieval & Filtering (get_video_scenes)
    # -------------------------------------------------------------------------
    def test_get_video_scenes_filtering_by_label_and_confidence(self):
        mock_video = Video(
            video_id=self.video_id,
            user_id=self.user_id,
            title="Object Detection Lecture",
        )

        scene1 = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=self.video_id,
            scene_index=1,
            start_seconds=0.0,
            end_seconds=10.0,
            start_timecode="00:00:00.000",
            end_timecode="00:00:10.000",
            start_frame=0,
            end_frame=300,
            keyframe_path="kf/1.jpg",
            keyframe_url="https://pub.dev/1.jpg",
            caption="Professor introducing",
            script="Hello",
            detected_objects_json={
                "keyframe_bboxes": [
                    {"track_id": 1, "label": "person", "confidence": 0.95, "bbox_norm": [0.1, 0.1, 0.4, 0.9]}
                ],
                "tracks_summary": [
                    {"track_id": 1, "label": "person", "avg_confidence": 0.93, "first_seen_sec": 0.0, "last_seen_sec": 10.0, "frames_appeared": 30}
                ],
            },
            object_counts_json={"person": 1},
        )

        scene2 = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=self.video_id,
            scene_index=2,
            start_seconds=10.0,
            end_seconds=25.0,
            start_timecode="00:00:10.000",
            end_timecode="00:00:25.000",
            start_frame=300,
            end_frame=750,
            keyframe_path="kf/2.jpg",
            keyframe_url="https://pub.dev/2.jpg",
            caption="Slide showing laptop code",
            script="Look at the screen",
            detected_objects_json={
                "keyframe_bboxes": [
                    {"track_id": 2, "label": "laptop", "confidence": 0.82, "bbox_norm": [0.2, 0.3, 0.8, 0.7]}
                ],
                "tracks_summary": [
                    {"track_id": 2, "label": "laptop", "avg_confidence": 0.80, "first_seen_sec": 10.0, "last_seen_sec": 24.0, "frames_appeared": 40}
                ],
            },
            object_counts_json={"laptop": 1},
        )

        mock_db = MagicMock()
        # Mock Video query
        mock_db.query.return_value.filter.return_value.first.return_value = mock_video
        # Mock VideoScene query
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [scene1, scene2]

        # 1. Unfiltered: returns all 2 scenes
        res_all = get_video_scenes(
            video_id=self.video_id,
            object_label=None,
            min_confidence=None,
            db=mock_db,
            current_user=self.regular_user,
        )
        self.assertTrue(res_all.success)
        self.assertEqual(len(res_all.data), 2)

        # 2. Filter by label="laptop": returns only scene 2
        res_laptop = get_video_scenes(
            video_id=self.video_id,
            object_label="laptop",
            min_confidence=None,
            db=mock_db,
            current_user=self.regular_user,
        )
        self.assertEqual(len(res_laptop.data), 1)
        self.assertEqual(res_laptop.data[0].scene_index, 2)

        # 3. Filter by label="PERSON" (case-insensitive): returns scene 1
        res_person = get_video_scenes(
            video_id=self.video_id,
            object_label="PERSON",
            min_confidence=None,
            db=mock_db,
            current_user=self.regular_user,
        )
        self.assertEqual(len(res_person.data), 1)
        self.assertEqual(res_person.data[0].scene_index, 1)

        # 4. Filter by label="laptop" and min_confidence=0.90 (laptop is 0.82): returns 0 scenes
        res_high_conf = get_video_scenes(
            video_id=self.video_id,
            object_label="laptop",
            min_confidence=0.90,
            db=mock_db,
            current_user=self.regular_user,
        )
        self.assertEqual(len(res_high_conf.data), 0)

        # 5. Filter by non-existent label="car": returns 0 scenes
        res_none = get_video_scenes(
            video_id=self.video_id,
            object_label="car",
            min_confidence=None,
            db=mock_db,
            current_user=self.regular_user,
        )
        self.assertEqual(len(res_none.data), 0)

    # -------------------------------------------------------------------------
    # 4. Video Objects Aggregation Endpoint (get_video_objects)
    # -------------------------------------------------------------------------
    def test_get_video_objects_aggregation(self):
        mock_video = Video(
            video_id=self.video_id,
            user_id=self.user_id,
            title="Video Objects Aggregation Test",
        )

        mock_summary = Summary(
            summary_id=uuid.uuid4(),
            video_id=self.video_id,
            summary_text="Text",
            transcript_text="Transcript",
            model_used="Model",
            processing_time=1.0,
            chapters_json=[],
            keyframes_json=[],
            objects_summary_json={
                "total_unique_tracks": 3,
                "label_distribution": {"person": 2, "cell phone": 1},
                "most_frequent_objects": ["person", "cell phone"],
                "sample_fps_used": 3,
            },
        )

        scene1 = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=self.video_id,
            scene_index=1,
            start_seconds=0.0,
            end_seconds=15.0,
            start_timecode="00:00:00.000",
            end_timecode="00:00:15.000",
            start_frame=0,
            end_frame=450,
            keyframe_path="kf/1.jpg",
            keyframe_url="https://pub.dev/1.jpg",
            caption="Scene 1",
            detected_objects_json={
                "keyframe_bboxes": [
                    {"track_id": 1, "label": "person", "confidence": 0.95, "bbox_norm": [0.1, 0.1, 0.4, 0.9]}
                ],
                "tracks_summary": [
                    {"track_id": 1, "label": "person", "avg_confidence": 0.94, "first_seen_sec": 0.0, "last_seen_sec": 14.0, "frames_appeared": 40}
                ],
            },
            object_counts_json={"person": 1},
        )

        scene2 = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=self.video_id,
            scene_index=2,
            start_seconds=15.0,
            end_seconds=30.0,
            start_timecode="00:00:15.000",
            end_timecode="00:00:30.000",
            start_frame=450,
            end_frame=900,
            keyframe_path="kf/2.jpg",
            keyframe_url="https://pub.dev/2.jpg",
            caption="Scene 2",
            detected_objects_json={
                "keyframe_bboxes": [
                    {"track_id": 1, "label": "person", "confidence": 0.90, "bbox_norm": [0.1, 0.1, 0.4, 0.9]},
                    {"track_id": 2, "label": "cell phone", "confidence": 0.75, "bbox_norm": [0.5, 0.5, 0.6, 0.7]},
                ],
                "tracks_summary": [
                    {"track_id": 1, "label": "person", "avg_confidence": 0.88, "first_seen_sec": 15.0, "last_seen_sec": 30.0, "frames_appeared": 40},
                    {"track_id": 2, "label": "cell phone", "avg_confidence": 0.72, "first_seen_sec": 16.0, "last_seen_sec": 22.0, "frames_appeared": 18},
                ],
            },
            object_counts_json={"person": 1, "cell phone": 1},
        )

        mock_db = MagicMock()
        def mock_query(model):
            mock_q = MagicMock()
            if model == Video:
                mock_q.filter.return_value.first.return_value = mock_video
            elif model == Summary:
                mock_q.filter.return_value.first.return_value = mock_summary
            elif model == VideoScene:
                mock_q.filter.return_value.order_by.return_value.all.return_value = [scene1, scene2]
            return mock_q
        mock_db.query.side_effect = mock_query

        res = get_video_objects(
            video_id=self.video_id,
            min_confidence=None,
            db=mock_db,
            current_user=self.regular_user,
        )

        self.assertTrue(res.success)
        dto: VideoObjectsDetailResponseDTO = res.data
        self.assertEqual(dto.total_unique_objects, 2)
        labels = [o.label for o in dto.objects]
        self.assertIn("person", labels)
        self.assertIn("cell phone", labels)

        # "person" appeared in 2 scenes
        person_obj = next(o for o in dto.objects if o.label == "person")
        self.assertEqual(person_obj.scenes_count, 2)
        self.assertEqual(len(person_obj.scenes), 2)
        self.assertEqual(person_obj.scenes[0].scene_index, 1)
        self.assertEqual(person_obj.scenes[1].scene_index, 2)

        # "cell phone" appeared in 1 scene
        phone_obj = next(o for o in dto.objects if o.label == "cell phone")
        self.assertEqual(phone_obj.scenes_count, 1)
        self.assertEqual(phone_obj.scenes[0].scene_index, 2)

    # -------------------------------------------------------------------------
    # 5. Database Model Columns
    # -------------------------------------------------------------------------
    def test_database_model_columns_exist(self):
        self.assertTrue(hasattr(VideoScene, "detected_objects_json"))
        self.assertTrue(hasattr(VideoScene, "object_counts_json"))
        self.assertTrue(hasattr(Summary, "objects_summary_json"))
        self.assertTrue(hasattr(EvaluationLog, "od_precision"))
        self.assertTrue(hasattr(EvaluationLog, "od_recall"))
        self.assertTrue(hasattr(EvaluationLog, "od_map50"))
        self.assertTrue(hasattr(EvaluationLog, "od_map50_95"))


if __name__ == "__main__":
    unittest.main()

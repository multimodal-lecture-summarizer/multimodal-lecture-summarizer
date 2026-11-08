"""Scene OCR contract: worker dict -> DB row -> camelCase API DTO."""

from __future__ import annotations

import unittest
import uuid
from unittest.mock import MagicMock, patch

from app.api.v1.jobs import sync_job_status
from app.core.constants import JobStatus
from app.models.summary import Summary
from app.models.video import Video, VideoScene
from app.schemas.video import VideoSceneDTO

BLOCK = {
    "line_index": 0,
    "text": "Gradient descent",
    "confidence": 0.9731,
    "bbox_norm": [0.12, 0.08, 0.61, 0.15],
    "polygon_norm": [[0.12, 0.08], [0.61, 0.08], [0.61, 0.15], [0.12, 0.15]],
}


def _scene_dict(**extra):
    base = {
        "scene_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "start_timecode": "00:00:00.000",
        "end_timecode": "00:00:30.000",
        "start_frame": 0,
        "end_frame": 900,
        "keyframe_path": "keyframes/1.jpg",
        "keyframe_url": "https://pub.dev/1.jpg",
        "caption": "Slide",
        "script": "Hello class",
    }
    base.update(extra)
    return base


class SceneOcrContractTests(unittest.TestCase):
    def test_dto_maps_worker_ocr_blocks_and_dumps_camel_case(self):
        raw = _scene_dict(
            scene_id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            ocr_text="Gradient descent",
            ocr_blocks=[BLOCK],
        )
        dto = VideoSceneDTO.model_validate(raw)
        self.assertEqual(dto.ocr_text, "Gradient descent")
        self.assertEqual(len(dto.ocr_blocks_json), 1)
        dump = dto.model_dump(by_alias=True)
        self.assertEqual(dump["ocrText"], "Gradient descent")
        block = dump["ocrBlocksJson"][0]
        self.assertEqual(block["lineIndex"], 0)
        self.assertEqual(block["bboxNorm"], BLOCK["bbox_norm"])
        self.assertEqual(block["polygonNorm"], BLOCK["polygon_norm"])

    def test_dto_accepts_legacy_row_without_ocr(self):
        row = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=uuid.uuid4(),
            **{k: v for k, v in _scene_dict().items()},
        )
        dto = VideoSceneDTO.model_validate(row, from_attributes=True)
        self.assertIsNone(dto.ocr_text)
        self.assertIsNone(dto.ocr_blocks_json)

    def _sync(self, scene):
        job = MagicMock()
        job.job_id = uuid.uuid4()
        job.video_id = uuid.uuid4()
        job.status = JobStatus.RUNNING
        result = {
            "stage": "completed", "progress": 100, "duration": 60.0,
            "video_title": "t", "video_file_path": "videos/t.mp4",
            "summary": "s", "chapters": [], "keyframes": [],
            "key_takeaways": [], "transcript_text": "x", "scenes": [scene],
        }
        res = MagicMock()
        res.state = "SUCCESS"
        res.result = result
        db = MagicMock()
        video = MagicMock()
        video.video_id = job.video_id
        video.rag_status = "PENDING"

        def query(model):
            q = MagicMock()
            if model == Video:
                q.filter.return_value.first.return_value = video
                q.filter.return_value.update.return_value = 1
            elif model == Summary:
                q.filter.return_value.first.return_value = None
            return q

        db.query.side_effect = query
        added = []
        db.add.side_effect = added.append
        with patch("celery.result.AsyncResult", return_value=res), \
             patch("app.core.celery_app.celery_app.send_task"):
            sync_job_status(job, db=db)
        return [e for e in added if isinstance(e, VideoScene)]

    def test_sync_persists_ocr_fields(self):
        scenes = self._sync(_scene_dict(ocr_text="Gradient descent", ocr_blocks=[BLOCK]))
        self.assertEqual(len(scenes), 1)
        self.assertEqual(scenes[0].ocr_text, "Gradient descent")
        self.assertEqual(scenes[0].ocr_blocks_json, [BLOCK])

    def test_sync_without_ocr_keys_stores_null(self):
        scenes = self._sync(_scene_dict())
        self.assertIsNone(scenes[0].ocr_text)
        self.assertIsNone(scenes[0].ocr_blocks_json)


if __name__ == "__main__":
    unittest.main()

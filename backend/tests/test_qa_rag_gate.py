"""Q&A stays closed until the RAG index is ready, and reindex only queues the stored-summary task."""

import unittest
import uuid
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.api.v1.qa import ask_question
from app.api.v1.videos import reindex_video_index
from app.core.constants import RagStatus, VideoStatus
from app.core.exceptions import ConflictException, NotFoundException
from app.models.summary import Summary
from app.models.video import Video
from app.schemas.qa import QAAskRequest
from app.schemas.video import VideoDTO


def _user(user_id):
    return SimpleNamespace(user_id=user_id)


def _video(user_id, rag_status):
    video = Video(
        video_id=uuid.uuid4(),
        user_id=user_id,
        language="vi",
        status=VideoStatus.DONE,
        rag_status=rag_status,
        title="Lecture",
        uploaded_at=datetime.utcnow(),
    )
    video.jobs = []
    return video


def _db_for_question(video):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = video
    return db


class QaRagGateTests(unittest.TestCase):
    def test_ask_returns_409_until_ready(self):
        user_id = uuid.uuid4()
        payload = QAAskRequest(question="Nội dung chính là gì?")
        for status in (RagStatus.PENDING, RagStatus.PROCESSING, RagStatus.FAILED):
            with self.subTest(status=status):
                video = _video(user_id, status)
                with patch("app.api.v1.qa.chromadb_service") as chroma:
                    with self.assertRaises(ConflictException) as caught:
                        ask_question(video.video_id, payload, _db_for_question(video), _user(user_id))
                self.assertEqual(caught.exception.status_code, 409)
                self.assertEqual(caught.exception.details["ragStatus"], status.value)
                chroma.query_similar_chunks_with_metadata.assert_not_called()

    def test_reindex_queues_stored_summary_task(self):
        user_id = uuid.uuid4()
        video = _video(user_id, RagStatus.FAILED)
        summary = Summary(video_id=video.video_id, summary_text="Đã lưu", keyframes_json=[])
        db = MagicMock()

        def query(model):
            result = MagicMock()
            if model is Video:
                result.filter.return_value.first.return_value = video
            elif model is Summary:
                result.filter.return_value.first.return_value = summary
            return result

        db.query.side_effect = query
        celery = MagicMock()
        with patch("app.core.celery_app.celery_app", celery):
            with patch.object(VideoDTO, "model_validate", return_value=MagicMock()):
                response = reindex_video_index(video.video_id, db, _user(user_id))

        self.assertTrue(response.success)
        self.assertEqual(video.rag_status, RagStatus.PROCESSING)
        celery.send_task.assert_called_once_with("ai_workers.reindex_video", args=[str(video.video_id)])

    def test_reindex_while_processing_does_not_queue_again(self):
        user_id = uuid.uuid4()
        video = _video(user_id, RagStatus.PROCESSING)
        db = MagicMock()

        def query(model):
            result = MagicMock()
            if model is Video:
                result.filter.return_value.first.return_value = video
            elif model is Summary:
                result.filter.return_value.first.return_value = Summary(video_id=video.video_id, summary_text="Đã lưu")
            return result

        db.query.side_effect = query
        celery = MagicMock()
        with patch("app.core.celery_app.celery_app", celery):
            with patch.object(VideoDTO, "model_validate", return_value=MagicMock()):
                response = reindex_video_index(video.video_id, db, _user(user_id))

        self.assertTrue(response.success)
        celery.send_task.assert_not_called()

    def test_reindex_without_summary_is_not_found(self):
        user_id = uuid.uuid4()
        video = _video(user_id, RagStatus.PENDING)
        db = MagicMock()

        def query(model):
            result = MagicMock()
            if model is Video:
                result.filter.return_value.first.return_value = video
            else:
                result.filter.return_value.first.return_value = None
            return result

        db.query.side_effect = query
        with self.assertRaises(NotFoundException):
            reindex_video_index(video.video_id, db, _user(user_id))

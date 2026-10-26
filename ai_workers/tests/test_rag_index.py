"""Phase-1 RAG index contract. No ASR, vision, or LLM models."""

from __future__ import annotations

import os
import sys
import tempfile
from unittest.mock import patch

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.chromadb import ChromaDBService
from ai_workers.modules.fusion.quality_postprocess import normalize_for_backend
from ai_workers.modules.summarization.summarizer import Summarizer, normalize_slides


class HashEmbedding(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            vector = [0.0] * 8
            for index, char in enumerate(text or ""):
                vector[index % 8] += (ord(char) % 31) / 31.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name() -> str:
        return "hash_rag_test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


def _close(service: ChromaDBService) -> None:
    client = getattr(service, "client", None)
    if client is not None:
        client.close()


def _service(tmp_dir: str) -> ChromaDBService:
    service = ChromaDBService()
    service.host = ""
    service.persist_dir = tmp_dir
    service.collection_name = "lecture_rag_phase1"
    service._embedding_function = HashEmbedding()
    service._initialized = False
    return service


def test_legacy_keyframe_keeps_ocr_caption_objects_and_url():
    summarizer = Summarizer()
    captured = []

    class RecordingChroma:
        def add_transcript_chunks(self, video_id, chunks, metadatas):
            captured.append((video_id, chunks, metadatas))
            return True

        def delete_transcript_chunks(self, video_id):
            return True

    slides = [{
        "timestamp": 5.0,
        "description": "Sơ đồ kiến trúc thị giác máy tính",
        "imageUrl": "https://cdn.example/kf.png",
        "ocr_text": "YOLO and ByteTrack",
        "object_counts_json": "{\"person\": 1, \"laptop\": 2}",
    }]
    utterances = [{"start": 0.0, "end": 10.0, "text": "Hôm nay học object detection."}]

    with patch("app.services.chromadb.chromadb_service", RecordingChroma()):
        assert summarizer.build_rag_index("vid-legacy", utterances, slides) is True

    chunks, metadatas = captured[0][1], captured[0][2]
    assert len(chunks) == 1
    text = chunks[0]
    assert "Văn bản trên slide: YOLO and ByteTrack" in text
    assert "Mô tả hình ảnh: Sơ đồ kiến trúc thị giác máy tính" in text
    assert "Đối tượng xuất hiện:" in text
    assert "1 person" in text
    assert "2 laptop" in text
    assert "Ảnh slide: https://cdn.example/kf.png" in text
    assert metadatas[0]["keyframe_url"] == "https://cdn.example/kf.png"
    assert metadatas[0]["start_seconds"] == 0.0
    assert metadatas[0]["end_seconds"] > metadatas[0]["start_seconds"]


def test_normalize_slides_reads_old_and_new_time_fields():
    slides = normalize_slides([
        {"timestamp": 0.0, "description": "Cũ"},
        {"start_seconds": 40.0, "end_seconds": 55.0, "caption": "Mới", "keyframe_url": "http://new"},
    ])
    assert slides[0]["start_seconds"] == 0.0
    assert slides[0]["end_seconds"] == 40.0
    assert slides[0]["caption"] == "Cũ"
    assert slides[1]["end_seconds"] == 55.0
    assert slides[1]["keyframe_url"] == "http://new"


def test_normalize_for_backend_keeps_both_keyframe_schemas():
    _, keyframes = normalize_for_backend([], [{
        "timestamp": 12.0,
        "description": "Slide text",
        "imageUrl": "http://img/a.png",
        "ocr_text": "OCR",
        "object_counts": {"person": 1},
    }])
    keyframe = keyframes[0]
    assert keyframe["timestamp"] == 12.0
    assert keyframe["start_seconds"] == 12.0
    assert keyframe["description"] == "Slide text"
    assert keyframe["caption"] == "Slide text"
    assert keyframe["imageUrl"] == "http://img/a.png"
    assert keyframe["keyframe_url"] == "http://img/a.png"
    assert keyframe["ocr_text"] == "OCR"
    assert keyframe["object_counts"] == {"person": 1}


def test_replace_index_twice_does_not_duplicate_or_keep_stale_chunks():
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = _service(tmp_dir)
        try:
            _assert_replace_is_stable(service)
        finally:
            _close(service)


def _assert_replace_is_stable(service: ChromaDBService) -> None:
    first = service.replace_transcript_chunks(
        "vid-1",
        ["[00:00] Lời giảng: một | Văn bản trên slide: OCR", "[00:25] Lời giảng: hai"],
        [
            {"video_id": "vid-1", "start_seconds": 0.0, "end_seconds": 25.0, "timecode": "00:00", "keyframe_url": "http://a"},
            {"video_id": "vid-1", "start_seconds": 25.0, "end_seconds": 50.0, "timecode": "00:25", "keyframe_url": "http://b"},
        ],
    )
    second = service.replace_transcript_chunks(
        "vid-1",
        ["[00:00] Lời giảng: một | Văn bản trên slide: OCR"],
        [{"video_id": "vid-1", "start_seconds": 0.0, "end_seconds": 25.0, "timecode": "00:00", "keyframe_url": "http://a"}],
    )
    assert first is True
    assert second is True
    assert service.count_transcript_chunks("vid-1") == 1

    hits = service.query_similar_chunks_with_metadata("vid-missing", "WhisperX PySceneDetect CLIP", limit=4)
    assert hits == []

    own_hits = service.query_similar_chunks("vid-1", "OCR", limit=4)
    assert own_hits
    assert any("OCR" in doc for doc in own_hits)
    assert all("WhisperX" not in doc for doc in own_hits)


def test_summarizer_replace_is_idempotent_for_the_same_evidence():
    utterances = [{"start": 0.0, "end": 8.0, "text": "Nội dung bài giảng."}]
    slides = [{
        "start_seconds": 0.0,
        "end_seconds": 20.0,
        "caption": "Mở đầu",
        "ocr_text": "Tiêu đề",
        "keyframe_url": "http://img/frame.png",
        "object_counts": {"laptop": 1},
    }]
    with tempfile.TemporaryDirectory() as tmp_dir:
        service = _service(tmp_dir)
        try:
            summarizer = Summarizer()
            with patch("app.services.chromadb.chromadb_service", service):
                assert summarizer.build_rag_index("vid-same", utterances, slides) is True
                assert summarizer.build_rag_index("vid-same", utterances, slides) is True
            assert service.count_transcript_chunks("vid-same") == 1
            document = service.query_similar_chunks("vid-same", "Tiêu đề", limit=2)[0]
            assert "Văn bản trên slide: Tiêu đề" in document
            assert "Mô tả hình ảnh: Mở đầu" in document
            assert "1 laptop" in document
            assert "Ảnh slide: http://img/frame.png" in document
        finally:
            _close(service)

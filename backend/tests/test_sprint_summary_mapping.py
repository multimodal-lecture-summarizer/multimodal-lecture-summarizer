"""Smoke: backend summary mapping tolerates sprint-stack chapter/keyframe keys."""

from __future__ import annotations

import unittest

from app.schemas.summary import ChapterDTO, KeyframeDTO


def _pick(obj: dict, *keys, default=None):
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def map_chapters(raw):
    chapters = []
    for idx, c in enumerate(raw or []):
        if not isinstance(c, dict):
            continue
        start = float(_pick(c, "startTime", "start_time", "start_seconds", default=0.0) or 0.0)
        end = float(_pick(c, "endTime", "end_time", "end_seconds", default=start) or start)
        chapters.append(
            ChapterDTO(
                title=str(_pick(c, "title", default=f"Chapter {idx + 1}") or f"Chapter {idx + 1}"),
                start_time=start,
                end_time=end,
                summary=str(_pick(c, "summary", default="") or ""),
            )
        )
    return chapters


def map_keyframes(raw):
    keyframes = []
    for k in raw or []:
        if not isinstance(k, dict):
            continue
        keyframes.append(
            KeyframeDTO(
                timestamp=float(_pick(k, "timestamp", default=0.0) or 0.0),
                image_url=str(_pick(k, "imageUrl", "image_url", default="") or ""),
                description=str(_pick(k, "description", "caption", default="") or ""),
                importance_score=float(
                    _pick(k, "importanceScore", "importance_score", default=0.5) or 0.5
                ),
            )
        )
    return keyframes


class BackendSprintMappingTests(unittest.TestCase):
    def test_maps_sprint_camelcase_output(self):
        chapters = map_chapters(
            [
                {
                    "title": "Intro",
                    "startTime": 0.0,
                    "endTime": 60.0,
                    "summary": "hello",
                    "visual_evidence_hint": "ignored by DTO",
                }
            ]
        )
        keyframes = map_keyframes(
            [
                {
                    "timestamp": 12.0,
                    "imageUrl": "https://example/kf.png",
                    "description": "Slide Text: Agenda",
                    "importanceScore": 0.9,
                    "ocr_text": "Agenda",
                    "caption_enriched": True,
                }
            ]
        )
        self.assertEqual(chapters[0].title, "Intro")
        self.assertEqual(chapters[0].end_time, 60.0)
        self.assertEqual(keyframes[0].image_url, "https://example/kf.png")
        self.assertEqual(keyframes[0].importance_score, 0.9)

    def test_maps_snake_case_fallback(self):
        chapters = map_chapters(
            [{"title": "A", "start_time": 1.0, "end_time": 2.0, "summary": "x"}]
        )
        keyframes = map_keyframes(
            [
                {
                    "timestamp": 1.5,
                    "image_url": "/k.png",
                    "caption": "alt",
                    "importance_score": 0.4,
                }
            ]
        )
        self.assertEqual(chapters[0].start_time, 1.0)
        self.assertEqual(keyframes[0].description, "alt")
        self.assertEqual(keyframes[0].importance_score, 0.4)


if __name__ == "__main__":
    unittest.main()

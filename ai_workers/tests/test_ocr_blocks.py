"""OCR block parsing and extraction in visual_v2 semantic analysis."""

from __future__ import annotations

import cv2
import numpy as np

from ai_workers.modules.visual_v2 import semantic
from ai_workers.modules.visual_v2.semantic import _parse_ocr_lines


def _line(poly, text, conf):
    return [poly, (text, conf)]


def test_parse_normalizes_and_builds_bbox():
    lines = [_line([[10, 20], [110, 20], [110, 40], [10, 40]], "Hello", 0.97314)]
    blocks = _parse_ocr_lines(lines, 200, 100)
    assert len(blocks) == 1
    b = blocks[0]
    assert b["text"] == "Hello"
    assert b["confidence"] == 0.9731
    assert b["bbox_norm"] == [0.05, 0.2, 0.55, 0.4]
    assert b["polygon_norm"][0] == [0.05, 0.2]
    assert b["line_index"] == 0


def test_parse_rotated_polygon_uses_min_max():
    lines = [_line([[50, 0], [100, 50], [50, 100], [0, 50]], "Rot", 0.9)]
    b = _parse_ocr_lines(lines, 100, 100)[0]
    assert b["bbox_norm"] == [0.0, 0.0, 1.0, 1.0]


def test_parse_clamps_out_of_bounds():
    lines = [_line([[-10, -5], [300, -5], [300, 500], [-10, 500]], "Edge", 0.8)]
    b = _parse_ocr_lines(lines, 100, 100)[0]
    assert b["bbox_norm"] == [0.0, 0.0, 1.0, 1.0]


def test_parse_skips_bad_entries_and_keeps_line_index():
    good = _line([[0, 0], [10, 0], [10, 10], [0, 10]], "ok", 0.9)
    lines = [
        _line([[0, 0], [10, 0], [10, 10], [0, 10]], "   ", 0.9),  # blank text
        ["broken"],  # malformed
        _line([[0, 0], [1, 1]], "two points", 0.9),  # bad polygon
        good,
    ]
    blocks = _parse_ocr_lines(lines, 100, 100)
    assert [b["text"] for b in blocks] == ["ok"]
    assert blocks[0]["line_index"] == 3


def test_parse_empty_and_limit():
    assert _parse_ocr_lines(None, 100, 100) == []
    assert _parse_ocr_lines([], 100, 100) == []
    assert _parse_ocr_lines([good_line := _line([[0, 0], [1, 0], [1, 1], [0, 1]], "a", 0.9)], 0, 100) == []
    many = [_line([[0, 0], [1, 0], [1, 1], [0, 1]], f"t{i}", 0.9) for i in range(10)]
    assert len(_parse_ocr_lines(many, 100, 100, max_blocks=3)) == 3


class _FakeOcr:
    def __init__(self, result=None, exc=None):
        self._result, self._exc = result, exc

    def ocr(self, path, cls=True):
        if self._exc:
            raise self._exc
        return self._result


def _png(tmp_path, name="kf.png", w=200, h=100):
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))
    return str(path)


def test_extract_keeps_legacy_text_and_adds_blocks(tmp_path, monkeypatch):
    result = [[
        _line([[0, 0], [100, 0], [100, 20], [0, 20]], "Title", 0.99),
        _line([[0, 30], [100, 30], [100, 50], [0, 50]], "Body", 0.8),
    ]]
    monkeypatch.setattr(semantic, "load_paddleocr", lambda: _FakeOcr(result))
    scenes = [{"keyframe_path": _png(tmp_path)}, {"keyframe_path": str(tmp_path / "missing.png")}]
    analyzer = semantic.SemanticAnalyzer.__new__(semantic.SemanticAnalyzer)
    analyzer.extract_ocr_paddleocr(scenes)
    assert scenes[0]["ocr_text"] == "Title | Body"
    assert [b["text"] for b in scenes[0]["ocr_blocks"]] == ["Title", "Body"]
    assert scenes[1]["ocr_text"] == "" and scenes[1]["ocr_blocks"] == []


def test_extract_handles_empty_result_and_engine_error(tmp_path, monkeypatch):
    analyzer = semantic.SemanticAnalyzer.__new__(semantic.SemanticAnalyzer)

    monkeypatch.setattr(semantic, "load_paddleocr", lambda: _FakeOcr([None]))
    scenes = [{"keyframe_path": _png(tmp_path)}]
    analyzer.extract_ocr_paddleocr(scenes)
    assert scenes[0]["ocr_text"] == "" and scenes[0]["ocr_blocks"] == []

    monkeypatch.setattr(semantic, "load_paddleocr", lambda: _FakeOcr(exc=RuntimeError("boom")))
    scenes = [{"keyframe_path": _png(tmp_path, "b.png")}]
    analyzer.extract_ocr_paddleocr(scenes)
    assert scenes[0]["ocr_text"] == "" and scenes[0]["ocr_blocks"] == []

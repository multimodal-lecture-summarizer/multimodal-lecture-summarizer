import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from ai_workers.modules.visual_v2 import semantic
from ai_workers.modules.visual_v2.resident_models import (
    CLIP_SLOT,
    FLORENCE_SLOT,
    OCR_SLOT,
    ResidentSlot,
)


@pytest.fixture(autouse=True)
def _clean_slots():
    for slot in (CLIP_SLOT, FLORENCE_SLOT, OCR_SLOT):
        slot.clear()
    yield
    for slot in (CLIP_SLOT, FLORENCE_SLOT, OCR_SLOT):
        slot.clear()


def test_slot_loads_once_and_can_be_cleared():
    slot, loads = ResidentSlot("x"), []
    first = slot.get_or_load(lambda: loads.append(1) or object())
    assert slot.get_or_load(lambda: loads.append(1) or object()) is first
    assert len(loads) == 1 and slot.loaded

    slot.clear()
    assert not slot.loaded
    slot.get_or_load(lambda: loads.append(1) or object())
    assert len(loads) == 2


def test_concurrent_requests_share_one_load():
    slot, loads, results = ResidentSlot("x"), [], []

    def slow():
        loads.append(1)
        time.sleep(0.1)
        return object()

    threads = [threading.Thread(target=lambda: results.append(slot.get_or_load(slow))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(loads) == 1
    assert all(r is results[0] for r in results)


def _fake_clip_classes():
    model = MagicMock()
    model.to.return_value = model
    processor_cls, model_cls = MagicMock(), MagicMock()
    processor_cls.from_pretrained.return_value = object()
    model_cls.from_pretrained.return_value = model
    return processor_cls, model_cls


def test_clip_is_reused_across_jobs_when_resident():
    processor_cls, model_cls = _fake_clip_classes()
    with patch.object(semantic, "CLIPProcessor", processor_cls), patch.object(semantic, "CLIPModel", model_cls), \
         patch.object(semantic, "visual_resident_enabled", return_value=True):
        first = semantic.SemanticAnalyzer().load_clip()
        second = semantic.SemanticAnalyzer().load_clip()

    assert first is second
    assert model_cls.from_pretrained.call_count == 1


def test_clip_is_reloaded_each_job_when_not_resident():
    processor_cls, model_cls = _fake_clip_classes()
    with patch.object(semantic, "CLIPProcessor", processor_cls), patch.object(semantic, "CLIPModel", model_cls), \
         patch.object(semantic, "visual_resident_enabled", return_value=False):
        semantic.SemanticAnalyzer().load_clip()
        semantic.SemanticAnalyzer().load_clip()

    assert model_cls.from_pretrained.call_count == 2


def test_paddleocr_engine_is_kept_only_when_resident():
    with patch.object(semantic, "_build_paddleocr", side_effect=lambda: object()) as build:
        with patch.object(semantic, "visual_resident_enabled", return_value=True):
            assert semantic.load_paddleocr() is semantic.load_paddleocr()
        assert build.call_count == 1

        OCR_SLOT.clear()
        with patch.object(semantic, "visual_resident_enabled", return_value=False):
            assert semantic.load_paddleocr() is not semantic.load_paddleocr()


def test_preload_does_nothing_when_residency_is_off():
    with patch.object(semantic, "visual_resident_enabled", return_value=False), \
         patch.object(semantic, "SemanticAnalyzer") as analyzer:
        semantic.preload_visual_models()

    analyzer.assert_not_called()

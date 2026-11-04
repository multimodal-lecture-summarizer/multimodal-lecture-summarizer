"""Keep CLIP, Florence-2 and PaddleOCR loaded between jobs.

Reloading them for every video costs tens of seconds on the visual branch. Each model lives in a
`ResidentSlot` (see ai_workers/core/resident_slot.py for the single-flight and lock-order rules).
"""

from __future__ import annotations

from ai_workers.core.config import flag_enabled, worker_settings
from ai_workers.core.resident_slot import ResidentSlot


def visual_resident_enabled() -> bool:
    return flag_enabled(worker_settings.VISUAL_KEEP_RESIDENT)


CLIP_SLOT = ResidentSlot("CLIP")
FLORENCE_SLOT = ResidentSlot("Florence-2")
OCR_SLOT = ResidentSlot("PaddleOCR")
DETECTOR_SLOT = ResidentSlot("RT-DETR")

"""Keep CLIP, Florence-2 and PaddleOCR loaded between jobs.

Reloading them for every video costs tens of seconds on the visual branch. Each model lives in a
slot whose loading is single-flight: a job that starts while the start-up preload is still
running waits for it instead of loading a second copy.

Lock order when several are needed (never the reverse): FLORENCE_INFERENCE_LOCK -> a slot's lock
-> MODEL_LOAD_LOCK.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from ai_workers.core.config import flag_enabled, worker_settings


def visual_resident_enabled() -> bool:
    return flag_enabled(worker_settings.VISUAL_KEEP_RESIDENT)


class ResidentSlot:
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.Lock()
        self._value: Any = None

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def get_or_load(self, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if self._value is None:
                started = time.perf_counter()
                self._value = loader()
                print(f"[Resident] {self.name} loaded in {time.perf_counter() - started:.1f}s and kept.", flush=True)
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None


CLIP_SLOT = ResidentSlot("CLIP")
FLORENCE_SLOT = ResidentSlot("Florence-2")
OCR_SLOT = ResidentSlot("PaddleOCR")

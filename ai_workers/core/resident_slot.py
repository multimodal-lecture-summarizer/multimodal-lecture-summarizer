"""A slot that keeps one loaded model alive between jobs, loading it at most once at a time.

A job that starts while the start-up preload is still running waits for it instead of loading a
second copy. Lock order when several are needed (never the reverse):
FLORENCE_INFERENCE_LOCK -> a slot's lock -> MODEL_LOAD_LOCK.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable


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

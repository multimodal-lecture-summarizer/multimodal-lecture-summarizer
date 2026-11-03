import threading
import time

from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK


def test_model_loads_do_not_overlap_across_threads():
    active = []
    overlaps = []

    def load():
        with MODEL_LOAD_LOCK:
            active.append(1)
            if len(active) > 1:
                overlaps.append(1)
            time.sleep(0.05)
            active.pop()

    threads = [threading.Thread(target=load) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps == []


def test_lock_is_reentrant_for_nested_loads():
    with MODEL_LOAD_LOCK:
        with MODEL_LOAD_LOCK:
            pass

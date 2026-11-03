"""Run independent pipeline branches concurrently, with a sequential fallback on GPU/RAM exhaustion."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Sequence


def is_resource_exhaustion(exc: BaseException) -> bool:
    """True for CUDA/host out-of-memory style failures that a sequential re-run may avoid."""
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    from ai_workers.modules.visual_v2.florence_runtime import FlorenceResourceError

    if isinstance(exc, FlorenceResourceError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "insufficient ram" in text


def run_branches(
    branches: Sequence[tuple[str, Callable[[], Any]]],
    *,
    parallel: bool,
    abort: threading.Event | None = None,
    on_fallback: Callable[[str, BaseException], None] | None = None,
    cleanup: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run each (name, fn) branch and return {name: result}.

    parallel=False runs them in order, exactly like inline code. With parallel=True every branch
    starts at once; if a branch dies of resource exhaustion it is re-run alone after the others
    finish. Any other exception propagates: the one that happened first in time. When `abort` is
    given it is set on the first such failure so sibling branches can stop at their next check.
    """
    if not parallel or len(branches) < 2:
        return {name: fn() for name, fn in branches}

    results: dict[str, Any] = {}
    errors: dict[str, BaseException] = {}
    first_fatal: list[BaseException] = []
    lock = threading.Lock()

    def guarded(fn: Callable[[], Any]) -> Callable[[], Any]:
        def run() -> Any:
            try:
                return fn()
            except BaseException as exc:  # noqa: BLE001 - recorded, re-raised via the future
                if not is_resource_exhaustion(exc):
                    with lock:
                        if not first_fatal:
                            first_fatal.append(exc)
                    if abort is not None:
                        abort.set()
                raise
        return run

    with ThreadPoolExecutor(max_workers=len(branches), thread_name_prefix="branch") as pool:
        futures = {name: pool.submit(guarded(fn)) for name, fn in branches}
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised or retried below
                errors[name] = exc

    if first_fatal:
        raise first_fatal[0]

    for name, fn in branches:
        exc = errors.get(name)
        if exc is None:
            continue
        if not is_resource_exhaustion(exc):
            raise exc
        if on_fallback is not None:
            on_fallback(name, exc)
        if cleanup is not None:
            cleanup(f"parallel branch {name} fallback")
        results[name] = fn()
    return results

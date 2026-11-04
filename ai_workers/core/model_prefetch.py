"""Download model weights into the local caches while the worker boots.

A fresh Colab runtime has empty caches, so the first job otherwise spends minutes downloading
Whisper, Qwen3-ASR, CLIP, PaddleOCR and LED inside the pipeline. This only downloads files; it
never loads a model, so it needs little RAM/VRAM and can overlap with worker start-up and with
the first job (the Hugging Face cache is lock-safe). Every task is best-effort.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from ai_workers.core.config import flag_enabled, worker_settings
from ai_workers.core.torch_load_patch import patch_transformers_torch_load_check

CLIP_REPO = "openai/clip-vit-base-patch32"
# Weight formats transformers never picks on this stack; skipping them roughly halves the download.
_SKIP_PATTERNS = ["*.h5", "*.msgpack", "flax_model*", "tf_model*", "rust_model*", "*.onnx", "*.ot"]


def prefetch_enabled() -> bool:
    return flag_enabled(worker_settings.PREFETCH_MODELS)


# huggingface_hub creates tqdm bars per download; started from several threads at once, the
# first bars race on tqdm's class lock ("type object 'tqdm' has no attribute '_lock'").
# Hub downloads therefore run one at a time; PaddleOCR (a different downloader) overlaps with them.
_HUB_DOWNLOAD_LOCK = threading.Lock()


def _snapshot(repo_id: str, **kwargs) -> None:
    from huggingface_hub import snapshot_download

    with _HUB_DOWNLOAD_LOCK:
        snapshot_download(repo_id, **kwargs)


def _prefetch_whisper() -> None:
    from faster_whisper.utils import download_model

    cache_dir = Path(worker_settings.CACHE_DIR)
    size = worker_settings.FASTER_WHISPER_MODEL or "small"
    if (cache_dir / f"faster-whisper-{size}" / "model.bin").exists():
        return  # AudioTranscriber already prefers this local copy
    cache_dir.mkdir(parents=True, exist_ok=True)
    with _HUB_DOWNLOAD_LOCK:
        download_model(size, cache_dir=str(cache_dir))


def _prefetch_qwen() -> None:
    _snapshot(worker_settings.QWEN_ASR_MODEL, ignore_patterns=_SKIP_PATTERNS)
    _snapshot(worker_settings.QWEN_ALIGNER_MODEL, ignore_patterns=_SKIP_PATTERNS)
    # With the weights local, load the model onto the GPU now (while the worker is idle) so the
    # first job does not pay the ~2 min load either. A job that starts meanwhile waits for it.
    from ai_workers.modules.audio_v2.qwen_transcriber import preload_qwen_model

    preload_qwen_model()


def _prefetch_clip() -> None:
    # The CLIP repo ships only pytorch_model.bin, so fetch whichever weights exist.
    _snapshot(CLIP_REPO, ignore_patterns=_SKIP_PATTERNS)
    # CLIP and Florence-2 (vendored, nothing to download) are loaded now and kept for every job.
    from ai_workers.modules.visual_v2.semantic import preload_visual_models

    preload_visual_models()


def _prefetch_led() -> None:
    _snapshot(
        worker_settings.LED_MODEL,
        revision=worker_settings.LED_MODEL_REVISION or None,
        ignore_patterns=_SKIP_PATTERNS,
    )
    # With the weights local, load LED onto the GPU now so no job pays for it.
    from ai_workers.modules.summarization.led_summarizer import preload_led_model

    preload_led_model()


def _prefetch_paddleocr() -> None:
    # Building the engine is what triggers PaddleOCR's own model download; with residency on the
    # engine is kept, otherwise it is discarded.
    from ai_workers.modules.visual_v2.semantic import load_paddleocr

    load_paddleocr()


def prefetch_tasks() -> list[tuple[str, Callable[[], None]]]:
    tasks: list[tuple[str, Callable[[], None]]] = [("whisper", _prefetch_whisper)]
    if worker_settings.ASR_ENGLISH_BACKEND == "qwen3":
        tasks.append(("qwen3-asr", _prefetch_qwen))
    tasks.append(("clip", _prefetch_clip))
    tasks.append(("paddleocr", _prefetch_paddleocr))
    if worker_settings.SUMMARIZER_ENGLISH_BACKEND == "led":
        tasks.append(("led", _prefetch_led))
    return tasks


def run_prefetch(log: Callable[[str], None] = print) -> dict[str, bool]:
    """Run every prefetch task (3 at a time); returns {name: succeeded}."""
    outcome: dict[str, bool] = {}
    # The start-up preload loads the same .bin checkpoints a job does, so it needs the same patch.
    patch_transformers_torch_load_check()

    def run(name: str, fn: Callable[[], None]) -> None:
        started = time.perf_counter()
        try:
            fn()
            outcome[name] = True
            log(f"[Prefetch] {name} ready in {time.perf_counter() - started:.1f}s.")
        except Exception as exc:  # noqa: BLE001 - the model is simply fetched on first use instead
            outcome[name] = False
            log(f"[Prefetch] {name} skipped after {time.perf_counter() - started:.1f}s: {exc!r}")

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="prefetch") as pool:
        for name, fn in prefetch_tasks():
            pool.submit(run, name, fn)
    log(f"[Prefetch] Finished in {time.perf_counter() - started:.1f}s: {outcome}")
    return outcome


def start_prefetch_thread(log: Callable[[str], None] = print) -> threading.Thread | None:
    """Start prefetching in the background; returns None when disabled."""
    if not prefetch_enabled():
        return None
    thread = threading.Thread(target=run_prefetch, args=(log,), name="model-prefetch", daemon=True)
    thread.start()
    return thread

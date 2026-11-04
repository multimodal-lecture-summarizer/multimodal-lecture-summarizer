"""English-only ASR with Qwen3-ASR plus Qwen3-ForcedAligner word timestamps.

Qwen3-ASR was benchmarked on English TED-LIUM only, and the forced aligner has no
Vietnamese support, so this engine is never used for `vi` or `auto` sources.
Output matches AudioTranscriber.transcribe() so Stage 5 timeline alignment is unchanged.
"""

from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ai_workers.core.config import worker_settings
from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber
from ai_workers.modules.visual_v2.florence_runtime import FLORENCE_INFERENCE_LOCK

logger = logging.getLogger("ai_workers.modules.audio_v2.qwen_transcriber")

SAMPLE_RATE = 16000
# The aligner accepts up to 5 minutes of speech per call; stay below with margin.
CHUNK_SECONDS = 240.0
# Cut at the quietest point inside this trailing window to avoid splitting words.
CUT_SEARCH_SECONDS = 10.0
CUT_FRAME_SECONDS = 0.2


def split_audio_chunks(audio: np.ndarray, sr: int = SAMPLE_RATE) -> list[tuple[float, np.ndarray]]:
    """Split audio into <= CHUNK_SECONDS pieces cut at low-energy points.

    Returns (offset_seconds, samples) pairs.
    """
    chunk_len = int(CHUNK_SECONDS * sr)
    if len(audio) <= chunk_len:
        return [(0.0, audio)]

    frame = int(CUT_FRAME_SECONDS * sr)
    search = int(CUT_SEARCH_SECONDS * sr)
    chunks: list[tuple[float, np.ndarray]] = []
    start = 0
    while start < len(audio):
        end = start + chunk_len
        if end >= len(audio):
            chunks.append((start / sr, audio[start:]))
            break
        window = audio[end - search:end]
        n_frames = len(window) // frame
        energy = np.abs(window[: n_frames * frame]).reshape(n_frames, frame).mean(axis=1)
        end = end - search + int(np.argmin(energy)) * frame + frame // 2
        chunks.append((start / sr, audio[start:end]))
        start = end
    return chunks


def pipeline_batch_size(chunk_count: int) -> int:
    """ASR batch size, capped so there are at least two batches.

    ASR on the first batch can then start while later chunks are still being denoised.
    """
    batch = max(1, worker_settings.QWEN_ASR_BATCH_SIZE)
    return max(1, min(batch, -(-chunk_count // 2)))


def build_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group timed words into segments using the same cut rules as the Whisper path."""
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for w in words:
        if current is not None:
            gap = w["start"] - current["words"][-1]["end"]
            if current["end"] - current["start"] > 15.0 and gap > 1.0:
                segments.append(current)
                current = None
        if current is None:
            current = {"start": w["start"], "end": w["end"], "text": w["word"], "words": []}
        else:
            current["text"] += " " + w["word"]
        current["words"].append(w)
        current["end"] = w["end"]

        dur = current["end"] - current["start"]
        if (dur > 5.0 and w["word"].endswith((".", "?", "!"))) or dur > 20.0:
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def validate_words(words: list[dict[str, Any]], text: str, duration: float) -> None:
    """Raise if aligner output is implausible, so the caller falls back to Whisper.

    Guards against a wrong timestamp unit or a missing aligner result, which would
    otherwise silently corrupt Stage 5 timeline alignment.
    """
    if text and not words:
        raise ValueError("Qwen3-ASR returned text but no word timestamps")
    starts = [w["start"] for w in words]
    if any(b < a for a, b in zip(starts, starts[1:])):
        raise ValueError("Qwen3-ASR word timestamps are not monotonic")
    if words and words[-1]["end"] > duration + 5.0:
        raise ValueError(
            f"Qwen3-ASR timestamps ({words[-1]['end']:.1f}) exceed audio duration ({duration:.1f}); unit mismatch?"
        )


class QwenEnglishTranscriber(AudioTranscriber):
    """Drop-in replacement for AudioTranscriber for English audio."""

    def _load_model(self) -> Any:
        started = time.perf_counter()
        import torch
        from qwen_asr import Qwen3ASRModel

        t_import = time.perf_counter() - started
        device_map = "cuda:0" if self.device == "cuda" else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading Qwen3-ASR ({worker_settings.QWEN_ASR_MODEL}) on {device_map}...")
        # Serialized with the visual branch's CLIP/Florence loads (see model_load_lock).
        waiting = time.perf_counter()
        with MODEL_LOAD_LOCK:
            t_wait = time.perf_counter() - waiting
            loading = time.perf_counter()
            model = Qwen3ASRModel.from_pretrained(
                worker_settings.QWEN_ASR_MODEL,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=max(1, worker_settings.QWEN_ASR_BATCH_SIZE),
                max_new_tokens=worker_settings.QWEN_ASR_MAX_NEW_TOKENS,
                forced_aligner=worker_settings.QWEN_ALIGNER_MODEL,
                forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
            )
        print(
            f"[ASR] Qwen3-ASR load: import {t_import:.1f}s, waited for load lock {t_wait:.1f}s, "
            f"from_pretrained (ASR + aligner) {time.perf_counter() - loading:.1f}s.",
            flush=True,
        )
        return model

    def _timed_load_model(self) -> tuple[Any, float]:
        started = time.perf_counter()
        model = self._load_model()
        return model, time.perf_counter() - started

    def transcribe(self, audio_path: str, source_language: str | None = None) -> dict[str, Any]:
        if source_language not in (None, "en"):
            raise ValueError(f"Qwen3-ASR engine is English-only, got source_language={source_language!r}")

        try:
            return self._transcribe_qwen(audio_path)
        except Exception:  # missing package, weights, OOM, implausible output
            logger.warning("Qwen3-ASR failed; falling back to Faster-Whisper", exc_info=True)
        # Fallback runs outside the except block so the failed model's frames can be freed first.
        gc.collect()
        fallback = AudioTranscriber({"model_name": worker_settings.FASTER_WHISPER_MODEL})
        return fallback.transcribe(audio_path, source_language="en")

    @classmethod
    def _transcribe_batch(cls, model: Any, batch: list[tuple[float, np.ndarray]]) -> list[Any]:
        # Florence-2 captioning switches on process-wide deterministic mode, under which the
        # aligner's CUDA cumsum raises. Inference therefore shares Florence's lock (the visual
        # branch only holds it while captioning, so this waits at most that long).
        with FLORENCE_INFERENCE_LOCK:
            return cls._run_batch(model, batch)

    @staticmethod
    def _run_batch(model: Any, batch: list[tuple[float, np.ndarray]]) -> list[Any]:
        """One forward pass for the batch; if batching is rejected, redo the chunks one by one."""
        if len(batch) > 1:
            try:
                results = model.transcribe(
                    audio=[(chunk, SAMPLE_RATE) for _offset, chunk in batch],
                    language=["English"] * len(batch),
                    return_time_stamps=True,
                )
                if len(results) == len(batch):
                    return list(results)
                logger.warning("Qwen3-ASR batch returned %d results for %d chunks", len(results), len(batch))
            except Exception:
                logger.warning("Qwen3-ASR batched call failed; retrying chunk by chunk", exc_info=True)
        return [
            model.transcribe(
                audio=(chunk, SAMPLE_RATE),
                language="English",
                return_time_stamps=True,
            )[0]
            for _offset, chunk in batch
        ]

    def _transcribe_qwen(self, audio_path: str) -> dict[str, Any]:
        started = time.perf_counter()
        raw = self.load_samples(audio_path)
        chunks = split_audio_chunks(raw)
        batch_size = pipeline_batch_size(len(chunks))

        # Three things overlap: the model loads, a worker denoises chunk by chunk (CPU), and this
        # thread runs ASR on each batch as soon as its chunks are denoised (GPU).
        ready: queue.Queue = queue.Queue()
        stop = threading.Event()
        denoise_seconds = [0.0]

        def denoise_chunks() -> None:
            began = time.perf_counter()
            parts: list[np.ndarray] = []
            try:
                for offset, chunk in chunks:
                    if stop.is_set():
                        return
                    clean = self.denoise_samples(chunk)
                    parts.append(clean)
                    ready.put((offset, clean))
                denoise_seconds[0] = time.perf_counter() - began
                if parts:
                    try:
                        self.remember_denoised(audio_path, np.concatenate(parts))
                    except Exception:  # cache is only an optimisation
                        logger.debug("Could not cache denoised audio", exc_info=True)
                ready.put(None)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the consuming thread
                ready.put(exc)

        words: list[dict[str, Any]] = []
        texts: list[str] = []
        done = 0
        asr_started = started
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="qwen-asr") as pool:
            load_future = pool.submit(self._timed_load_model)
            pool.submit(denoise_chunks)
            model = None
            try:
                model, t_load = load_future.result()
                asr_started = time.perf_counter()
                print(
                    f"[ASR] Model ready {asr_started - started:.1f}s after start "
                    f"(load {t_load:.1f}s); {len(chunks)} chunks, batch={batch_size}.",
                    flush=True,
                )
                pending: list[tuple[float, np.ndarray]] = []
                while True:
                    item = ready.get()
                    if isinstance(item, BaseException):
                        raise item
                    if item is not None:
                        pending.append(item)
                        if len(pending) < batch_size:
                            continue
                    if pending:
                        batch, pending = pending, []
                        results = self._transcribe_batch(model, batch)
                        for (offset, _chunk), result in zip(batch, results):
                            if result.text.strip():
                                texts.append(result.text.strip())
                            for ts in result.time_stamps or []:
                                word = ts.text.strip()
                                if word:
                                    # Timestamps are assumed to be seconds (README does not say; verify on a real clip).
                                    words.append({
                                        "word": word,
                                        "start": offset + float(ts.start_time),
                                        "end": offset + float(ts.end_time),
                                    })
                        done += len(batch)
                        print(
                            f"[ASR] Chunks {done}/{len(chunks)} done at {time.perf_counter() - started:.1f}s.",
                            flush=True,
                        )
                    if item is None:
                        break
            except BaseException:
                stop.set()  # let the denoise worker end at its next chunk
                raise
            finally:
                model = None
                gc.collect()
                if self.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

        print(
            f"[ASR] {len(raw) / SAMPLE_RATE:.0f}s of audio: denoise {denoise_seconds[0]:.1f}s, "
            f"ASR {time.perf_counter() - asr_started:.1f}s after model load, "
            f"all overlapped into {time.perf_counter() - started:.1f}s.",
            flush=True,
        )
        full_text = " ".join(texts)
        validate_words(words, full_text, len(raw) / SAMPLE_RATE)
        segments = build_segments(words)
        print(f"[OK] Completed Qwen3-ASR transcription. Text length: {len(full_text)} characters.")
        return {
            "text": full_text,
            "segments": segments,
            "language": "en",
            "source_language": "en",
            "asr_model": worker_settings.QWEN_ASR_MODEL,
        }

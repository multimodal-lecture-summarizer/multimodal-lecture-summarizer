"""English-only ASR with Qwen3-ASR plus Qwen3-ForcedAligner word timestamps.

Qwen3-ASR was benchmarked on English TED-LIUM only, and the forced aligner has no
Vietnamese support, so this engine is never used for `vi` or `auto` sources.
Output matches AudioTranscriber.transcribe() so Stage 5 timeline alignment is unchanged.
"""

from __future__ import annotations

import gc
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ai_workers.core.config import worker_settings
from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

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
        import torch
        from qwen_asr import Qwen3ASRModel

        device_map = "cuda:0" if self.device == "cuda" else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading Qwen3-ASR ({worker_settings.QWEN_ASR_MODEL}) on {device_map}...")
        # Serialized with the visual branch's CLIP/Florence loads (see model_load_lock).
        with MODEL_LOAD_LOCK:
            return Qwen3ASRModel.from_pretrained(
                worker_settings.QWEN_ASR_MODEL,
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=max(1, worker_settings.QWEN_ASR_BATCH_SIZE),
                max_new_tokens=worker_settings.QWEN_ASR_MAX_NEW_TOKENS,
                forced_aligner=worker_settings.QWEN_ALIGNER_MODEL,
                forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
            )

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

    @staticmethod
    def _transcribe_batch(model: Any, batch: list[tuple[float, np.ndarray]]) -> list[Any]:
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
        t0 = time.perf_counter()
        # Denoising (CPU) and loading the model (disk/GPU) don't depend on each other, so overlap them.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-load") as pool:
            load_future = pool.submit(self._timed_load_model)
            audio = self.reduce_noise(audio_path)
            t_denoise = time.perf_counter() - t0
            model, t_load = load_future.result()
        print(
            f"[ASR] Denoise {t_denoise:.1f}s, model load {t_load:.1f}s, "
            f"both finished after {time.perf_counter() - t0:.1f}s."
        )
        words: list[dict[str, Any]] = []
        texts: list[str] = []
        try:
            chunks = split_audio_chunks(audio)
            batch_size = max(1, worker_settings.QWEN_ASR_BATCH_SIZE)
            t0 = time.perf_counter()
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
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
                print(
                    f"[ASR] Chunks {min(i + batch_size, len(chunks))}/{len(chunks)} done "
                    f"(batch={len(batch)}) at {time.perf_counter() - t0:.1f}s.",
                    flush=True,
                )
            print(
                f"[ASR] Transcribed {len(audio) / SAMPLE_RATE:.0f}s of audio in "
                f"{time.perf_counter() - t0:.1f}s (batch_size={batch_size}).",
                flush=True,
            )
        finally:
            del model
            gc.collect()
            if self.device == "cuda":
                import torch
                torch.cuda.empty_cache()

        full_text = " ".join(texts)
        validate_words(words, full_text, len(audio) / SAMPLE_RATE)
        segments = build_segments(words)
        print(f"[OK] Completed Qwen3-ASR transcription. Text length: {len(full_text)} characters.")
        return {
            "text": full_text,
            "segments": segments,
            "language": "en",
            "source_language": "en",
            "asr_model": worker_settings.QWEN_ASR_MODEL,
        }

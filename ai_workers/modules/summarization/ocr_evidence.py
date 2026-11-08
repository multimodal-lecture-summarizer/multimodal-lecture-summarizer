"""Interleave slide OCR into the lecture transcript so LED sees what was on screen.

Pure text logic, no model imports. With no usable OCR the result is exactly the plain
transcript (" ".join of stripped utterance texts), so callers can fall back losslessly.
"""

from __future__ import annotations

import re
from typing import Any

MAX_OCR_SHARE = 0.20  # total OCR words / transcript words
MAX_WORDS_PER_SLIDE = 60
SLIDE_PREFIX = "Slide:"

_LINE_SPLIT = re.compile(r"\s*\|\s*|\r?\n")
_HAS_WORD = re.compile(r"[^\W\d_]{3}")  # a run of 3+ letters; drops page numbers and stray symbols
_TRAILING_PUNCT = ".:;,"


def split_ocr_lines(ocr_text: str | None) -> list[str]:
    """Split stored OCR text (' | ' joined, or newline separated) into whitespace-normalized lines."""
    if not ocr_text:
        return []
    lines = (" ".join(part.split()) for part in _LINE_SPLIT.split(str(ocr_text)))
    return [line for line in lines if line]


def _usable_lines(ocr_text: str | None) -> list[str]:
    cleaned = (line.rstrip(_TRAILING_PUNCT).strip() for line in split_ocr_lines(ocr_text))
    return [line for line in cleaned if _HAS_WORD.search(line)]


def _utterance_start(utterance: dict, fallback: float) -> float:
    for key in ("start", "start_seconds"):
        try:
            return float(utterance[key])
        except (KeyError, TypeError, ValueError):
            continue
    return fallback


def merge_speech_and_ocr(utterances: list[dict], slides: list[dict]) -> tuple[str, dict[str, Any]]:
    """Return (source, {"transcript_words", "ocr_words", "slides_used"}) for LED."""
    from ai_workers.modules.summarization.summarizer import normalize_slides

    spoken = [(u, (u.get("text") or "").strip()) for u in utterances]
    spoken = [(u, text) for u, text in spoken if text]
    transcript = " ".join(text for _, text in spoken)
    transcript_words = len(transcript.split())
    plain = (transcript, {"transcript_words": transcript_words, "ocr_words": 0, "slides_used": 0})

    # Whole-video dedup by casefolded line, first occurrence wins: handles repeated footers
    # and slides that build up across several keyframes.
    seen: set[str] = set()
    per_slide: list[tuple[float, list[str]]] = []
    for slide in normalize_slides(slides):
        fresh: list[str] = []
        for line in _usable_lines(slide["ocr_text"]):
            key = line.casefold()
            if key not in seen:
                seen.add(key)
                fresh.append(line)
        if fresh:
            per_slide.append((slide["start_seconds"], fresh))
    if not per_slide:
        return plain

    budget = int(transcript_words * MAX_OCR_SHARE)
    share = min(MAX_WORDS_PER_SLIDE, budget // len(per_slide))
    if share <= 0:
        return plain

    blocks: list[tuple[float, str]] = []
    ocr_words = 0
    for start, lines in per_slide:
        left = share - 1  # the "Slide:" prefix is a word too
        picked: list[str] = []
        for line in lines:
            count = len(line.split())
            if count <= left:
                picked.append(line)
                left -= count
        if picked:
            ocr_words += share - left  # lines plus prefix
            blocks.append((start, f"{SLIDE_PREFIX} {'. '.join(picked)}."))
    if not blocks:
        return plain

    parts: list[str] = []
    pending = iter(blocks)  # normalize_slides already sorted by start
    nxt = next(pending, None)
    last_start = 0.0
    for utterance, text in spoken:
        last_start = _utterance_start(utterance, last_start)
        while nxt is not None and nxt[0] <= last_start:
            parts.append(nxt[1])
            nxt = next(pending, None)
        parts.append(text)
    while nxt is not None:
        parts.append(nxt[1])
        nxt = next(pending, None)

    return " ".join(parts), {
        "transcript_words": transcript_words,
        "ocr_words": ocr_words,
        "slides_used": len(blocks),
    }

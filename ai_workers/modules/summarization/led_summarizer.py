"""LED map-reduce summarizer for English output.

Reproduces the benchmarked profile `led-ngram3` (allenai/led-large-16384-arxiv, fp16, 4096-token
encoder windows padded to a multiple of 1024, global attention on the first token, 4 beams,
no_repeat_ngram_size=3, 256-token map/reduce notes, 768-token final note). Input is the transcript with slide OCR interleaved by time
(LED_INCLUDE_OCR, on by default); with it off the input is the plain transcript the benchmark used.
Only the summary text comes from LED; title, takeaways and chapter summaries stay extractive.
Any failure returns None so the caller falls through to the LLM path.
"""

from __future__ import annotations

import datetime as dt
import gc
import logging
import math
import re
import time
from typing import Any, Callable

from ai_workers.core.config import flag_enabled, worker_settings
from ai_workers.core.model_load_lock import MODEL_LOAD_LOCK
from ai_workers.core.resident_slot import ResidentSlot

logger = logging.getLogger("ai_workers.modules.summarization.led_summarizer")

ENCODER_CAP = 4096
PADDING_MULTIPLE = 1024
SPECIAL_TOKENS = 2  # <s> ... </s>
PAYLOAD_CAP = ENCODER_CAP - SPECIAL_TOKENS
MAP_TOKENS = 256
REDUCE_TOKENS = 256
FINAL_TOKENS = 768
SEPARATOR = "\n\n"
MAX_DEPTH = 8
# Benchmark S/M/L word views (120/240) and the 350-word delivery cap.
WORD_CAPS = {"short": 120, "medium": 240, "detailed": 350}


def _say(message: str) -> None:
    """Worker console line; the Celery log setup does not surface this module's logger output."""
    print(f"[LED] {message}", flush=True)


class LedError(RuntimeError):
    """LED could not produce a usable summary."""


def sentence_prefix(text: str, word_cap: int) -> str:
    """Longest leading run of whole sentences within word_cap (first sentence is cut if too long)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    # LED often stops without a terminal EOS; drop the unterminated tail when a whole sentence exists.
    if len(sentences) > 1 and not re.search(r"[.!?][\"')\]]*$", sentences[-1]):
        sentences.pop()
    picked: list[str] = []
    used = 0
    for sentence in sentences:
        count = len(sentence.split())
        if used + count > word_cap:
            break
        picked.append(sentence)
        used += count
    if picked:
        return " ".join(picked)
    return " ".join(text.split()[:word_cap])


def led_map_reduce(
    source_ids: list[int],
    encode: Callable[[str], list[int]],
    generate: Callable[[list[int], int], str],
) -> str:
    """Summarize token ids, covering every source token. generate(ids, max_new_tokens) -> text."""
    if not source_ids:
        raise LedError("empty source")

    def note(ids: list[int], budget: int) -> tuple[str, list[int]]:
        text = generate(ids, budget).strip()
        if not text:
            raise LedError("empty generated note")
        return text, encode(text)

    if len(source_ids) <= PAYLOAD_CAP:
        return note(source_ids, FINAL_TOKENS)[0]

    notes = [
        note(source_ids[i:i + PAYLOAD_CAP], MAP_TOKENS)
        for i in range(0, len(source_ids), PAYLOAD_CAP)
    ]
    separator = encode(SEPARATOR)
    for _ in range(MAX_DEPTH):
        if len(notes) == 1:
            return note(notes[0][1], FINAL_TOKENS)[0]

        groups: list[list[tuple[str, list[int]]]] = []
        current: list[tuple[str, list[int]]] = []
        size = 0
        for item in notes:
            extra = len(item[1]) + (len(separator) if current else 0)
            if current and size + extra > PAYLOAD_CAP:
                groups.append(current)
                current, size = [], 0
                extra = len(item[1])
            current.append(item)
            size += extra
        if current:
            if len(current) == 1 and groups and len(groups[-1]) >= 3:
                current.insert(0, groups[-1].pop())
            # Deviation from the benchmark: it aborts when a lone tail cannot borrow a note
            # (previous group < 3); here the tail is reduced alone, which only adds one cheap call.
            groups.append(current)

        reduced = []
        for group in groups:
            payload: list[int] = []
            for position, item in enumerate(group):
                if position:
                    payload.extend(separator)
                payload.extend(item[1])
            reduced.append(note(payload, REDUCE_TOKENS))
        if len(reduced) >= len(notes):
            raise LedError("reduction does not shrink the note count")
        notes = reduced
    raise LedError("maximum reduction depth reached")


class LedModel:
    """Loads LED once and exposes encode/generate for led_map_reduce."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoTokenizer, LEDForConditionalGeneration

        self._torch = torch
        name = worker_settings.LED_MODEL
        revision = worker_settings.LED_MODEL_REVISION or None
        logger.info("Loading LED %s on cuda (fp16)", name)
        # Serialized with the other Transformers model loads (start-up preload runs several at once).
        with MODEL_LOAD_LOCK:
            self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
            self.model = (
                LEDForConditionalGeneration.from_pretrained(
                    name, revision=revision, dtype=torch.float16, attn_implementation="eager"
                )
                .to("cuda")
                .eval()
            )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def generate(self, ids: list[int], max_new_tokens: int) -> str:
        torch = self._torch
        wrapped = self.tokenizer.build_inputs_with_special_tokens(ids)
        padded = math.ceil(len(wrapped) / PADDING_MULTIPLE) * PADDING_MULTIPLE
        padding = padded - len(wrapped)
        input_ids = torch.tensor([wrapped + [self.tokenizer.pad_token_id] * padding], device="cuda")
        attention_mask = torch.tensor([[1] * len(wrapped) + [0] * padding], device="cuda")
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()


LED_SLOT = ResidentSlot("LED")


def led_resident_enabled() -> bool:
    return flag_enabled(worker_settings.LED_KEEP_RESIDENT)


def load_led() -> LedModel:
    """The LED model; kept on the GPU between jobs when residency is on."""
    return LED_SLOT.get_or_load(LedModel) if led_resident_enabled() else LedModel()


def preload_led_model() -> None:
    """Load LED ahead of the first job. No-op unless residency is on and a CUDA GPU exists."""
    if not led_resident_enabled() or worker_settings.SUMMARIZER_ENGLISH_BACKEND != "led":
        return
    import torch

    if torch.cuda.is_available():
        load_led()


def led_enabled(output_language: str | None) -> bool:
    return (
        worker_settings.SUMMARIZER_ENGLISH_BACKEND == "led"
        and (output_language or "").strip().lower() == "en"
    )


def build_led_result(
    utterances: list[dict],
    slides: list[dict],
    chapters: list[dict],
    job_id: str,
    output_language: str,
    length_key: str,
) -> dict[str, Any] | None:
    """Full summary result with an LED summary, or None to let the caller use the LLM path."""
    if not led_enabled(output_language):
        _say(
            f"Bỏ qua LED (backend={worker_settings.SUMMARIZER_ENGLISH_BACKEND}, "
            f"output_language={output_language!r}); chỉ dùng khi backend=led và output_language=en."
        )
        return None
    transcript = " ".join(
        (u.get("text") or "").strip() for u in utterances if (u.get("text") or "").strip()
    )
    if not transcript:
        _say("Bỏ qua LED: transcript rỗng.")
        return None

    try:
        import torch

        if not torch.cuda.is_available():
            _say("Bỏ qua LED: không có GPU CUDA, chuyển sang LLM.")
            return None
        _say(f"Đang chuẩn bị {worker_settings.LED_MODEL} để tóm tắt (job {job_id}).")
        load_started = time.perf_counter()
        led = load_led()
        load_seconds = time.perf_counter() - load_started
    except Exception as err:
        _say(f"LED không khả dụng, chuyển sang LLM: {err!r}")
        logger.warning("LED unavailable for job %s; using the LLM path", job_id, exc_info=True)
        return None

    generate_calls = [0]

    def counted_generate(ids: list[int], max_new_tokens: int) -> str:
        generate_calls[0] += 1
        return led.generate(ids, max_new_tokens)

    source, ocr_words = transcript, 0
    if worker_settings.LED_INCLUDE_OCR:
        try:
            from ai_workers.modules.summarization.ocr_evidence import merge_speech_and_ocr

            source, stats = merge_speech_and_ocr(utterances, slides)
            ocr_words = stats["ocr_words"]
            _say(
                f"Đầu vào LED: {stats['transcript_words']} từ lời giảng, "
                f"{ocr_words} từ OCR từ {stats['slides_used']} slide."
            )
        except Exception as err:
            source, ocr_words = transcript, 0
            _say(f"Không ghép được OCR, dùng transcript: {err!r}")

    failed = False
    generate_started = time.perf_counter()
    try:
        raw = led_map_reduce(led.encode(source), led.encode, counted_generate)
    except Exception as err:
        failed = True
        _say(f"LED tóm tắt thất bại, chuyển sang LLM: {err!r}")
        logger.warning("LED summarization failed for job %s; using the LLM path", job_id, exc_info=True)
        return None
    finally:
        # A kept model survives a good run; after a failure it is dropped and reloaded next time.
        if failed or not led_resident_enabled():
            LED_SLOT.clear()
            try:
                led.close()
            except Exception:  # a CUDA error after OOM must not block the LLM fallback
                logger.warning("Closing LED failed for job %s", job_id, exc_info=True)
        del led
    generate_seconds = time.perf_counter() - generate_started
    _say(
        f"LED: model sẵn sàng sau {load_seconds:.1f}s, sinh {generate_calls[0]} lần trong {generate_seconds:.1f}s."
    )

    summary = sentence_prefix(raw, WORD_CAPS.get(length_key, WORD_CAPS["medium"]))
    if not summary:
        _say("LED trả về bản tóm tắt rỗng, chuyển sang LLM.")
        return None
    _say(f"LED hoàn tất: {len(summary.split())} từ (mức {length_key}).")

    from ai_workers.modules.summarization.extractive_fallback import ExtractiveSummarizer

    try:
        result = ExtractiveSummarizer().generate_fallback(
            utterances=utterances,
            slides=slides,
            chapters=chapters,
            llm_error=None,
            job_id=job_id,
            output_language="en",
            summary_length=length_key,
        )
    except Exception:
        logger.warning("Extractive structure failed for job %s; using the LLM path", job_id, exc_info=True)
        return None
    if result.get("status") == "failed":
        return None

    word_count = len(summary.split())
    reading_time_sec = max(10, int((word_count / 200) * 60))
    model_used = f"LED ({worker_settings.LED_MODEL}, map-reduce{' + OCR' if ocr_words > 0 else ''})"
    result.update(
        {
            "summary": summary,
            "executive_summary": summary,
            "model_used": model_used,
            "fallback_used": False,
            "summary_method": "led",
            "structure_method": "extractive",
            "word_count": word_count,
            "reading_time_sec": reading_time_sec,
            "llm_error": None,
            "summary_variants": {
                length_key: {
                    "summary_text": summary,
                    "key_takeaways_json": result.get("key_takeaways", []),
                    "chapters_json": result.get("chapters", []),
                    "word_count": word_count,
                    "reading_time_sec": reading_time_sec,
                    "model_used": model_used,
                    "summary_length": length_key,
                    "generated_at": dt.datetime.utcnow().isoformat(),
                }
            },
        }
    )
    return result

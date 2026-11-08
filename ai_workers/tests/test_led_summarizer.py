from unittest.mock import patch

import pytest

from ai_workers.core.config import worker_settings
from ai_workers.modules.summarization import led_summarizer as led
from ai_workers.modules.summarization.led_summarizer import (
    FINAL_TOKENS,
    MAP_TOKENS,
    PAYLOAD_CAP,
    REDUCE_TOKENS,
    LedError,
    led_map_reduce,
    sentence_prefix,
)


def _encode(text):
    return text.split()  # one token per whitespace-separated word; separator "\n\n" -> no tokens


def _recorder(note_words=200):
    calls = []

    def generate(ids, budget):
        calls.append((len(ids), budget))
        return " ".join(["w"] * note_words) + "."

    return calls, generate


def test_sentence_prefix_keeps_whole_sentences():
    text = "One two three. Four five six seven. Eight nine."
    assert sentence_prefix(text, 7) == "One two three. Four five six seven."
    assert sentence_prefix(text, 3) == "One two three."


def test_sentence_prefix_drops_unterminated_tail():
    assert sentence_prefix("One two three. Four five six seven eight", 20) == "One two three."
    assert sentence_prefix("only a fragment", 20) == "only a fragment"


def test_sentence_prefix_cuts_oversized_first_sentence():
    assert sentence_prefix("a b c d e f g h", 3) == "a b c"


def test_short_source_is_a_single_final_pass():
    calls, generate = _recorder()
    led_map_reduce(list(range(1000)), _encode, generate)
    assert calls == [(1000, FINAL_TOKENS)]


def test_long_source_maps_every_token_then_reduces_then_finalizes():
    calls, generate = _recorder()
    source = list(range(PAYLOAD_CAP * 2 + 1812))
    led_map_reduce(source, _encode, generate)
    assert sum(n for n, _ in calls[:3]) == len(source)  # the 3 map windows cover every token once
    assert [b for _, b in calls] == [MAP_TOKENS] * 3 + [REDUCE_TOKENS, FINAL_TOKENS]


def test_many_notes_need_two_reduction_layers():
    calls, generate = _recorder(note_words=200)
    led_map_reduce(list(range(PAYLOAD_CAP * 25)), _encode, generate)
    # 25 map notes of 200 tokens: 20 + 5 per reduce group, then the 2 results merge, then the final pass.
    assert len(calls) == 25 + 2 + 1 + 1
    assert [n for n, _ in calls[25:]] == [20 * 200, 5 * 200, 2 * 200, 200]
    assert calls[-1][1] == FINAL_TOKENS
    assert all(b == REDUCE_TOKENS for _, b in calls[25:-1])


def test_reduction_that_does_not_shrink_fails():
    _, generate = _recorder(note_words=PAYLOAD_CAP - 10)
    with pytest.raises(LedError):
        led_map_reduce(list(range(PAYLOAD_CAP * 3)), _encode, generate)


def test_empty_source_and_empty_note_fail():
    with pytest.raises(LedError):
        led_map_reduce([], _encode, lambda ids, b: "x")
    with pytest.raises(LedError):
        led_map_reduce([1, 2, 3], _encode, lambda ids, b: "   ")


class _FakeLed:
    def __init__(self, text="Alpha beta gamma. Delta epsilon. Zeta eta theta iota kappa."):
        self.text = text
        self.closed = False

    def encode(self, text):
        return _encode(text)

    def generate(self, ids, budget):
        return self.text

    def close(self):
        self.closed = True


UTTERANCES = [
    {"start": 0.0, "end": 5.0, "text": "Today we study gradient descent and why it converges."},
    {"start": 5.0, "end": 12.0, "text": "The learning rate controls the step size of every update."},
    {"start": 12.0, "end": 20.0, "text": "Too large a rate makes the loss diverge instead of shrinking."},
]
CHAPTERS = [{"title": "Intro", "startTime": 0.0, "endTime": 20.0}]


def _build(length_key="medium", output_language="en"):
    return led.build_led_result(
        UTTERANCES, [], CHAPTERS, job_id="j", output_language=output_language, length_key=length_key
    )


@pytest.fixture
def led_on(monkeypatch):
    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "led")


def test_disabled_backend_or_vietnamese_output_returns_none(monkeypatch):
    assert _build() is None  # conftest default is "llm"
    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "led")
    assert _build(output_language="vi") is None


def test_no_cuda_returns_none(led_on):
    with patch("torch.cuda.is_available", return_value=False):
        assert _build() is None


def test_success_builds_full_schema_with_led_summary(led_on):
    fake = _FakeLed()
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "LedModel", return_value=fake):
        result = _build(length_key="short")
    assert result["summary"] == "Alpha beta gamma. Delta epsilon. Zeta eta theta iota kappa."
    assert result["summary_method"] == "led" and result["fallback_used"] is False
    assert result["structure_method"] == "extractive"
    assert "LED" in result["model_used"]
    for key in ("video_title", "key_takeaways", "chapters", "word_count", "reading_time_sec", "summary_variants"):
        assert key in result
    assert result["summary_variants"]["short"]["summary_text"] == result["summary"]
    assert fake.closed


def test_summary_respects_tier_word_cap(led_on):
    long_text = ". ".join(["word " * 9 + "end"] * 40) + "."  # 10-word sentences
    with patch("torch.cuda.is_available", return_value=True), \
         patch.object(led, "LedModel", return_value=_FakeLed(long_text)):
        result = _build(length_key="short")
    assert 0 < len(result["summary"].split()) <= led.WORD_CAPS["short"]


def test_model_load_or_generation_failure_returns_none(led_on):
    with patch("torch.cuda.is_available", return_value=True), \
         patch.object(led, "LedModel", side_effect=RuntimeError("out of memory")):
        assert _build() is None

    broken = _FakeLed()
    broken.generate = lambda ids, budget: (_ for _ in ()).throw(RuntimeError("oom"))
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "LedModel", return_value=broken):
        assert _build() is None
    assert broken.closed


def test_close_failure_still_returns_summary_path_not_exception(led_on):
    flaky = _FakeLed()
    flaky.generate = lambda ids, budget: (_ for _ in ()).throw(RuntimeError("oom"))
    flaky.close = lambda: (_ for _ in ()).throw(RuntimeError("cuda error"))
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "LedModel", return_value=flaky):
        assert _build() is None  # caller falls back to the LLM instead of crashing


def test_structure_failure_after_led_returns_none(led_on):
    with patch("torch.cuda.is_available", return_value=True), \
         patch.object(led, "LedModel", return_value=_FakeLed()), \
         patch("ai_workers.modules.summarization.extractive_fallback.ExtractiveSummarizer.generate_fallback",
               side_effect=RuntimeError("boom")):
        assert _build() is None


def test_summarizer_returns_led_result_when_available():
    from ai_workers.modules.summarization.summarizer import Summarizer

    sentinel = {"summary": "from led", "summary_method": "led"}
    with patch.object(led, "build_led_result", return_value=sentinel):
        out = Summarizer().summarize(UTTERANCES, [], CHAPTERS, job_id="j", output_language="en")
    assert out is sentinel


class _RecordingLed(_FakeLed):
    def __init__(self):
        super().__init__()
        self.sources = []

    def encode(self, text):
        self.sources.append(text)
        return super().encode(text)


SLIDES = [{"start_seconds": 5.0, "ocr_text": "Gradient descent notes"}]
PLAIN = " ".join(u["text"] for u in UTTERANCES)


def _build_with(slides, utterances=UTTERANCES):
    fake = _RecordingLed()
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "LedModel", return_value=fake):
        result = led.build_led_result(
            utterances, slides, CHAPTERS, job_id="j", output_language="en", length_key="medium"
        )
    return result, fake


def test_ocr_enabled_interleaves_slides_and_marks_model_used(led_on, monkeypatch):
    monkeypatch.setattr(worker_settings, "LED_INCLUDE_OCR", True)
    result, fake = _build_with(SLIDES)
    assert "Slide: Gradient descent notes." in fake.sources[0]
    assert result["model_used"].endswith("+ OCR)")


def test_ocr_disabled_feeds_plain_transcript(led_on, monkeypatch):
    monkeypatch.setattr(worker_settings, "LED_INCLUDE_OCR", False)
    result, fake = _build_with(SLIDES)
    assert fake.sources[0] == PLAIN
    assert "+ OCR" not in result["model_used"]


def test_slides_without_ocr_feed_plain_transcript(led_on, monkeypatch):
    monkeypatch.setattr(worker_settings, "LED_INCLUDE_OCR", True)
    result, fake = _build_with([{"start_seconds": 5.0, "ocr_text": None}])
    assert fake.sources[0] == PLAIN
    assert "+ OCR" not in result["model_used"]


def test_merge_failure_keeps_led_on_transcript(led_on, monkeypatch):
    monkeypatch.setattr(worker_settings, "LED_INCLUDE_OCR", True)
    with patch("ai_workers.modules.summarization.ocr_evidence.merge_speech_and_ocr", side_effect=ValueError("boom")):
        result, fake = _build_with(SLIDES)
    assert fake.sources[0] == PLAIN
    assert result is not None and "+ OCR" not in result["model_used"]


def test_empty_transcript_with_ocr_returns_none(led_on, monkeypatch):
    monkeypatch.setattr(worker_settings, "LED_INCLUDE_OCR", True)
    result, fake = _build_with(SLIDES, utterances=[{"start": 0.0, "text": "  "}])
    assert result is None and fake.sources == []

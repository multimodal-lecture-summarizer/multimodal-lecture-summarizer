from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ai_workers.modules.audio_v2 import factory
from ai_workers.modules.audio_v2.qwen_transcriber import (
    CHUNK_SECONDS,
    SAMPLE_RATE,
    QwenEnglishTranscriber,
    build_segments,
    split_audio_chunks,
)
from ai_workers.modules.audio_v2.transcriber import AudioTranscriber


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_short_audio_is_single_chunk():
    audio = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    assert split_audio_chunks(audio) == [(0.0, audio)] or len(split_audio_chunks(audio)) == 1


def test_long_audio_chunks_cover_everything_and_cut_in_silence():
    total = int(CHUNK_SECONDS * 2.5 * SAMPLE_RATE)
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, total).astype(np.float32)
    quiet_at = int((CHUNK_SECONDS - 4) * SAMPLE_RATE)
    audio[quiet_at:quiet_at + SAMPLE_RATE] = 0.0

    chunks = split_audio_chunks(audio)

    assert sum(len(c) for _, c in chunks) == total
    assert all(len(c) <= CHUNK_SECONDS * SAMPLE_RATE for _, c in chunks)
    first_cut = len(chunks[0][1])
    assert quiet_at <= first_cut <= quiet_at + SAMPLE_RATE
    assert chunks[1][0] == pytest.approx(first_cut / SAMPLE_RATE)


def test_build_segments_cuts_on_punctuation_after_five_seconds():
    words = [_w("hello", 0.0, 1.0), _w("there", 1.0, 4.0), _w("friend.", 4.0, 6.0), _w("next", 6.5, 7.0)]
    segs = build_segments(words)
    assert [s["text"] for s in segs] == ["hello there friend.", "next"]
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 6.0
    assert len(segs[0]["words"]) == 3


def test_build_segments_hard_cut_after_twenty_seconds():
    words = [_w(f"w{i}", i * 2.0, i * 2.0 + 2.0) for i in range(12)]
    segs = build_segments(words)
    assert len(segs) >= 2
    assert all(s["end"] - s["start"] <= 22.0 for s in segs)


def test_build_segments_empty():
    assert build_segments([]) == []


def test_transcribe_offsets_chunk_timestamps():
    fake_result = SimpleNamespace(
        text="hi there.",
        time_stamps=[
            SimpleNamespace(text="hi", start_time=0.5, end_time=0.9),
            SimpleNamespace(text="there.", start_time=1.0, end_time=1.5),
        ],
    )
    fake_model = SimpleNamespace(transcribe=lambda **kw: [fake_result])
    audio = np.zeros(int(CHUNK_SECONDS * 1.5 * SAMPLE_RATE), dtype=np.float32)

    with patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")), \
         patch.object(QwenEnglishTranscriber, "load_samples", return_value=audio), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", side_effect=lambda s: s), \
         patch.object(QwenEnglishTranscriber, "_load_model", return_value=fake_model):
        out = QwenEnglishTranscriber().transcribe("x.wav", "en")

    starts = [w["start"] for s in out["segments"] for w in s["words"]]
    assert starts[0] == 0.5
    assert starts[2] > CHUNK_SECONDS * 0.9  # second chunk shifted by its offset
    assert out["language"] == "en"


def test_qwen_failure_falls_back_to_whisper():
    whisper_result = {"text": "ok", "segments": [], "language": "en", "asr_model": "small"}
    with patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")), \
         patch.object(QwenEnglishTranscriber, "_load_model", side_effect=ImportError("qwen_asr missing")), \
         patch.object(QwenEnglishTranscriber, "load_samples", return_value=np.zeros(10, dtype=np.float32)), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", side_effect=lambda s: s), \
         patch.object(AudioTranscriber, "transcribe", return_value=whisper_result) as whisper:
        out = QwenEnglishTranscriber().transcribe("x.wav", "en")
    assert out == whisper_result
    whisper.assert_called_once_with("x.wav", source_language="en")


def test_validate_words_rejects_implausible_output():
    from ai_workers.modules.audio_v2.qwen_transcriber import validate_words

    validate_words([_w("a", 0, 1), _w("b", 1, 2)], "a b", duration=10)
    validate_words([], "", duration=10)
    with pytest.raises(ValueError):
        validate_words([], "text without words", duration=10)
    with pytest.raises(ValueError):
        validate_words([_w("a", 5, 6), _w("b", 1, 2)], "a b", duration=10)
    with pytest.raises(ValueError):  # milliseconds mistaken for seconds
        validate_words([_w("a", 500, 900)], "a", duration=10)


def test_implausible_timestamps_trigger_whisper_fallback():
    bad = SimpleNamespace(text="hi", time_stamps=[SimpleNamespace(text="hi", start_time=500, end_time=900)])
    fake_model = SimpleNamespace(transcribe=lambda **kw: [bad])
    whisper_result = {"text": "hi", "segments": [], "language": "en", "asr_model": "small"}
    with patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")), \
         patch.object(QwenEnglishTranscriber, "load_samples", return_value=np.zeros(SAMPLE_RATE * 5, dtype=np.float32)), \
         patch.object(QwenEnglishTranscriber, "denoise_samples", side_effect=lambda s: s), \
         patch.object(QwenEnglishTranscriber, "_load_model", return_value=fake_model), \
         patch.object(AudioTranscriber, "transcribe", return_value=whisper_result):
        assert QwenEnglishTranscriber().transcribe("x.wav", "en") == whisper_result


def test_asr_identifier_per_language_and_backend():
    whisper = factory.worker_settings.FASTER_WHISPER_MODEL
    with patch.object(factory.worker_settings, "ASR_ENGLISH_BACKEND", "qwen3"):
        en_qwen = factory.asr_identifier("en")
        assert "Qwen3-ASR" in en_qwen
        assert factory.asr_identifier("vi") == whisper
        auto_id = factory.asr_identifier("auto")
        assert whisper in auto_id and en_qwen in auto_id  # covers both possible engines
    with patch.object(factory.worker_settings, "ASR_ENGLISH_BACKEND", "whisper"):
        assert factory.asr_identifier("en") == whisper
        assert factory.asr_identifier("auto") == whisper


def _routing(detected, probability=0.95):
    from ai_workers.modules.audio_v2.routing_transcriber import LanguageRoutingTranscriber

    router = LanguageRoutingTranscriber.__new__(LanguageRoutingTranscriber)
    if isinstance(detected, Exception):
        router.detect_language = lambda path: (_ for _ in ()).throw(detected)
    else:
        router.detect_language = lambda path: (detected, probability)
    return router


@pytest.mark.parametrize(
    "detected,probability,expect_qwen",
    [("en", 0.95, True), ("en", 0.5, False), ("vi", 0.99, False), ("fr", 0.9, False)],
)
def test_auto_routes_by_detected_language(detected, probability, expect_qwen):
    whisper_result = {"text": "w", "segments": [], "language": detected, "asr_model": "small"}
    qwen_result = {"text": "q", "segments": [], "language": "en", "asr_model": "Qwen"}
    with patch.object(QwenEnglishTranscriber, "transcribe", return_value=qwen_result) as qwen, \
         patch.object(AudioTranscriber, "transcribe", return_value=whisper_result) as whisper:
        out = _routing(detected, probability).transcribe("x.wav", "auto")
    assert (out is qwen_result) is expect_qwen
    assert qwen.called is expect_qwen
    assert whisper.called is (not expect_qwen)


def test_auto_detection_failure_uses_whisper():
    whisper_result = {"text": "w", "segments": [], "language": "en", "asr_model": "small"}
    with patch.object(QwenEnglishTranscriber, "transcribe") as qwen, \
         patch.object(AudioTranscriber, "transcribe", return_value=whisper_result):
        out = _routing(RuntimeError("no audio")).transcribe("x.wav", "auto")
    assert out is whisper_result
    qwen.assert_not_called()


def test_factory_builds_router_for_auto_only():
    from ai_workers.modules.audio_v2.routing_transcriber import LanguageRoutingTranscriber

    with patch.object(factory.worker_settings, "ASR_ENGLISH_BACKEND", "qwen3"), \
         patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")):
        assert isinstance(factory.build_transcriber("auto"), LanguageRoutingTranscriber)
        assert type(factory.build_transcriber("vi")) is AudioTranscriber
    with patch.object(factory.worker_settings, "ASR_ENGLISH_BACKEND", "whisper"), \
         patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")):
        assert type(factory.build_transcriber("auto")) is AudioTranscriber


def test_default_backend_is_qwen_for_english_only():
    from ai_workers.core.config import WorkerSettings

    assert WorkerSettings().ASR_ENGLISH_BACKEND == "qwen3"


def test_transcribe_rejects_non_english():
    with patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")):
        with pytest.raises(ValueError):
            QwenEnglishTranscriber().transcribe("x.wav", "vi")


@pytest.mark.parametrize(
    "backend,lang,expect_qwen",
    [("qwen3", "en", True), ("qwen3", "vi", False), ("qwen3", "auto", False), ("whisper", "en", False)],
)
def test_factory_selects_engine(backend, lang, expect_qwen):
    with patch.object(factory.worker_settings, "ASR_ENGLISH_BACKEND", backend), \
         patch.object(AudioTranscriber, "__init__", lambda self, config=None: setattr(self, "device", "cpu")):
        engine = factory.build_transcriber(lang)
    assert isinstance(engine, QwenEnglishTranscriber) is expect_qwen

def test_transcribe_batch_returns_one_result_per_chunk_in_order():
    from ai_workers.modules.audio_v2.qwen_transcriber import QwenEnglishTranscriber

    calls = []

    class Model:
        def transcribe(self, audio, language, return_time_stamps):
            calls.append((len(audio) if isinstance(audio, list) else 1, language))
            items = audio if isinstance(audio, list) else [audio]
            return [SimpleNamespace(text=f"c{int(a[0][0])}", time_stamps=[]) for a in items]

    batch = [(0.0, np.full(4, 0.0)), (240.0, np.full(4, 1.0))]
    results = QwenEnglishTranscriber._transcribe_batch(Model(), batch)

    assert [r.text for r in results] == ["c0", "c1"]
    assert calls == [(2, ["English", "English"])]


def test_transcribe_batch_falls_back_to_single_chunks_when_batching_fails():
    from ai_workers.modules.audio_v2.qwen_transcriber import QwenEnglishTranscriber

    class Model:
        def transcribe(self, audio, language, return_time_stamps):
            if isinstance(audio, list):
                raise TypeError("batching unsupported")
            return [SimpleNamespace(text=f"c{int(audio[0][0])}", time_stamps=[])]

    batch = [(0.0, np.full(4, 0.0)), (240.0, np.full(4, 1.0))]
    results = QwenEnglishTranscriber._transcribe_batch(Model(), batch)

    assert [r.text for r in results] == ["c0", "c1"]

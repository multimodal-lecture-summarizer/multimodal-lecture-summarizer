from unittest.mock import patch

import pytest

from ai_workers.core.config import worker_settings
from ai_workers.modules.summarization import led_summarizer as led

UTTERANCES = [
    {"start": 0.0, "end": 5.0, "text": "Today we study gradient descent and why it converges."},
    {"start": 5.0, "end": 12.0, "text": "The learning rate controls the step size of every update."},
]
CHAPTERS = [{"title": "Intro", "startTime": 0.0, "endTime": 12.0}]


class _FakeLed:
    def __init__(self, fail=False):
        self.fail = fail
        self.closed = False

    def encode(self, text):
        return text.split()

    def generate(self, ids, budget):
        if self.fail:
            raise RuntimeError("oom")
        return "Alpha beta gamma. Delta epsilon."

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _led_defaults(monkeypatch):
    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "led")
    led.LED_SLOT.clear()
    yield
    led.LED_SLOT.clear()


def _build():
    return led.build_led_result(UTTERANCES, [], CHAPTERS, job_id="j", output_language="en", length_key="medium")


def test_kept_model_survives_a_good_run_and_is_reused():
    fake = _FakeLed()
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "led_resident_enabled", return_value=True), \
         patch.object(led, "LedModel", return_value=fake) as factory:
        assert _build() is not None
        assert not fake.closed and led.LED_SLOT.loaded
        assert _build() is not None
    assert factory.call_count == 1


def test_kept_model_is_dropped_after_a_failed_run():
    broken = _FakeLed(fail=True)
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "led_resident_enabled", return_value=True), \
         patch.object(led, "LedModel", return_value=broken):
        assert _build() is None
    assert broken.closed and not led.LED_SLOT.loaded


def test_model_is_loaded_and_closed_per_job_when_residency_is_off():
    models = [_FakeLed(), _FakeLed()]
    with patch("torch.cuda.is_available", return_value=True), patch.object(led, "led_resident_enabled", return_value=False), \
         patch.object(led, "LedModel", side_effect=models):
        _build()
        _build()
    assert all(m.closed for m in models)
    assert not led.LED_SLOT.loaded


def test_preload_needs_residency_the_led_backend_and_cuda(monkeypatch):
    with patch.object(led, "led_resident_enabled", return_value=False), patch.object(led, "load_led") as load:
        led.preload_led_model()
    load.assert_not_called()

    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "llm")
    with patch.object(led, "led_resident_enabled", return_value=True), patch.object(led, "load_led") as load:
        led.preload_led_model()
    load.assert_not_called()

    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "led")
    with patch("torch.cuda.is_available", return_value=False), \
         patch.object(led, "led_resident_enabled", return_value=True), patch.object(led, "load_led") as load:
        led.preload_led_model()
    load.assert_not_called()

    with patch("torch.cuda.is_available", return_value=True), \
         patch.object(led, "led_resident_enabled", return_value=True), patch.object(led, "load_led") as load:
        led.preload_led_model()
    load.assert_called_once()

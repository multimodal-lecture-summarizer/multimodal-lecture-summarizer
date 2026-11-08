import pytest

from ai_workers.core.config import worker_settings


@pytest.fixture(autouse=True)
def _no_led_weights_in_tests(monkeypatch):
    """Keep summarizer tests on the LLM path: on a CUDA machine the LED default would download weights.

    LED tests opt in with monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "led").
    """
    monkeypatch.setattr(worker_settings, "SUMMARIZER_ENGLISH_BACKEND", "llm")

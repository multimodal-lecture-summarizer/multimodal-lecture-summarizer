"""Test seam: engine selection lives in audio_v2.factory, so pipeline tests that mock
`audio_v2.transcriber.AudioTranscriber` must route the factory to that same mock.

The real factory is loaded lazily from its file: importing the audio_v2 package at
collection time would pull in the heavy ASR/diarization modules before the mocks exist.
"""

import importlib.util
from pathlib import Path
from types import ModuleType

_FACTORY_PATH = Path(__file__).resolve().parents[1] / "modules" / "audio_v2" / "factory.py"
_real_factory = None


def real_asr_identifier(source_language=None):
    """Production fingerprint identifier, independent of any sys.modules mock of the factory."""
    global _real_factory
    if _real_factory is None:
        spec = importlib.util.spec_from_file_location("_real_audio_factory", _FACTORY_PATH)
        _real_factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_real_factory)
    return _real_factory.asr_identifier(source_language)


def factory_stub(audio_mod: ModuleType) -> ModuleType:
    stub = ModuleType("ai_workers.modules.audio_v2.factory")
    stub.build_transcriber = lambda source_language=None: audio_mod.AudioTranscriber()
    stub.asr_identifier = real_asr_identifier
    return stub

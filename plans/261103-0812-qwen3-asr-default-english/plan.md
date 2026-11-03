# Qwen3-ASR default for English

Status: implemented, not committed. Real-clip validation pending.

## Requirements (user-confirmed)
- Output: `source_language="en"` uses Qwen3-ASR 0.6B + Qwen3-ForcedAligner-0.6B by default; same result dict as Whisper path.
- `vi` stays on Faster-Whisper (aligner has no Vietnamese). `auto` (UI default) detects language first with Whisper (`audio_v2/routing_transcriber.py`): English (p >= 0.7) -> Qwen, anything else or detection failure -> Whisper. Focus is English video; Vietnamese is secondary. Checkpoint fingerprint for `auto` covers both engines.
- Faster-Whisper stays in code as fallback: Qwen load/runtime failure falls back to Whisper for that job.
- Out of scope: removing Whisper deps, 1.7B model, Vietnamese aligner, WER benchmark on app videos.
- Constraints: transformers==4.57.6 pin (Florence-2) must not break; env var `ASR_ENGLISH_BACKEND=whisper` restores old behavior.

## Touchpoints (done)
- `ai_workers/core/config.py`: `ASR_ENGLISH_BACKEND` default `qwen3`, `QWEN_*` settings
- `ai_workers/modules/audio_v2/qwen_transcriber.py`: engine, chunking, output validation, Whisper fallback
- `ai_workers/modules/audio_v2/factory.py`: `uses_qwen`, `asr_identifier`, `build_transcriber`
- `ai_workers/tasks.py`: `process_video` uses factory; checkpoint fingerprint uses `asr_identifier`; `result["asr_model"]` reports engine actually run
- `ai_workers/requirements.txt`: `qwen-asr>=0.0.6`
- `ai_workers/tests/test_qwen_transcriber.py`: 16 tests
- `docs/current-system-architecture.md` Stage 1

## Acceptance (verified)
- [x] en -> Qwen engine; vi/auto -> Whisper; whisper flag -> Whisper.
- [x] Qwen exception or implausible timestamps -> Whisper result returned.
- [x] non-en passed directly to Qwen engine raises.
- [x] Worker tests pass per file (12 of 13 files; `test_florence_runtime` has 5 environment failures: torch 2.6.0 vs pinned 2.5.1). Combined single-run crashes in pytest capture, also without the new tests.

## Open risks / follow-ups
- Timestamp unit assumed seconds (guarded by `validate_words`); `qwen-asr` never run for real.
- `qwen-asr` pulls gradio and bumps pydantic/fastapi in the worker env.
- No Qwen vs Whisper `small` WER comparison; Qwen path has no VAD.
- `test_reliability_chaos.py:352`, `test_extractive_fallback.py:448` assert a mocked `AudioTranscriber` not called; pipeline now goes through factory, so they may pass vacuously.
- `.env.example` does not list `ASR_ENGLISH_BACKEND` / `QWEN_*`; weights use the HF default cache, not `CACHE_DIR`.

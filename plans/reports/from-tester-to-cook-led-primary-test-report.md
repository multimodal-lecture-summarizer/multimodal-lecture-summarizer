# LED primary summarizer: per-file test report

Date: 2026-11-03 | Runner: `python -m pytest <file>` (Python 3.11.1 global, pytest 9.0.3), `PYTHONPATH=.`, one run per file, repo root.
Note: `backend/.venv` has no pytest; global python used (has torch, transformers, celery). No source or test files modified.

## Per-file results

| # | File | Result | Time |
|---|------|--------|------|
| 1 | test_led_summarizer.py | 15 passed | 7.6s |
| 2 | test_ai_capability.py | 7 passed | 13s |
| 3 | test_checkpoint_persistence.py | 7 passed | 51s |
| 4 | test_extractive_fallback.py | 14 passed | 65s |
| 5 | test_florence_golden.py | 1 skipped (needs `RUN_FLORENCE_SMOKE_TEST=1`) | 25s |
| 6 | test_florence_runtime.py | **16 passed, 5 failed** | 13s |
| 7 | test_llm_resilience.py | 14 passed | 16s |
| 8 | test_object_tracker.py | 7 passed | 18s |
| 9 | test_quality_postprocess.py | 4 passed | <1s |
| 10 | test_qwen_transcriber.py | 22 passed | 5s |
| 11 | test_rag_index.py | 5 passed | 5s |
| 12 | test_reliability_chaos.py | 12 passed | 57s |
| 13 | test_task_preflight.py | 5 passed | 5s |
| 14 | test_timeline_chaptering.py | 1 passed | 2s |

Totals: 129 passed, 5 failed, 1 skipped (135 collected). Wall time about 5 min.

## Failures

All 5 in `test_florence_runtime.py` (known env issue):
- test_cpu_is_resolved_without_cuda_dependency
- test_cuda_request_falls_back_to_cpu_when_cuda_is_unavailable
- test_cuda_request_uses_cuda_when_available
- test_installed_environment_matches_contract
- test_invalid_device_fails_fast

Error (same for all): `FlorenceRuntimeError: Unsupported Florence-2 runtime: torch==2.6.0+cu124 (expected 2.5.1...); Pillow==12.3.0 (expected 12.2.0); numpy==2.4.6 (expected 1.26.4); torchvision==0.21.0+cu124 (expected 0.20.1)`. Version gate in `ai_workers/modules/visual_v2/florence_runtime.py` fires before the device/CUDA logic under test. Fix is env (pin to requirements), not code.

## LED relation check

- `led_summarizer.py`: NOT related. No failures in LED tests (15/15). No failing file imports it except via `summarizer.py`.
- `summarizer.py` (early LED hook in `Summarizer.summarize`, ~L777): NOT related. Hook returns None when LED disabled. Summarizer-touching tests (checkpoint, extractive, llm_resilience, object_tracker, rag_index, reliability_chaos) all pass. `conftest.py` forces `SUMMARIZER_ENGLISH_BACKEND="llm"` so those run the LLM path.
- `core/config.py`: NOT related. `florence_runtime.py` does not import config. Config changes (`ENABLE_OBJECT_TRACKING=False`, `PADDLEOCR_LANG`, `SUMMARIZER_ENGLISH_BACKEND`, `LED_MODEL*`) do not break any test. Object-tracker tests pass with the flag default off, since they call the tracker directly.

## Concerns

- LED hook runs in other suites only under the `llm` override. LED-on path is covered only by `test_led_summarizer.py`. No real-GPU/LED-weights test run (CUDA absent or not exercised).
- Florence golden test skipped; real checkpoint not validated.
- Tests use global Python, not the project venv `backend/.venv` (no pytest there). Torch 2.6.0 in use vs pinned 2.5.1 is the root of the Florence failures.
- Test run did not use `pytest --cov`; no coverage numbers gathered.

Status: DONE_WITH_CONCERNS
Summary: 14 files run separately. LED file 15/15 pass. 129 passed, 5 failed (all florence_runtime env version gate), 1 skipped. No failure touches led_summarizer.py, summarizer.py LED hook, or core/config.py.
Concerns: Florence env drift (torch 2.6.0 vs pinned 2.5.1); LED-on path only covered in its own file; no coverage run.

Unresolved questions:
- Should the project venv (`backend/.venv`) get pytest so runs match the pinned env? Current runs used global Python.
- Is a Florence smoke run (`RUN_FLORENCE_SMOKE_TEST=1`) wanted before merge?

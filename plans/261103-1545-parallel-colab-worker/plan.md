# Parallel Colab GPU worker

Status: in-progress. Scope confirmed by user: audio/visual branches parallel inside one job + R2 upload pool + parallel notebook setup; flag with sequential fallback.

## Acceptance
- `ENABLE_PARALLEL_BRANCHES` (default false; notebook sets true). Off = current sequential behavior unchanged.
- On: audio (ASR + diarization) and visual (scene → OD&T → semantic) run concurrently; outputs identical in shape to sequential.
- A branch failing with OOM is re-run sequentially after the other finishes; other errors and revocation propagate unchanged.
- Progress/log updates stay thread-safe, progress never goes backwards; Redis access serialized (broker pool is capped at 2 connections).
- R2 keyframe upload uses a thread pool; failed uploads keep local URL as before.
- Notebook setup: apt/git-lfs, clone, venv creation overlap; single install step.

## Out of scope
- Multiple jobs per worker (pool/concurrency unchanged), model prefetch, pipeline algorithm changes.

## Touchpoints
- `ai_workers/core/parallel_branches.py` (new, testable helper)
- `ai_workers/core/config.py` (+1 setting)
- `ai_workers/tasks.py` (`process_video` stages 1–4, R2 upload)
- `ai_workers/colab_gpu_worker.ipynb`
- `docs/colab-gpu-worker-guide.md`
- Tests: `ai_workers/tests/test_parallel_branches.py`; existing process_video tests must still pass.

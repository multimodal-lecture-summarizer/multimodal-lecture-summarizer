---
phase: 4
title: Verification on Colab T4
status: completed
priority: P2
dependencies:
  - phase-03
---

# Phase 4: Verification on Colab T4

## Overview
Verify the patched stack actually runs rate-limit-free on Colab T4, with budget parity and fallback, before handing off to `260831-fix-notebooks-3-5` for final consistency sweep.

## Requirements
- Functional: `04` completes end-to-end on T4 with `GEMINI_API_KEY` unset; HF path and deterministic path both produce `eval_results` with ROUGE/coverage.
- Non-functional: Total wall time <10 min on T4; VRAM <12GB; no 429; `token_usage` ≤ D-T08 for all variants.

## Architecture
```
Verification matrix (run on T4):

  Run A (default): LLM_PREFERENCE=auto, no key  -> expect HF (or deterministic on CPU)  -> 25 lectures -> 4 variants -> metrics
  Run B (offline): LLM_PREFERENCE=deterministic -> expect Deterministic v2                -> same
  Run C (opt-in):  LLM_PREFERENCE=gemini + key   -> expect Gemini with backoff            -> same (only if key present)

Each run asserts:
  - SummaryResult.status == "ok"
  - token_usage["source_tokens"] <= 32000 and ["output_tokens"] <= 512 (D-T08)
  - No exception / empty string (the old Gemini failure mode)
  - Forest plot + stat table still render (D-T07)
```

## Related Code Files
- Modify: (none, verification only)
- Reference: `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb` (patched), `benchmarks/metrics/summarization_metrics.py`, `benchmarks/metrics/statistics.py`

## Implementation Steps
1. **Colab T4 dry-run A:** Fresh runtime, T4 enabled, no `GEMINI_API_KEY`, `%cd` to repo, `LLM_PREFERENCE=auto`. Run `04` cell-by-cell. Capture `print(f"[LLM Engine] Using ...")` and `eval_results` shape. Assert `04` did not call Gemini (grep logs for `GeminiLLMEngine`).
2. **Dry-run B:** Same runtime, `LLM_PREFERENCE=deterministic`. Verify deterministic v2 produces chapter bullets (`**Chapter` prefix) and `unsupported` claim rate computed.
3. **(Optional) Dry-run C:** Only if user provides key: `LLM_PREFERENCE=gemini` with backoff. Verify that a simulated 429 (by spamming 30 calls) still falls back instead of aborting.
4. **Budget check:** After each run, assert `cfg_s0.max_source_tokens==32000` and `cfg_s0.max_output_tokens==512` (copy D-T08 guard from cell 6). Fail if any `SummaryResult.token_usage` exceeds.
5. **Cross-plan hand-off:** After A+B pass, run `ck plan status` on both `260831-ratelimit-free-summarization` and `260831-fix-notebooks-3-5` to confirm `04` now satisfies both real-data and rate-limit-free invariants. Mark this plan's phases completed via `ck plan check`.

## Success Criteria
- [ ] Run A completes with `HuggingFaceLLMEngine` or `DeterministicAbstractiveEngine` (log line present), zero `429`/`ResourceExhausted` in stderr.
- [ ] Run B completes with `DeterministicAbstractiveEngine` v2, all 25 lectures produce bullet summaries, `ROUGE-1/2/L` non-zero.
- [ ] D-T08 budget holds for S0/S1/S3/S4 (`source_tokens ≤32000`, `output_tokens ≤512`).
- [ ] `ck plan check 4` succeeds and plan status is `completed`.

## Risk Assessment
- **Risk:** HF model download times out on Colab (network). **Mitigation:** Run B already proves deterministic path works offline; A is allowed to fallback to deterministic and still pass.
- **Risk:** T4 VRAM OOM with Qwen 1.5B FP16. **Mitigation:** Design specifies fallback to BART or deterministic on OOM exception; verification must assert OOM is caught, not crash.
- **Risk:** Fix-notebooks plan's `USE_CACHED_FALLBACK` changes lecture count 25→20, affecting verification baseline. **Mitigation:** Verification asserts `len(real_lecture_items) in {20,25}` not exact 25.

<!-- Updated: Validation Session 2 - Run A (auto→Qwen) + Run B (deterministic) mandatory, no Drive cache, B proves offline path -->
**Validation Update (2026-08-31):** Verification requires Run A `auto→Qwen2.5-1.5B` and Run B `deterministic` both pass D-T08 budget; C `gemini` optional. No Drive cache — each run pulls HF fresh (60s ok). Deterministic must be proven as guaranteed offline path.

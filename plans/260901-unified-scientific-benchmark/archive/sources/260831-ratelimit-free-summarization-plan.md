---
title: Notebook 04 RateLimit-Free Hierarchical Summarization
description: >-
  Replace Gemini API dependency in 04_phase4_hierarchical_summarization with
  local/offline alternatives (HF small LM + enhanced Deterministic + optional
  cached Gemini) to eliminate rate-limit failures on Colab T4 while preserving
  D-T08 budget parity and RQ2 metric integrity.
status: completed
priority: P1
branch: main
tags:
  - notebook
  - rq2
  - summarization
  - gemini
  - ratelimit
  - colab
  - llm-engine
blockedBy:
  - 260831-fix-notebooks-3-5
blocks: []
created: '2026-08-31T13:46:42.696Z'
createdBy: 'ck:plan'
source: skill
---

# Notebook 04 RateLimit-Free Hierarchical Summarization

## Overview

`04_phase4_hierarchical_summarization.ipynb` currently calls `benchmarks/models/llm_engine.py:110` `get_llm_engine()` → `GeminiLLMEngine` (`gemini-2.5-flash`) for every chapter/section. On 25 VT-SSum lectures, S0(1) + S1(6-10) + S3(4-8) + S4(4-8) = **~15-27 Gemini calls/run**. Free-tier 429/rate-limit aborts the whole notebook and blocks RQ2 evaluation. This plan makes 04 **rate-limit-free** by introducing a pluggable offline-first engine stack and patching the notebook + `llm_engine.py`/`summarization.py` to default to it on Colab T4.

## Goals

1. 04 runs **zero Gemini calls by default** on Colab T4; optional Gemini is opt-in only.
2. Preserve `D-T08` equal budget (≤32k source / ≤512 output tokens) and `D-T07` stats across S0/S1/S3/S4.
3. Add a Colab-T4-fit local HF engine (Qwen2.5-1.5B or BART-large-cnn, ~3-6GB VRAM FP16) + an upgraded `DeterministicAbstractiveEngine` as guaranteed fallback.
4. Keep a single `get_llm_engine(preference)` factory so 04/05 and `run_rq2/3_benchmark.py` share the same selection logic; no duplicate engine code.
5. Notebook 04 gains a 1-cell config (`LLM_PREFERENCE="hf"|"deterministic"|"gemini"`) with auto-detect, retry-free, and clear logging of which backend was used.

## Non-Goals

- Changing chaptering (C5) or retrieval (Q0-Q3); only the summarization LLM backend.
- Training/fine-tuning a new summarizer; re-use existing HF checkpoints.
- Modifying frozen manifest or VT-SSum data.

## Constraints

- **YAGNI/KISS/DRY:** Minimal new files; edit `llm_engine.py` + `summarization.py` + `04` cell only. No new microservices.
- **Colab T4:** 15GB VRAM, no persistent disk, `pip install` must stay <2 min, model load <60s with `HF_HOME=/root/.cache/huggingface`.
- **Offline-first:** Default path must not require `GEMINI_API_KEY`; `get_llm_engine("auto")` must never raise 429 to caller — it falls back silently with a warning.
- **Respect development rules:** Keep `BaseLLMEngine` contract; all engines honor `max_tokens`/`temperature` truncation.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Research & RateLimit Analysis](./phase-01-research-ratelimit-analysis.md) | Completed |
| 2 | [Alternative Engine Design](./phase-02-alternative-engine-design.md) | Completed |
| 3 | [Implementation & Notebook Patch](./phase-03-implementation-notebook-patch.md) | Completed |
| 4 | [Verification on Colab T4](./phase-04-verification-on-colab-t4.md) | Completed |

## Dependencies

- **Blocked by `260831-fix-notebooks-3-5`:** That plan edits the same `04` cell (real VT-SSum + real C5 boundaries). This plan edits the **LLM backend** inside that same cell. Merge order: fix-notebooks first, then apply this plan's engine switch on top. If fix-notebooks is not yet merged, this plan's patch must be rebased.
- No dependency on `260830-1917-scientific-benchmark`.

## Validation Log

### Session 1 — 2026-08-31 (plan creation)
- **Scope challenge:** User explicitly requested "không dùng Gemini vì ratelimit, cho hướng khác" — not a scope creep; Gemini is a single external dependency, replacement is a focused infra swap, not a feature expansion. Approved for P1.
- **Mode:** `--auto` → detected moderate complexity (single notebook + 2 backend files) → 4 phases, no red-team needed but verification on real T4 required.

### Session 2 — 2026-08-31 (validate)
**Verification Results (Standard tier, 4 phases):**
- Claims checked: 10 | Verified: 10 | Failed: 0 | Unverified: 0 | Tier: Standard
- All symbols verified: `BaseLLMEngine`, `GeminiLLMEngine`, `HuggingFaceLLMEngine`, `DeterministicAbstractiveEngine`, `get_llm_engine`, `S0/S1/S3/S4`, `04` notebook exists, `requirements.lock.txt` present.

**Questions & Answers:**
1. **[Engine]** Default `auto` engine = **Qwen2.5-1.5B-Instruct** (Recommended). Rationale: best chapter synthesis, 3.5GB FP16 fits T4, fallback deterministic on OOM. BART rejected as default (faster but weaker instruction following).
2. **[CPU fallback]** `auto` on CPU/no HF → **auto fallback deterministic with warning** (Recommended), not fail-loud. Rationale: Colab CPU-only must still produce metrics; fail-loud would block offline verification.
3. **[Gemini 429]** `preference=gemini` on 429 → **retry 2× (2s/4s) + fallback HF/deterministic, never return ""** (Recommended). Rationale: old code returns `""` → ROUGE=0 silent failure; backoff+fallback preserves budget and metrics.
4. **[HF cache]** Cache HF model on Drive? → **No cache** (Recommended). Rationale: KISS, 60s extra per run acceptable, avoids Drive mount dependency; HF singleton already avoids per-chapter reloads.

**Confirmed Decisions (propagate to phases):**
- Phase 2: lock `Qwen2.5-1.5B-Instruct` as primary, `deterministic` as guaranteed fallback; no Drive cache; `BART-large-cnn` remains alternative only.
- Phase 3: implement `get_llm_engine(preference)` with auto→HF→deterministic, gemini→retry+fallback, deterministic→direct; single `llm` instance injected into S0/S1/S3/S4.
- Phase 4: verification runs A (auto→Qwen), B (deterministic) mandatory; C (gemini) optional.

### Whole-Plan Consistency Sweep — 2026-08-31
- Re-read `plan.md` + all `phase-*.md`. No stale terms. `LLM_PREFERENCE` naming consistent across plan/phase 2/phase 3. `D-T08` budget invariant unchanged. `get_llm_engine` contract consistent with `BaseLLMEngine.generate`. No contradictions. Ready for `/ck:cook`.

## Open Questions

- None — all 4 validation questions answered. No unresolved questions.

## Related Code Files

- Modify: `benchmarks/models/llm_engine.py`, `benchmarks/models/summarization.py`, `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb`
- Reference: `experiments/notebooks/03_phase3_representation_and_chaptering.ipynb` (loads C5 checkpoint, not affected), `benchmarks/models/retrieval_qa.py` (shares `llm_engine`)
- Create (optional): `benchmarks/models/hf_engine.py` only if `HuggingFaceLLMEngine` grows >100 LOC — otherwise extend in-place.

## Success Criteria

- [ ] `04` completes on Colab T4 without `GEMINI_API_KEY` set, with `HF available → Qwen/BART`, otherwise `Deterministic` — zero 429.
- [ ] All 4 variants S0/S1/S3/S4 produce `SummaryResult` with `token_usage` ≤ D-T08 and ROUGE/coverage metrics computed.
- [ ] `get_llm_engine(preference="hf"|"deterministic"|"auto")` contract documented and tested with `preference="gemini"` opt-in still working when key present.

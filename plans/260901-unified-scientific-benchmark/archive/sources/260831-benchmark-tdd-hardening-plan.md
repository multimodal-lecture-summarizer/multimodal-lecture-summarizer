---
title: "Benchmark Data Scale & RateLimit Hardening (TDD)"
description: "TDD hardening for 260830-1917-scientific-benchmark: expand YTSeg subset 100-300 for RQ1 power, VISTA 300 subset with self-ASR for RQ2 primary, and make 04/05 rate-limit-free and Colab-resume-safe with cached predictions — all with tests-first per phase."
status: pending
priority: P1
branch: "main"
tags: [benchmark, tdd, ytseg, vista, ratelimit, colab, rq1, rq2, rq3]
blockedBy: [260831-fix-notebooks-3-5, 260831-ratelimit-free-summarization]
blocks: []
created: "2026-09-01T01:59:25.266Z"
createdBy: "ck:plan"
source: skill
---

# Benchmark Data Scale & RateLimit Hardening (TDD)

## Overview

`plans/260830-1917-scientific-benchmark` is blocked on three operational failures: **RQ1 underpowered (20 lectures → CIs cross 0)**, **RQ2 rate-limit + Colab 12h kill (30 Gemini calls/run)**, and **RQ3 untested at scale (30 QA vs 5,252)**. This hardening plan fixes all three **with tests-first (TDD) per phase**, so the master benchmark's M1/M3/M4 exits are actually reachable. It implements the scoped **C-light** from `reports/brainstorm-260831-ratelimit-data-expansion.md` (YTSeg 300 + VISTA 300, not 19k/18k) with a 2-week fallback gate to `B` (YTSeg 100 + TIB 80).

Primary thesis claims C1-C3 remain falsifiable; non-goals are full 19k/18k scale, finetune, and IRB expansion (D-S03 stays single-author + LLM-as-a-Judge).

## Goals

1. **RQ1 power:** YTSeg lecture/science subset `n=100-300` with frozen `manifests/frozen_manifest_v2.json`, `feature_store` on Drive per D-T09, `C5>C1` detectable at `d=0.3` (back-solved via D-T10 pilot).
2. **RQ2 reliability:** `04` finishes <10 min / 25 videos (<30 min / 100) on T4 with **0 Gemini calls** (`Qwen2.5-1.5B` singleton else `Deterministic` v2), chunked + `outputs/phase4_cache/` resume survives Colab kill, D-T08 `32k/512` strictly enforced.
3. **RQ3 scale:** `05` uses full `EduVidQA` 5,252 QA offline (`all-MiniLM-L6-v2` SBERT + BM25 hybrid, `top_k=3` D-T08), not 30-item demo, with `Q3>Q0` measurable.
4. **TDD gates:** Each phase writes failing tests first (manifest leakage, ASR WER, LLM fallback, cache resume, budget), then green, then refactor — regression-safe for frozen `C5/C6` (D-T02).

## Non-Goals

- Full YTSeg 19,299 / VISTA 18,599 (1.93TB) — out of scope for 6-month thesis; scoped to 300 subsets.
- Training a Video-LLM from scratch; finetune Qwen is optional appendix, not gate.
- Second annotator / IRB expansion — stays per D-S03.
- Redistribution of raw video — IDs + features only per D-S04.

## Constraints

- **TDD:** Every phase has `tests/test_*.py` written first and failing before implementation (see `output-standards.md`).
- **D-T02/D-T04/D-T07/D-T08/D-T09 frozen:** No architecture change to C5/C6, OCR=PaddleOCR v3 0.6, DINOv2 ViT-S/14, Holm within RQ, HF weights on local SSD not Drive.
- **Colab T4:** 15GB VRAM, 12h limit, `feature_store` sharded, `HF_HOME=/root/.cache/huggingface`.
- **YAGNI/KISS/DRY:** One new script per dataset (`fetch_ytseg_subset.py`, `vista_subset_asr.py`), reuse `benchmarks/core/feature_store.py` and `llm_engine.py` from `4013290`.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [YTSeg Subset & Feature Store (TDD)](./phase-01-ytseg-subset-feature-store-tdd.md) | Pending |
| 2 | [VISTA Subset ASR Pipeline (TDD)](./phase-02-vista-subset-asr-pipeline-tdd.md) | Pending |
| 3 | [RateLimit-Free Summarization & Retrieval Hardening (TDD)](./phase-03-ratelimit-free-summarization-retrieval-hardening-tdd.md) | Pending |
| 4 | [Validation & Stats Gate (TDD)](./phase-04-validation-stats-gate-tdd.md) | Pending |

## Dependencies

- **Blocked by `260830-1917-scientific-benchmark`:** This is the hardening overlay for its M1 (dataset gate) and M3/M4 (RQ1/RQ2). Implements D-T12 Nominal Path 1 (VISTA approved 2026-08-31) in scoped form.
- **Blocked by `260831-fix-notebooks-3-5`:** Reuses its real-data `03` (`cached_features` + rebuilt `targets`) as seed; hardening expands from 20→300, not from 0.
- **Blocked by `260831-ratelimit-free-summarization` (done):** Reuses its `llm_engine.py` (`Qwen` + deterministic v2 + retry) as baseline for Phase 3 chunked hardening.

## Validation Log

### Session 1 — 2026-09-01 (plan creation, --tdd)
- **Mode:** `--tdd` → per-phase scout + tests-first, 4 phases, Standard tier (10 claims/phase).
- **Brainstorm source:** `reports/brainstorm-260831-ratelimit-data-expansion.md` (C-light scoped, B fallback). User chose C but agreed to report-only; this plan turns C-light into TDD-tracked implementation with fallback gate at Week 6 per `05-6month-timeline.md`.
- **Scope challenge:** No scope creep — hardening is within `00-master-plan.md` §8 In Scope (frozen feature extraction, chaptering, summarization, retrieval) and does not add UI/billing.

### Session 2 — 2026-09-01 (validate)
**Verification Results (Standard tier, 4 phases):**
- Claims checked: 10 | Verified: 10 | Failed: 0 | Unverified: 0
- All symbols verified: `C5/C6` frozen D-T02, `YTSeg` 19k, `VISTA` 18k, `llm_engine.py` hardened 4013290, `03/04/05` notebooks exist.

**Questions & Answers:**
1. **[YTSeg n]** `n=300` (Recommended) — D-T10 needs ≥100 for `d=0.3`, 300 gives CI width ~1/√n and still fits 30GB Drive.
2. **[VISTA gate]** User chose **Bắt buộc VISTA** (no fallback) — overrides D-T12 Tier 2 fallback. Risk: WER>30% or attrition>50% will block thesis. Documented as user decision; plan will not auto-fallback to TIB.
3. **[TDD]** User chose **Bỏ TDD** — override `--tdd` mode. Phases will implement directly without red→green tests-first. Success criteria updated to reflect no TDD gate (tests become post-hoc verification, not pre-condition).
- **Whole-plan consistency sweep:** `LLM_PREFERENCE` and `D-T08` still consistent, but TDD flag in title now stale (kept for traceability) and VISTA fallback gate disabled.

**Confirmed Decisions (propagate):**
- Phase 1: `n=300` locked, over-fetch 500, no fallback to 100.
- Phase 2: VISTA mandatory (300), no TIB fallback on WER/attrition — thesis blocks if VISTA fails.
- Phase 3/4: TDD dropped — tests become post-hoc, not failing-first.

### Whole-Plan Consistency Sweep — 2026-09-01
- Stale `TDD` in title/phase names kept for history but validation notes TDD dropped.
- `VISTA fallback` removed from Phase 2 success criteria and risk mitigation — now fail-loud.

## Open Questions

- None — validation closed with 3 questions answered. Drive quota 30GB and VISTA mandatory risk acknowledged.

## Related Code Files

- Create: `benchmarks/scripts/fetch_ytseg_subset.py`, `benchmarks/scripts/vista_subset_asr.py`, `tests/test_ytseg_manifest.py`, `tests/test_vista_asr.py`, `tests/test_phase4_cache.py`
- Modify: `benchmarks/models/llm_engine.py` (already hardened in 4013290, Phase 3 adds chunked cache), `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb` (add resume), `experiments/notebooks/05_phase5_evidence_retrieval_and_qa.ipynb` (scale to full EduVidQA), `benchmarks/core/feature_store.py` (shard support)
- Reference: `plans/260830-1917-scientific-benchmark/01-dataset-manifest.md` (gate), `02-benchmark-matrix.md` (C1-C6/Q0-Q3), `decisions-log.md` (D-T02..D-T14)

## Success Criteria

- [ ] `frozen_manifest_v2.json` with `n≥100` YTSeg subset, 0 leakage, per-video predictions retained.
- [ ] `04` on T4: 100 videos <30 min, 0 Gemini, cache hit >80% on resume, D-T08 obeyed.
- [ ] `05` on full EduVidQA 5k QA completes <30 min offline, `Q3>Q0` CI reported.
- [ ] All TDD test files existed failing before implementation and now pass.


---
title: "Fix Multimodal Notebooks 3-5 to Use Real Data"
description: "Rework experiments/notebooks 03/04/05 to consume real cached multimodal features and honest boundaries/references instead of synthetic randn/placeholder data, without re-executing notebooks."
status: pending
priority: P1
branch: "main"
tags: [notebook, benchmark, rq1, rq2, rq3, data-integrity]
blockedBy: []
blocks: [260831-ratelimit-free-summarization]
created: "2026-08-31T10:26:54.933Z"
createdBy: "ck:plan"
source: skill
---

# Fix Multimodal Notebooks 3-5 to Use Real Data

## Overview

The three core research notebooks (`experiments/notebooks/03_phase3_representation_and_chaptering.ipynb`, `04_phase4_hierarchical_summarization.ipynb`, `05_phase5_evidence_retrieval_and_qa.ipynb`) currently present **synthetic or fabricated results** while their **backend model code is real** (PyTorch C1–C6, SBERT+LLM summarization/QA). This mismatch makes the notebooks unusable as research evidence.

The repo **already contains real data** that should be consumed:
- `benchmarks/data/cached_features/*.pt` (20 real lectures: text 384d, visual 384d, OCR 384d, acoustic **32d**, targets, ground-truth boundaries, real ASR transcript).
- `benchmarks/data/dataset.py` (`LectureFeatureDataset`, `collate_lecture_batches`, `create_lecture_splits`) — the exact load path used by `run_rq1_benchmark.py`.
- `experiments/datasets/eduviqa/q_and_a.json` (20 real lectures with `video_name` matching cached `.pt`; real Q&A pairs) → **real transcript + QA path for Phase 5**, already used by `run_rq3_benchmark.py`.
- `plans/260830-1917-scientific-benchmark/probes/cache/vtssum/test/` (962 real VT-SSum JSONs: real transcript + segmentation + extractive reference labels).
- `plans/260830-1917-scientific-benchmark/probes/cache/eduvidqa/data/real_world_test.csv` (269 real QA pairs; QA-only — secondary source for Phase 5).

This plan edits the notebook cell code **and the related reference CLI scripts** so they load the **real data** and **real model outputs**, clear/blow away stale fabricated outputs, and fail loudly when real data is absent — **without executing the notebooks** (user explicitly requested no run; static edits only).

## Goals
1. Phase 3: replace `np.random.randn` synthetic features with real `cached_features/*.pt` (via `create_lecture_splits`/`collate_lecture_batches`); fix acoustic dim 64→32; **train + save a real C5 checkpoint** for downstream phases.
2. Phase 4: replace hand-computed (cumulative-sum) fake `c5_predicted_boundaries` with the real C5 checkpoint output; fix the template-fake OCR in the notebook **and** `run_rq2_benchmark.py`.
3. Phase 5: stop injecting answers into fabricated transcripts; **reuse `q_and_a.json` + real cached transcripts** (matching `video_name`); fix the boundary-noise + template-OCR fabrication in the notebook **and** `run_rq3_benchmark.py`.
4. Consistency sweep: cross-check the three notebooks + two CLI scripts against each other and the frozen manifest, then hand off for a re-run.

## Constraints
- **Do NOT execute notebooks or scripts.** All edits are source edits (`*.ipynb` JSON cells + `run_rq2/rq3_benchmark.py`).
- **Respect `./docs/development-rules.md`**.
- Minimal footprint: prefer editing existing cells/code over writing new files; only add a small loader/checkpoint helper if genuinely needed (YAGNI).
- Preserve the real backend code (`chaptering.py`, `summarization.py`, `retrieval_qa.py`, `dataset.py`) — they are correct.
- `run_rq1_benchmark.py` is already clean (0 `randn`, real `test_out.predicted_boundaries`) — do not regress it.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Phase 3: Real Cached Features](./phase-01-phase-3-real-cached-features.md) | Pending |
| 2 | [Phase 4: Real C5 Boundaries & OCR](./phase-02-phase-4-real-c5-boundaries-ocr.md) | Pending |
| 3 | [Phase 5: Real EduVidQA Transcripts](./phase-03-phase-5-real-eduvidqa-transcripts.md) | Pending |
| 4 | [Consistency & Verification](./phase-04-consistency-verification.md) | Pending |

## Dependencies

- Phase 2 depends on Phase 1 (real C5 boundaries come from Phase 3's real-data-trained model).
- No cross-plan dependencies with `260830-1917-scientific-benchmark` — that plan is the parent benchmark; this is a notebook-integrity fix within it. Re-check `decisions-log.md` for D-T04/D-T07/D-T08/D-T09 before finalizing.

## Validation Log

### Session 1 — 2026-08-31
**Trigger:** `/ck:plan validate` requested by user after plan creation.

#### Whole-Plan Consistency Sweep (Step 7)
Completed a sweep across all 4 phase files + plan.md. Re-read every phase, reconciled stale references, and removed the previous conflicts (e.g. "gate off S4 for missing OCR" → "fix template OCR in notebook AND run_rq2"; "train-or-load in-session" → "load `checkpoints/c5_real.pt`").
- **Critical finding:** cached `.pt` `targets` are broken (near-empty, misaligned). Resolved by user decision to rebuild `targets` with a guard.
- **Correction during sweep:** initial suggestion to use VT-SSum segmentation as GT was **wrong** — verified **0 overlap** between VT-SSum test videos and the 20 cached lectures (different datasets). Reverted GT source to the cache's own `ground_truth_boundaries`.
- **Outcome:** all 4 phase files now consistent with the decided GT (rebuild `targets` from `ground_truth_boundaries`), the C5 checkpoint path (`checkpoints/c5_real.pt`), the split seed (42), and the widened scope (`run_rq2/rq3_benchmark.py`).

#### Verification Results (Step 2.5)
- **Tier:** Standard (4 phases)
- **Claims checked:** 15
- **Verified:** 14 | **Failed:** 1 | **Unverified:** 0

| Claim | Result | Evidence |
|-------|--------|----------|
| `benchmarks/data/cached_features/*.pt` has acoustic dim 32 | VERIFIED | `CA01.pt`: `acoustic_features (11,32)` |
| `cached_features` has 20 real lectures | VERIFIED | `manifest.json`: `total_lectures: 20`; 20 `.pt` files |
| `create_lecture_splits` / `collate_lecture_batches` exist | VERIFIED | `benchmarks/data/dataset.py:63`, `:105` |
| `C5_TemporalCrossAttentionTransformer` exists | VERIFIED | `benchmarks/models/chaptering.py:279` |
| `S4_MultimodalHierarchySummarizer` exists, `summarize` accepts `ocr_texts` | VERIFIED | `benchmarks/models/summarization.py:266`, `:271` |
| `run_rq1_benchmark.py` exists (Phase 1 reference) | VERIFIED | `benchmarks/scripts/run_rq1_benchmark.py` |
| `compute_all_qa_metrics` exists (Phase 3 reference) | VERIFIED | `benchmarks/metrics/qa_metrics.py:100` |
| VT-SSum test cache has 962 real JSONs with transcript+references | VERIFIED | sampled JSON: keys `id,title,info,url,segmentation,summarization`; real sentences |
| EduVidQA `real_world_test.csv` has NO transcripts (only url/id/q/a/time) | VERIFIED | CSV header + rows; no transcript column; yt-dlp referenced in probes |
| `run_rq3_benchmark.py` uses real `q_and_a.json` + cached transcripts | VERIFIED | `run_rq3_benchmark.py:44,77-78` |
| `run_rq3_benchmark.py` STILL has boundary noise + template OCR | VERIFIED (defect) | `run_rq3_benchmark.py:84,99` |
| `run_rq2_benchmark.py` uses real cached features but template OCR | VERIFIED (defect) | `run_rq2_benchmark.py:31,61,100` |
| `run_rq1_benchmark.py` clean (0 randn, real boundaries) | VERIFIED | randn count=0; uses `test_out.predicted_boundaries` |
| `q_and_a.json` is a list of 20 lectures with `video_name` matching cache | VERIFIED | `experiments/datasets/eduviqa/q_and_a.json`; e.g. "Thailand Stock Market..." |
| **Cache `targets` aligned with `ground_truth_boundaries`** | **FAILED** | `CA01`: targets.sum=0 vs 13 gtb; `CA05`: sum=2 vs 27 gtb; `DS12`: sum=2 vs 44; `DS11`: sum=1 vs 44; `CA05` timestamps non-monotonic `[0,0,15.96,0,28.44,...]` |

Design note from verification: `benchmarks/scripts/run_rq2_benchmark.py` and `run_rq3_benchmark.py` also exist — worth confirming these scripts already consume real data; if so they serve as stronger references than the notebook for Phases 2/3.

#### Critical Finding: Supervision Inconsistency in `cached_features/*.pt`
During the consistency sweep, spot-checking the real `.pt` cache revealed the **supervision labels are internally inconsistent**:
- `CA01`: `ground_truth_boundaries` has 13 boundary timestamps, but `targets.sum() == 0` (no boundary marked at all).
- `CA05`: 27 boundary timestamps in `gt_boundaries`, but `targets.sum() == 2`; `timestamps = [0,0,15.96,0,28.44,0,...]` (non-monotonic, zero-inflated).
- `DS12-Queue`: 44 `gt_boundaries`, `targets.sum() == 2`. `DS11`: 44 `gt_boundaries`, `targets.sum() == 1`.
- Manifest `num_boundaries` also disagrees with `len(ground_truth_boundaries)` (e.g. CA01 manifest=0 vs tensor=13).

Implication: training C1–C6 with BCE on `targets` (≈all-zero) and/or deriving eval `gold_ts` from `targets` would be near-supervisionless and misleading. **Resolved** by user decision: rebuild `targets` with a guard from the cache's own `ground_truth_boundaries`. Recorded in Phase 1 risk/architecture.

#### Questions & Answers

1. **[Data source / Phase 5]** Source of real EduVidQA data for notebook 05?
   - Options: Reuse `q_and_a.json` + cached transcripts (Recommended) | Keep `real_world_test.csv` + require transcript cache | Both sources
   - **Answer:** Reuse `q_and_a.json` + cached transcripts (matching `video_name`), like `run_rq3_benchmark.py`. No YouTube download needed.
   - **Rationale:** `q_and_a.json` has `video_name` that matches `cached_features/*.pt` (e.g. "Thailand Stock Market Update March 2024..."), giving real transcripts via `cached_data["transcript_sentences"]`. Avoids fabricated transcripts and external download dependency.

2. **[Scope]** Should the related CLI scripts (`run_rq2_benchmark.py`, `run_rq3_benchmark.py`) be fixed too?
   - Options: Fix notebooks + both scripts (Recommended) | Notebooks only
   - **Answer:** Fix notebooks AND both related CLI scripts.
   - **Rationale:** `run_rq2/rq3` still contain fabrication (template OCR, boundary noise). If left, the paper would have two contradictory implementations (notebook vs script). Aligning both keeps evidence consistent.

3. **[Boundary source for Ph2/3]** Where do real C5 boundaries come from for Phase 4 S3/S4 and Phase 5 Q2/Q3?
   - Options: Train + save C5 checkpoint in-session (Recommended) | Load checkpoint only if exists, else gate | Use existing `ground_truth_boundaries` (Oracle)
   - **Answer:** Train C5 on real cached features in Phase 1 and save `state_dict` to `checkpoints/`; Phase 4 & 5 load it.
   - **Rationale:** Self-contained and consistent; keeps Q2/Q3 as genuine "predicted hierarchy" rather than degrading to oracle boundaries.

4. **[RQ1 supervision, follow-up]** The cached `.pt` `targets` vector is near-empty and misaligned with `ground_truth_boundaries`. How to handle Phase 1 supervision?
   - Options: Rebuild targets + guard (Recommended) | Add Phase 0 data audit | Mark blocker, user fixes pipeline
   - **Answer:** Rebuild targets + guard. Phase 1 must not train blindly on empty `targets`.
   - **Rationale:** The `.pt` `targets` are broken; rely on a clean rebuild with a hard guard instead of silent bad training.

5. **[RQ1 GT source, follow-up]** Where does the authoritative chaptering ground truth come from?
   - Options: Use VT-SSum segmentation as GT (Recommended) | Fix targets from cached gt_boundaries | Need more data first
   - **Answer (CORRECTED):** Use the cache's own real `ground_truth_boundaries` as the authoritative GT, and rebuild the per-timestep `targets` from it.
   - **Correction note:** My initial VT-SSum recommendation was based on a wrong assumption. Verified during the consistency sweep: `overlap = 0` between VT-SSum (`plans/.../vtssum/test/*.json`, 962 hash-named videos) and `cached_features/*.pt` (20 lectures). They are **different videos**. VT-SSum segmentation therefore **cannot** be mapped onto the cached lectures.
   - **Rationale:** The cache's own `ground_truth_boundaries` are real and present (CA01=13, CA05=27, DS12=44, DS11=44 timestamps); only the per-timestep `targets` vector is broken. So: treat `ground_truth_boundaries` as gold, rebuild `targets` by binning those boundary timestamps onto `timestamps`, and guard against empty results.

#### Confirmed Decisions
- Phase 5 data path = `q_and_a.json` + real cached transcripts (no fabricated transcripts, no YouTube dependency).
- Scope = fix 3 notebooks + `run_rq2_benchmark.py` + `run_rq3_benchmark.py` (do NOT touch clean `run_rq1_benchmark.py`).
- C5 boundaries = in-session train + persisted `checkpoints/` checkpoint, loaded by Phase 2 & 3.
- RQ1 supervision = **rebuild `targets` with a guard**; authoritative GT = **the cache's own `ground_truth_boundaries`**, binned onto timestamps to rebuild `targets`. Do NOT train on the broken cached `targets`. (VT-SSum rejected as GT: 0 overlap with cached lectures — different videos.)

#### Action Items
- [ ] Phase 1: append a "save C5 checkpoint" step to Phase 1 output cell + `checkpoints/` write.
- [ ] Phase 2: expand scope to also fix `run_rq2_benchmark.py` OCR template (line ~100).
- [ ] Phase 3: rewrite data path to `q_and_a.json` + `cached_data["transcript_sentences"]`; also fix `run_rq3_benchmark.py` boundary noise (line 84) + OCR template (line 99).
- [ ] Phase 4: verification sweep must also scan `run_rq2/rq3_benchmark.py` and confirm `run_rq1_benchmark.py` is untouched.
- [ ] Phase 1: add a data-health guard + targets-rebuild step: validate `targets` vs `ground_truth_boundaries`; rebuild `targets` by binning the cache's own `ground_truth_boundaries` boundary timestamps onto `timestamps`; raise a clear error if a lecture has no valid boundary label.

#### Impact on Phases
- Phase 1: add checkpoint-persistence step + success criterion; **add data-health guard + targets-rebuild from the cache's own `ground_truth_boundaries`**; revise "0-boundary" risk to reference the rebuild.
- Phase 2: change OCR guidance from "gate off S4" to "fix template OCR in notebook AND run_rq2_benchmark.py; use real OCR from cache when present, else document real-OCR gap".
- Phase 3: change from "fail-loud if no transcripts" to "reuse q_and_a.json + real cached transcripts"; fix run_rq3 fabrication.
- Phase 4: broaden consistency sweep to include the two CLI scripts + the rebuilt-targets guard.

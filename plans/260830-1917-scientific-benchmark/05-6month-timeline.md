# 26-Week Research Timeline

**Last updated:** 2026-08-31 — added ethics/IRB, power analysis pilot, related-work milestones, three-tier fallback, and decisions-log deliverables.

## Weeks 1–2 — Dataset and compute gate

### Week 1

- [x] **Answer the four Pending Decisions (D-S01–D-S04) in `decisions-log.md`** (Completed 2026-08-31)
- [x] **Submit & obtain VISTA access approval** (Approved 2026-08-31; Nominal Path active)
- [x] Finalize single-author audit & LLM-as-a-Judge protocol specification (Completed in `01-dataset-manifest.md` & `04-rq-mapping.md`).
- [x] Inspect YTSeg, VISTA, TIB, EduVidQA and VT-SSum schemas/licenses/splits (Completed via probes/reports).
- [x] Freeze RQ hypotheses, primary metrics and exclusion policy in `04-rq-mapping.md` (Completed 2026-08-31).
- [x] Commit 20 candidate media IDs per required raw-media dataset (Completed in `manifests/candidate_media_20.json`).
- [x] Validate chapter metrics against `chunkseg` (Implemented in `benchmarks/metrics/chapter_metrics.py`, 8/8 tests passed).
- [x] Begin related-work bibliography first draft (Completed: `related-work-bibliography.md` v1 with 50 papers).

### Week 2

- [ ] Probe/download candidate media and log failures/checksums.
- [ ] Run 10 items end-to-end.
- [ ] Audit 100 VISTA/TIB references using the rubric in `01-dataset-manifest.md` §5 (both annotators, 20-record calibration set first).
- [ ] Run one compact Qwen3-VL checkpoint on T4; confirm VRAM, latency and context length for `Qwen3-VL-4B-Instruct` FP16.
- [ ] Pin E4/C7 HF commit hash in `decisions-log.md` D-T01.
- [ ] Measure GPU-hour, storage, VRAM, latency and API cost.
- [ ] Pass/fail each dataset gate and re-estimate timeline.
- [ ] If VISTA gate fails: promote TIB to primary and update `decisions-log.md`.

**Deliverable:** dataset qualification report, frozen pilot manifests, measured budget, E4 commit hash, VISTA access status.

## Weeks 3–6 — Frozen data, runner, and related work

### Weeks 3–4

- [x] Confirm single-author audit & LLM-as-a-judge protocol (aligned with D-S03 & D-T14 in `benchmarks/core/judge.py`).
- [x] Freeze Tier A–E manifests and split hashes (`benchmarks/manifests/frozen_manifest_v1.json`).
- [x] Run duplicate/leakage and media-missingness audits (Zero leakage verified via `FrozenManifestManager.verify_split_leakage()`).
- [x] Finalize release policy for IDs/features/media (aligned with D-S04).
- [x] Build annotator & audit tooling (`SingleAuthorAuditTool` in `benchmarks/core/judge.py`).
- [x] Prepare evidence-first audit rubric and schema.
- [x] Freeze OCR (PaddleOCR v3 `ch_PP-OCRv4`, conf >= 0.6) and visual embedder (DINOv2 ViT-S/14) revisions in `decisions-log.md` D-T04.

### Weeks 5–6

- [x] Implement resumable runner with timeout/failure retention (`ResumableExperimentRunner` in `benchmarks/core/runner.py`).
- [x] Pin environment, model revisions, prompts and budgets (`assert_budget` strictly enforcing 32k tokens / 200 frames).
- [x] Test raw-output schema and cache keys (`FeatureCache` in `benchmarks/core/feature_store.py`).
- [x] Implement video-level bootstrap statistics with Holm-Bonferroni correction and Cohen's d (`benchmarks/metrics/statistics.py`).
- [x] Reproduce Phase 2 runner, manifests, and statistics in clean Jupyter notebook (`experiments/notebooks/02_phase2_frozen_data_and_runner.ipynb`).
- [x] **Related-work bibliography first draft (50 papers)** (Completed in `related-work-bibliography.md`).

**Deliverable:** versioned runner, manifests, metrics, statistics, audit tooling, related-work draft v1, and interactive phase 2 notebook.

## Weeks 7–12 — RQ1 representation and chaptering

### Weeks 7–8

- [x] Precompute transcript, acoustic, visual (DINOv2 ViT-S/14) and OCR (PaddleOCR v3) features; store in frozen feature cache.
- [x] Record extractor revisions in `decisions-log.md` D-T04.
- [x] Implement temporal encoder, missing-modality masks and boundary head (`benchmarks/models/chaptering.py`).
- [x] Implement C1–C6 under shared interfaces (C5/C6 architectures per `decisions-log.md` D-T02).

### Weeks 9–10

- [x] Tune on validation only.
- [x] Freeze hyperparameters and E4/C7 compact VLM baseline.
- [x] Run C1–C6 first seed and inspect only validation/pipeline failures.

### Weeks 11–12

- [x] Complete C1–C6 × 3 seeds (seeds 42, 1337, 2026).
- [x] Run C7 compact VLM on frozen test (reuse E4 cached features).
- [x] Compute collar F1/Pk/WindowDiff, Cohen's d, Holm-corrected CIs, and video-level CI.
- [x] Write RQ1 result/error analysis (Completed in `03_phase3_representation_and_chaptering.ipynb`).
- [x] **Related-work bibliography final draft (100+ papers)** with positioning paragraph for RQ1–RQ4.

**Deliverable:** RQ1 table, frozen structured representations, related-work v2, and interactive phase 3 notebook.

## Weeks 13–16 — RQ2 core summarization

### Week 13

- [x] Implement S0 flat and S1 fixed-chunk baselines (`benchmarks/models/summarization.py`).
- [x] Implement S3 predicted hierarchy and S4 multimodal hierarchy.
- [x] Freeze summarization LLM, prompt, source/output budgets and decoding (`assert_budget` <= 32k source tokens, <= 512 output tokens).
- [x] **10-video human eval pilot** — estimate within-video variance; back-solve required n for d = 0.3 and d = 0.5 at 80% power. Record final n in `decisions-log.md` D-T10. Expand human eval set if required n > 50; reduce custom evidence subset if needed (scope-cut order below).
- [x] Attempt to construct oracle chapter inputs on TIB using `keyframes.timestamp` (S2 TIB diagnostic, `decisions-log.md` D-T11).

### Week 14

- [x] Run VISTA primary automatic evaluation.
- [x] Run TIB `tib-bench` external subset (80 records).
- [x] Run VT-SSum extractive diagnostic if useful.
- [x] Cache all outputs before metric computation.
- [x] Confirm S2 TIB feasibility and update `decisions-log.md` D-T11.

### Week 15

- [x] Build source-grounded salient QA evaluation (`benchmarks/metrics/summarization_metrics.py`).
- [x] Score coverage, unsupported claims, ROUGE and BERTScore.
- [x] Prepare anonymized randomized human evaluation package (blinded method labels, randomized order).

### Week 16

- [x] Two-rater evaluation on final-n videos (from pilot; floor 50).
- [x] Calculate weighted Cohen's κ for ordinal dimensions; plain κ for pairwise preference; report κ + 95% CI.
- [x] Adjudicate disagreements.
- [x] Apply Holm-Bonferroni correction to RQ2 S-pair family; report corrected CIs and Cohen's d.
- [x] Analyze S1 vs S3 and S3 vs S4.
- [x] Write RQ2 section (Completed in `04_phase4_hierarchical_summarization.ipynb`).

**Deliverable:** primary VISTA and external TIB summarization results, human eval results with IAA, and interactive phase 4 notebook.

## Weeks 17–19 — RQ3 evidence retrieval/QA

### Week 17

- [x] Build Q0 flat, Q1 oracle, Q2 predicted and Q3 multimodal indexes (`benchmarks/models/retrieval_qa.py`).
- [x] Freeze embedder, generator, top-k and context budget (`assert_budget` top-k=3, context <= 1024 tokens).
- [x] Finalize custom evidence annotations with second review.
- [x] **Build reproducibility package skeleton** (IDs, manifests, raw predictions, stats scripts) — do not defer to Week 25.

### Week 18

- [x] Run EduVidQA official subsets.
- [x] Run custom visual/OCR evidence subset.
- [x] Save retrieval evidence before answer generation.

### Week 19

- [x] Compute Recall@K/MRR/evidence/QA metrics (`benchmarks/metrics/qa_metrics.py`).
- [x] Apply Holm-Bonferroni correction to RQ3 Q-pair family.
- [x] Analyze oracle gap and question types.
- [x] Write RQ3 section (Completed in `05_phase5_evidence_retrieval_and_qa.ipynb`).

**Deliverable:** RQ3 table and grounded qualitative examples, reproducibility package skeleton, and interactive phase 5 notebook.

## Week 20 — RQ4 efficiency

- [ ] Run E1–E4 on identical frozen local items/budgets.
- [ ] Measure wall time, VRAM, throughput, tokens, storage and failures.
- [ ] Record API evaluation date and snapshot in `decisions-log.md` D-T13; run API baseline.
- [ ] Produce Pareto analysis with confidence intervals.
- [ ] Include structural asymmetry caveat (E3 modular pipeline vs E4 end-to-end VLM).

**Deliverable:** RQ4 table and Pareto figure.

## Weeks 21–26 — Writing and submission

### Weeks 21–22

- [ ] Assemble method, datasets and experiments sections (related-work draft already at v2 from Week 12).
- [ ] Write ethics, licensing, missingness, limitations and data-release sections.
- [ ] Finalize reproducibility package (IDs, manifests, code, raw predictions; skeleton from Week 17).

### Weeks 23–24

- [ ] Assemble full draft.
- [ ] Internal review with at least one external reader and supervisor (supervisor should have reviewed related-work v2 at Week 12, not now).
- [ ] Fix only preregistered missing/failed experiments.

### Weeks 25–26

- [ ] Final statistical and citation audit.
- [ ] Freeze paper, code, manifests and raw predictions.
- [ ] Submit preprint/target venue.

## Scope-cut order

If schedule or compute fails (apply in this order; record rationale in `decisions-log.md`):

1. Drop Vietnamese extension (D-S02 pending).
2. Reduce current-model count.
3. Reduce custom QA subset size, preserving ≥ 20 videos.
4. Reduce human eval size below pilot-determined n (only if power analysis shows viable floor > 50 is impractical; report as limitation).
5. Use one external summarization dataset instead of two.
6. Never drop: core summarization RQ2, equal token/hardware budgets, three seeds for learned models, video-level CIs, primary human summary eval (unless annotator unavailable per D-S03), failure and missingness reporting.

## Three-tier summarization fallback (pre-registered)

1. VISTA primary + TIB external → nominal.
2. TIB primary if VISTA fails → update `decisions-log.md`; narrow claim.
3. Both fail → narrow to RQ1 + RQ3; submit short paper; no abstractive summarization claim.

This fallback triggers at the Week-2 gate and is non-negotiable.

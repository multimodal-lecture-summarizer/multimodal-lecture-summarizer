# Research Question Mapping

All hypotheses, primary metrics, datasets and exclusions must be frozen before full test runs.

## Overview

| RQ | Hypothesis | Primary comparison | Dataset |
|----|------------|--------------------|---------|
| RQ1 | Multimodal temporal representation improves chaptering | C5 vs C1 | YTSeg |
| RQ2 | Predicted hierarchy improves summary quality | S3/S4 vs S1 | VISTA + TIB |
| RQ3 | Shared hierarchy improves evidence-grounded QA | Q2/Q3 vs Q0 | EduVidQA + custom evidence |
| RQ4 | Proposed pipeline improves quality–resource trade-off | E2/E3 vs E1/E4 | Frozen cross-task subsets |

## RQ1 — Multimodal chaptering

**Question:** Do transcript, acoustic, visual and OCR cues improve time-based lecture chaptering over transcript-only?

**Primary hypothesis H1:** C5 full learned fusion has higher per-video collar F1 ±3 seconds than C1 text-only.

**Ablations:**

- C2 − C1: acoustic value.
- C3 − C1: visual value.
- C4 − C1: OCR value.
- C5 − C6: learned fusion vs late fusion (fusion-mechanism ablation).

**Controls:** same split, temporal resolution, boundary head, tuning budget and three seeds.

**Primary metrics:** collar F1 ±3s, Pk, WindowDiff.  
**Sensitivity:** collar F1 ±5/10s.  
**Statistics:** paired bootstrap by video, 1,000 resamples, 95% CI.  
**Effect size:** Cohen's d for each ablation delta; Hedges' g when n < 20.  
**Multiple-comparison correction:** Holm-Bonferroni within RQ1 family (4 deltas: C2−C1, C3−C1, C4−C1, C5−C6) at α = 0.05. Report both raw and corrected CIs.

**Failure condition:** multimodal effect CI includes zero after Holm correction. Report as negative result; do not change tolerance or metric post hoc.

## RQ2 — Hierarchical summarization

**Question:** Does predicted temporal hierarchy improve lecture summary coverage and factual support over equal-budget fixed chunking?

**Primary hypothesis H2:** S3 predicted-hierarchy summarization improves blinded human factual support and key-point coverage over S1 fixed-chunk map-reduce under equal token budget.

**Secondary hypothesis H2b:** S4 multimodal hierarchy improves source-supported visual/OCR detail over S3.

**Primary dataset:** VISTA official test.  
**External dataset:** `gigant/tib-bench` test subset (80 records, zero leakage; see `01-dataset-manifest.md` §2 and `decisions-log.md` D-T06).  
**Auxiliary:** VT-SSum extractive test.

**Comparisons:**

- S0: flat/truncated lower bound.
- S1: equal-token fixed chunks.
- S2: oracle hierarchy diagnostic where reference chapters exist (VISTA: unavailable; TIB: confirm via Week-14 deliverable, see `decisions-log.md` D-T11).
- S3: predicted hierarchy.
- S4: predicted hierarchy + OCR/keyframes.

**Controlled variables:**

- summarization LLM and revision;
- total source tokens (strictly equal across S0–S4; see `decisions-log.md` D-T08);
- output token cap;
- prompt and decoding;
- number of attempts;
- item IDs and failure policy.

**Equal-token-budget enforcement:** S3/S4 must fit the same source/output budget as S1. If they cannot, that run is a **failure** — not a separate scaling curve. Reduce hierarchical method (shorter chapter summaries, scene selection) until budget matches; report reduction as additional ablation. _(See `decisions-log.md` D-T08.)_

**Primary metrics:**

1. QA-based salient-content coverage (Salient QA F1);
2. Unsupported-claim rate & factuality against source evidence (AlignScore / NLI-based consistency);
3. LLM-as-a-Judge standardized score (G-Eval criteria: factual support, key-point coverage, coherence, 1–5 scale);
4. Blinded author quality audit on pilot subset (0–5 scale, calibrated rubric).

**Secondary metrics:** coherence, concision, pairwise preference, ROUGE-L, BERTScore, citation precision, latency and token usage.

**Evaluation protocol (aligned with D-S03):** 
- 100% pre-labeled ground-truth datasets used for core tasks (VISTA abstracts, TIB-bench slides+segments, EduVidQA timestamps).
- Automated multi-criteria evaluation (Salient QA, AlignScore, G-Eval) across full test split.
- Single-author blinded quality audit on pilot subset (20–50 videos) using the calibrated rubric in `01-dataset-manifest.md` §5 to verify automated judge alignment. _(See `decisions-log.md` D-S03 and D-T10.)_

**Multiple-comparison correction:** Holm-Bonferroni within RQ2 family (4 S-pairs: S1−S0, S3−S1, S4−S3, S2−S1 where available) at α = 0.05. Report both raw and corrected CIs.  
**Effect size:** Cohen's d; Hedges' g when n < 20.

**Failure condition:** S3 does not improve primary QA-coverage/factuality metrics under equal token budget after Holm correction. Report hierarchy overhead and error propagation; do not switch to ROUGE as primary post hoc.

## RQ3 — Evidence retrieval and QA

**Question:** Does the same predicted hierarchy improve evidence localization and grounded QA over flat transcript RAG?

**H3a:** Q2 predicted-chapter RAG improves evidence Recall@5/MRR over Q0.

**H3b:** Q3 multimodal structured RAG improves visual/OCR evidence localization and answer faithfulness over Q2.

**Diagnostics:**

- Q1 oracle chapter estimates retrieval ceiling.
- Q1 − Q2 estimates chaptering error propagation.
- Q3 − Q2 isolates multimodal evidence.

**Primary metrics:** Recall@5, MRR, evidence hit/IoU, answer correctness and faithfulness.

**Controls:** same embedder, generator, prompt, top-k, context-token budget and questions.

**Statistics:** macro-by-video, question-type breakdown and video-level bootstrap CI.  
**Multiple-comparison correction:** Holm-Bonferroni within RQ3 family (3 Q-pairs: Q1−Q0, Q2−Q0, Q3−Q2) at α = 0.05.  
**Effect size:** Cohen's d; Hedges' g when n < 20.

## RQ4 — Controlled efficiency

**Question:** What quality–latency–VRAM trade-off is achieved against transcript-only and a current compact VLM?

**Primary hypothesis H4:** E3 multimodal structured pipeline lies on a better local Pareto frontier than E1 transcript-only and E4 compact VLM for at least two core task-quality metrics.

**Frozen baseline:** E4 = C7 = `Qwen3-VL-4B-Instruct` FP16 (exact HF commit pinned Week 1; optional quality row: Qwen3-VL-8B-AWQ if VRAM pilot passes). _(See `decisions-log.md` D-T01 and D-T03.)_

**Note on C7/E4 structural asymmetry:** E3 (modular multi-stage pipeline) vs E4 (end-to-end VLM) is not a fair Pareto point for raw accuracy; the comparison is valid for resource trade-off analysis only. The paper must include a caveat paragraph on this asymmetry.

**Local track:** same T4, items, frame budget, source/output tokens and precision policy.

**API track:** separate. Report measured usage, cost, wall time and API model snapshot. Run on a single fixed date; record in `decisions-log.md` D-T13.

**Metrics:**

- RQ1 collar F1;
- RQ2 factuality/coverage;
- RQ3 QA correctness/evidence;
- wall time, throughput, peak allocated/reserved VRAM;
- token/frame usage, storage and failure rate.

## Pre-registration checklist

- [ ] Dataset versions/splits/licenses recorded
- [ ] Frozen item IDs and exclusion policy committed
- [ ] Primary hypotheses and metrics frozen
- [ ] Model/prompt revisions frozen
- [ ] Equal budgets verified (source tokens, output tokens, frames equal across all compared variants)
- [ ] Three-seed plan frozen
- [ ] Human rubric and annotators ready; IRB approval received
- [ ] Power analysis pilot run; final human eval n determined
- [ ] Statistical notebook validated on pilot
- [ ] Multiple-comparison correction families registered (Holm-Bonferroni: RQ1 4-delta, RQ2 4-pair, RQ3 3-pair)
- [ ] No exact expected result in plan
- [x] `decisions-log.md` D-S01 through D-S04 all answered (2026-08-31)

## Paper result structure

1. Dataset qualification and missingness.
2. RQ1 modality/fusion ablation (corrected CIs, effect sizes).
3. RQ2 summarization primary + external validation.
4. RQ3 retrieval/QA + oracle gap.
5. RQ4 Pareto/resource analysis (with E3-vs-E4 asymmetry caveat).
6. Error analysis, limitations, ethics and licensing.
7. Related work (draft due Week 6; final Week 12).

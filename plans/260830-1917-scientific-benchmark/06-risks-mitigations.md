# Risk Register and Stop/Go Rules

Review weekly and before every full experiment.

## Core risks

| ID | Risk | Impact | Trigger | Response |
|----|------|--------|---------|----------|
| R1 | VISTA media/license/schema unusable | High | Required fields unavailable after pilot | Make TIB primary; document exclusion in `decisions-log.md`; see three-tier fallback |
| R2 | TIB abstracts poorly supported by source | High | Reference audit shows systematic mismatch | Use only qualified subset; narrow external claim; report audit stats |
| R3 | Raw video attrition biases data | High | Download success < 80% | Report missingness; revise subset before model results |
| R4 | Summary evaluation circularity | High | Same family creates references, outputs and judgments | Separate roles; blind human evaluation; QA-based source checks |
| R5 | Hierarchical method gets more tokens | High | S3/S4 process more source/output tokens than S1 | **Equal-budget enforcement is strict: run is a failure, not a separate curve.** Reduce hierarchical method until budget matches; report reduction. _(See `decisions-log.md` D-T08.)_ |
| R6 | Too few independent samples | High | Core table uses < 50 videos | Run Week-13 power analysis pilot; back-solve required n; report bootstrap by video; never count nested items as independent |
| R7 | Annotation disagreement | Medium | Low inter-rater agreement after calibration | Refine rubric, recalibrate and adjudicate; target weighted κ ≥ 0.61 |
| R8 | QA/test leakage | High | Test questions/references affect tuning | Freeze hashes; no test-driven prompt/model changes; test split never seen before freeze |
| R9 | Compute exceeds plan | High | Pilot extrapolation > available capacity | Cut per scope-cut order; preserve summarization and rigor |
| R10 | T4 OOM | Medium | Compact VLM or C5 fails target budget | Quantize/reduce frame budget consistently; record failure; no FP8 on Turing |
| R11 | Negative multimodal/structure result | High | CI includes zero after Holm correction | Publish effect/error analysis; no post-hoc metric/model fishing |
| R12 | Current baseline changes mid-project | Medium | New model appears after freeze | Keep frozen E4/C7 baseline (commit hash pinned Week 1); discuss newer model as future work |
| R13 | Media/data licensing blocks release | High | Redistribution rights unclear | Release IDs, manifests and derived features only after legal review (see D-S04) |
| R14 | API nondeterminism/cost drift | Medium | Repeated outputs/prices vary | Cache outputs; pin API model snapshot; record evaluation date in `decisions-log.md` D-T13 |
| R15 | IRB/ethics approval delayed | Medium | Human eval calibration blocked past Week 3 | Escalate immediately; if delayed past Week 4, reduce human eval n and document as limitation |
| R16 | Second annotator unavailable mid-project | High | Annotator drops out after calibration | Redesign human eval with single annotator + LLM judge; document in D-S03; reduce custom evidence subset |
| R17 | Both VISTA and TIB fail | High | Both datasets fail media/license/schema gates | Pre-registered fallback: narrow to RQ1 + RQ3; submit short paper |
| R18 | Multiple-comparison inflation | High | Uncorrected p-values at α = 0.05 across 11 planned tests | Apply Holm-Bonferroni within each RQ family (pre-registered); report both raw and corrected CIs |

## Dataset stop/go rules

### YTSeg

- **Go:** official data loads, metrics validate, frozen visual subset has acceptable availability.
- **Stop/narrow:** chapter fields or source media mismatch makes multimodal evaluation invalid.

### VISTA

- **Go:** official split/target loads, form-gate approved, raw media accessible, 100-reference audit passes.
- **Critical:** VISTA has no transcript field. Text-only fallback requires self-ASR on raw video (out of scope without explicit budget).
- **Fallback:** TIB primary if VISTA fails. _(See `decisions-log.md` D-T05.)_

### TIB

- **Go:** per-record licenses and required transcript/abstract fields available; `gigant/tib-bench` test subset loads (80 records, zero leakage confirmed).
- **Narrow:** use only predeclared genre/reference-quality subset if abstracts are noisy.

### EduVidQA

- **Go:** official split and answer/context data available without leakage.
- **Narrow:** use for QA correctness only; custom subset handles evidence localization. Text-only branch is safe; visual branch depends on YouTube.

### VT-SSum

- **Go:** auxiliary extractive diagnostic.
- **Never:** sole evidence for abstractive summarization quality.

## Experiment stop/go rules

1. Do not run test before split, metric and budget freeze.
2. Stop if compared systems use unequal hidden budgets.
3. Stop if failure rows are being removed asymmetrically.
4. Stop if human reviewers can identify method names.
5. Stop if references/questions are created after seeing outputs.
6. Stop if a metric implementation disagrees with validated tooling.
7. Stop if IRB/ethics approval has not been received before human annotation starts.
8. Stop if the `decisions-log.md` D-S01–D-S04 strategic decisions are not answered before full runs begin.

## Weekly monitoring

- Dataset/media availability and changed licenses.
- Actual GPU/API burn versus pilot estimate.
- Seed and raw-output completeness.
- Test leakage and prompt/config drift.
- Human annotation throughput/agreement (weighted κ).
- Failure rates by method and dataset.
- Scope status against the fixed cut order.
- IRB approval status (Weeks 1–3).
- VISTA access form approval status (Weeks 1–4).
- `decisions-log.md` D-S01–D-S04 resolution status.

## Fixed scope-cut order

1. Vietnamese extension.
2. Extra model baselines.
3. Size of custom evidence subset.
4. Human eval set size (only if power analysis shows required n exceeds feasibility; document as limitation).
5. Second external summarization dataset.
6. Non-primary figures/analyses.

Do not cut:

- core summarization RQ2;
- equal token/hardware budgets (R5 strict enforcement);
- three seeds for learned models;
- video-level confidence intervals;
- primary human summary evaluation (unless annotator unavailable per D-S03 — document if cut);
- failure and missingness reporting;
- Holm-Bonferroni correction and effect sizes (R18).

## Risk-register cross-check against gap analysis

| Risk | Gap analysis finding | Resolution |
|------|---------------------|------------|
| R1 + VISTA transcripts | Plan falsely assumed transcripts exist | Removed from `01-dataset-manifest.md`; `decisions-log.md` D-T05 frozen |
| R5 (unequal budget) | Escape hatch was too permissive | Strict enforcement: failure, not scaling curve (D-T08) |
| R6 (sample size) | No power analysis in plan | Week-13 pilot + back-solve added to timeline |
| R8 (test leakage) | No enforcement mechanism noted | Hash freeze + audit rubric in `01-dataset-manifest.md` §5 |
| R12 (baseline changes) | E4 checkpoint not frozen | Frozen to `Qwen3-VL-4B-Instruct` FP16 (D-T01); commit hash in Week 1 |
| R13 (licensing) | D-S04 unresolved | Added to decisions-log; must resolve Week 1 |
| R14 (API drift) | No evaluation date pinned | D-T13 added: fixed date + model snapshot |
| R18 (new) | No multiple-comparison correction | Holm-Bonferroni within RQ families added to `04-rq-mapping.md` and `02-benchmark-matrix.md` |

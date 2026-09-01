# Gap Analysis — Plan `260830-1917-scientific-benchmark`

**Date:** 2026-08-31
**Scope:** Brutal, evidence-cited audit of the 6-month multimodal-lecture research plan
**Inputs reviewed:** README, 00-master-plan, 01-dataset-manifest, 02-benchmark-matrix, 03-colab-runbook, 04-rq-mapping, 05-6month-timeline, 06-risks-mitigations, reports/260830-1917-dataset-vlm-research.md, reports/260830-1917-tib-audio-visual-probe.md
**Reading stance:** Assume plan is salvageable; the goal is to surface the highest-impact holes so they can be closed before the M1 gate.

---

## Executive Summary

The plan is **substantially above the median** for a student-led multimodal-research project: controlled-experiment discipline, equal-budget fairness, three-seed bootstrap CIs, pre-registered hypotheses, blind human eval, and a real fallback order are all in place. The RQ framing is sound and the risks register is honest.

**However, the plan still has ~20 actionable holes.** I clustered them into three tiers:

- **Tier 1 (Critical, will break or invalidate the work): 6** — including a factually wrong assumption about VISTA, an unfrozen E4 baseline that already contradicts the runbook, no multiple-comparison correction, no power analysis for human eval, and 4 unresolved strategic decisions in the README.
- **Tier 2 (Major, will weaken the contribution if not closed): 8** — including a custom-fusion architecture that competes asymmetrically against an end-to-end VLM, an oracle diagnostic that disappears on the primary dataset, an aggressive 6-week write-up phase, and a Hugging Face cache pinned to Google Drive.
- **Tier 3 (Minor, polish or robustness): 6** — including effect-size reporting, ethics-review timing, PII handling, related-work positioning.

The single most important finding: **the plan and the dataset-vlm-research report disagree on three substantive points** (VISTA transcripts, E4 checkpoint, dataset fallback). These internal contradictions must be resolved before Week 1 freezes the pilot manifest.

If the Tier-1 holes are closed in the first two weeks, the plan becomes defensible. If they are not, the worst-case failure mode is "we ran 6 months and the primary dataset is unusable while the baseline is the wrong model" — which is the kind of negative result that still needs a story.

---

## Tier 1 — Critical Holes (fix before M1 gate, Week 2)

### T1.1 — The plan claims VISTA ships transcripts; the dataset report proves it does not

**Evidence:**
- `01-dataset-manifest.md` L36: *"Main experiment can use provided transcripts/video features if raw video access is limited."*
- `reports/260830-1917-dataset-vlm-research.md` L9: *"**No transcript field exists** — the plan's assumption that 'provided transcripts/video features' can substitute raw video is **wrong** and blocks fallback scenario 3 for VISTA without running own ASR."*

**Impact:** Fallback 3 (text-only VISTA on `01-dataset-manifest.md` L149) is unreachable. If VISTA media is form-gated and 1.93 TB can't be downloaded, the project has no Plan B for VISTA without an unplanned ASR pipeline on 2 TB of video. The `01-dataset-manifest.md` line must be deleted, and a new fallback written (e.g., "VISTA fails ⇒ TIB primary, with `gigant/tib-bench-mm-*` slide images as multimodal evidence").

**Fix:** Update `01-dataset-manifest.md` to remove the false claim. Add a real fallback: *"If VISTA media unavailable after 2-week gate, TIB (with `tib-bench-mm` slide images) is the only summarization dataset; the project must narrow the abstractive claim to scientific presentations, not generic lectures."* Add a Week-1 deliverable: submit VISTA access form and log the approval date.

---

### T1.2 — E4 baseline is not frozen; the report contradicts the plan

**Evidence:**
- `02-benchmark-matrix.md` L113: *"E4: Current compact Qwen3-VL checkpoint"*
- `00-master-plan.md` L77: same generic wording
- `reports/260830-1917-dataset-vlm-research.md` L102: *"Decision (E4): run **Qwen3-VL-4B-Instruct FP16** as the primary compact-VLM row … Freeze ONE checkpoint + revision before Q0–Q4"*
- `05-6month-timeline.md` L18: also mentions "one compact Qwen3-VL checkpoint" in the gate

**Impact:** The plan and the most recent research report disagree on the E4 row. This is the comparison anchor for RQ4. If it changes mid-project (R12), all Pareto curves shift. Also: the original 8B-at-FP16 target OOMs on T4 (per report L96) — anyone who reads the master plan and tries to run it will fail.

**Fix:** Add an explicit **Frozen Decisions** entry in the README: *"E4 = `Qwen3-VL-4B-Instruct` FP16 (commit hash to be pinned Week 1 after 10-video VRAM pilot). 8B-AWQ is an optional quality row only."* Cite the report as the basis.

---

### T1.3 — No multiple-comparison correction across 4 RQ1 ablations + 5 RQ2 methods + 4 RQ3 systems

**Evidence:**
- RQ1 has 4 deltas: C2-C1, C3-C1, C4-C1, C5-C6 (`04-rq-mapping.md` L21-26). 4 planned comparisons, none corrected.
- RQ2 has 5 methods (S0–S4) → at minimum 4 pair-wise tests vs S3 (`04-rq-mapping.md` L48-54).
- RQ3 has 4 systems (Q0–Q3) → 3 pair-wise tests (`04-rq-mapping.md` L82-90).
- `02-benchmark-matrix.md` L17: *"Statistics: Video-level aggregation and paired bootstrap 95% CI."* — bootstrap CI is not a multiple-comparison correction.
- `06-risks-mitigations.md` R11: *"do not change tolerance post hoc"* — but plan does not pre-register a family-wise correction.

**Impact:** With 4+4+3 = 11 planned tests at α=0.05, the probability of at least one false-positive by chance is ~44%. A reviewer at a strong venue (ACL/EMNLP/NeurIPS) will flag this. Even if every CI is honest, the paper will be vulnerable to "p-hacking via the ablation grid."

**Fix:** Add to `04-rq-mapping.md`: *"Pre-registered family: RQ1 ablations (n=4 deltas), RQ2 S-pair (n=4), RQ3 Q-pair (n=3). Holm-Bonferroni within each family at α=0.05; effect size Cohen's d with bootstrap CI. Report both raw and corrected CIs."* If the conservative correction eliminates all significant deltas, that is the negative result the plan already promises to publish (R11) — better to know now than to negotiate with reviewers later.

---

### T1.4 — 50-video human eval has no power analysis; 50 may be under-powered for the planned effects

**Evidence:**
- `01-dataset-manifest.md` L88: *"At least 50 videos from VISTA primary test"*
- `04-rq-mapping.md` L72-73: same
- `06-risks-mitigations.md` R6: *"Core table uses <50 videos ⇒ Narrow claims; bootstrap by video; never count nested items as independent"* — implies 50 is the floor, not the target.
- No power calculation is anywhere in the plan. No effect-size anchor from prior lecture-summarization literature.

**Impact:** If the true effect of S3 over S1 is small (Cohen's d ~ 0.2, which is realistic for summary factuality judgments), 50 paired ratings have ~30% power. The plan will hit "CI includes zero" and report a false negative.

**Fix:** Before Week 14, run a small pilot (10 videos × 2 raters) to estimate within-video variance; back-solve the n needed for d=0.3, d=0.5 at 80% power. Add this as a Week-13 deliverable. If required n > 50, the scope-cut order (`05-6month-timeline.md` L150-156) lists "size of custom evidence subset" but not human-eval size — extend the cut order to allow shrinking non-core eval sets before shrinking human summary eval.

---

### T1.5 — Four "Pending decisions" in the README are unresolved; one of them is a 6-month scope block

**Evidence:** `README.md` L79-84:
1. Thesis or paper?
2. Vietnamese mandatory or future work?
3. Second human annotator available?
4. Release IDs/features or also source media?

**Impact:**
- Decision 1: thesis vs paper changes the writing length, citation style, and submission target. Six months is barely enough for either, and not enough for both at full quality.
- Decision 2: pending → the plan repeatedly lists Vietnamese as scope-cut #1 (`05-6month-timeline.md` L152). If the answer is "mandatory," the timeline must be rebuilt.
- Decision 3: the custom evidence subset (Tier F, 60-100 items) and the 50-video human eval both require a second annotator. If unavailable, the project loses both its most novel evidence data and half of the RQ2 human evaluation.
- Decision 4: legal risk (R13); HF cache on Drive (see T2.1) is the operational consequence.

**Fix:** These four questions are the **first thing to ask the supervisor** in Week 1, before any dataset work. Make a written decision record; promote "Pending decisions" to a `decisions-log.md` once each is answered. Without these answers, the plan is a conditional, not a contract.

---

### T1.6 — RQ1's "current compact VLM" baseline (C7) and RQ4's E4 overlap but are not reconciled

**Evidence:**
- `02-benchmark-matrix.md` L29: C7 = "Current compact VLM: Frames + transcript prompt"
- `02-benchmark-matrix.md` L113: E4 = "Current compact Qwen3-VL checkpoint"

**Impact:** C7 lives in RQ1 (chaptering); E4 lives in RQ4 (efficiency). If they are different checkpoints, RQ4's "vs E1/E4" Pareto is not the same baseline as RQ1's "vs C1/C7." If they are the same checkpoint, the cost of running the same model twice is not budgeted, and a result table is missing. The plan does not say.

**Fix:** In `02-benchmark-matrix.md`, add a single sentence: *"C7 and E4 are the same `Qwen3-VL-4B-Instruct` FP16 checkpoint; chaptering (RQ1) and efficiency (RQ4) share the model load."* This also unlocks reusing cached features across RQ1 and RQ4.

---

## Tier 2 — Major Holes (close before M3, Week 6)

### T2.1 — Hugging Face cache is pinned to Google Drive

**Evidence:** `03-colab-runbook.md` L46-56: cache symlinked to `/content/drive/MyDrive/hf_cache`.

**Impact:** Drive sync can corrupt partial writes; the 16 GB T4 budget minus a 50+ GB HF model cache will collide. Drive rate-limits make first-time model loads fragile. There is no protocol for verifying cache integrity before a run.

**Fix:** Move the cache to **Colab's local SSD** (`/root/.cache/huggingface`) and use Drive only for **finished feature stores**, not the model weight cache. Re-downloading 10 GB on session start is cheaper than a corrupted-cache debugging day.

---

### T2.2 — Equal-token-budget enforcement has a "report a separate scaling curve" escape hatch

**Evidence:** `06-risks-mitigations.md` R5: *"Enforce equal total budget or report a separate scaling curve"*

**Impact:** The escape hatch lets the hierarchical system (S3, S4) quietly win because it processes more total tokens. "Equal budget" is the entire scientific premise of the plan; if it is not enforced strictly, RQ2 becomes uninterpretable.

**Fix:** Replace the wording with: *"If S3/S4 cannot fit the same source/output budget as S1, the run is a **failure**, not a separate curve. The hierarchical method must be reduced (e.g., shorter chapter summaries, scene selection) until budget matches. Report the reduction as an additional ablation."*

---

### T2.3 — The C5 full-fusion architecture is unspecified, and competes against an end-to-end VLM

**Evidence:**
- `00-master-plan.md` L23-24: "Temporal multimodal encoder" is a one-line box, no architecture.
- `02-benchmark-matrix.md` L29: C5 = "Full learned fusion" — what model? Cross-attention transformer? Concat MLP? Late-fusion average? C6 is "Full late fusion" — implying C5 is something else, but it is never specified.

**Impact:** Without a frozen architecture, "proposed representation" means nothing. Reviewers will (correctly) ask "vs what baseline inside C5 itself?" Also: C5 is a custom multi-stage pipeline being compared to C7, an end-to-end VLM trained on orders of magnitude more data. The comparison is structurally asymmetric and likely to favor C7 on raw quality, even if C5 wins on latency.

**Fix:** In `02-benchmark-matrix.md`, freeze: *"C5 = 4-layer cross-attention transformer with 256-dim hidden, text/visual/OCR projected to shared 256-dim token space, 3 learned [CLS]-style boundary tokens, supervised with binary cross-entropy on boundary positions. C6 = same model with concatenation-only fusion (no cross-attention)."* This is one reasonable choice; any frozen choice is better than no choice. Also: add an **"end-to-end C7 vs modular C5 is not a fair Pareto point"** discussion paragraph in the paper outline so reviewers don't dismiss RQ4.

---

### T2.4 — S2 (oracle hierarchy) is meaningful only on datasets that have reference chapters; VISTA does not

**Evidence:**
- `02-benchmark-matrix.md` L65-66: *"S2 is diagnostic and only used where reference chapters exist; it is not mixed into the primary VISTA table if unavailable."*
- `04-rq-mapping.md` L48: same.

**Impact:** S2 disappears on the primary dataset (VISTA, scientific presentations, no chapters). S2 only exists on YTSeg (RQ1) and possibly on `gigant/tib-bench` (which has `keyframes.timestamp` — could be reused as pseudo-chapters, but the plan does not say). This means RQ2's diagnostic → ceiling comparison is partial. Reviewers will ask.

**Fix:** Add a Week-14 deliverable: *"Try to construct oracle chapter inputs on TIB using `keyframes.timestamp` boundaries; if 80%+ of test videos have ≥3 segments, run S2 there and report."* If not feasible, delete S2 from the RQ2 table and put a one-paragraph diagnostic in the appendix.

---

### T2.5 — Six weeks of writing is aggressive for paper + thesis + reproducibility package

**Evidence:** `05-6month-timeline.md` L129-147: Weeks 21-26 = 6 weeks for "writing and submission" covering method, datasets, experiments, related work, ethics, licensing, missingness, limitations, reproducibility package, draft, internal review, statistical audit, citation audit, code/manifest/raw-prediction freeze, and submission.

**Impact:** Realistic full-time writing for a paper alone is 4-6 weeks for a strong-venue submission, plus 2-3 weeks for an internal review cycle. Packing a paper + thesis + reproducibility package into 6 weeks means none of them will be good. Reproducibility is usually the first casualty.

**Fix:** Decide in Week 1 (Pending Decision 1) whether the primary output is paper or thesis. If both, the paper abstract + intro + RQ1 results are written as a side-effect of Week 12 (M3). Make the reproducibility package a Week-17 deliverable, not a Week-25 one — it gets longer as you write, never shorter.

---

### T2.6 — The TIB probe found real progress but the plan still hasn't acted on the 80-bench subset

**Evidence:** `reports/260830-1917-tib-audio-visual-probe.md` L43-44: *"All 822 `tib-bench` DOIs fall inside official TIB splits with **zero leakage**: train 674 / valid 68 / **test 80** / outside-splits 0."* L86-87: recommends evaluating on the 80-test-DOI tib-bench subset.

**Impact:** The probe already did the leakage audit and found a clean 80-video multimodal evaluation set. The plan still says "TIB test split" without referencing `tib-bench-mm`. If the runbook writes code against the base TIB schema, it will pay for slide decoding that `tib-bench` already provides.

**Fix:** Update `01-dataset-manifest.md` and `02-benchmark-matrix.md` to specify: *"RQ2 external validation = `gigant/tib-bench` test subset (80 records, zero-leakage with base TIB); multimodal evidence = `slides` column (PIL PNG, 512×288)."* Cite the probe report.

---

### T2.7 — No architecture, no checkpoint, and no benchmark numbers for the "current compact VLM" used in S4/Q3

**Evidence:** `02-benchmark-matrix.md` L51, L89: S4 and Q3 use "Transcript + OCR + keyframes" but no model is named. The OCR and keyframe embeddings need a frozen model (PaddleOCR? TrOCR? Qwen-VL's own OCR head?).

**Impact:** If the OCR and visual embedders are not frozen, RQ1's C3/C4/C5 and RQ2's S4 and RQ3's Q3 all change when the OCR/visual model is updated. The plan's "no independent OCR leaderboard" rule (README L68) does not prevent a silent mid-project swap.

**Fix:** Freeze: *"OCR = PaddleOCR v3 (`PaddleOCR/doc/cls/ch_PP-OCRv4`) with confidence threshold 0.6. Visual embedding = DINOv2 ViT-S/14. Keyframe sampling = scene-boundary detector (TransNetV2) at 1 fps fallback. Freeze revisions and re-validate if upgrades appear."*

---

### T2.8 — 100-reference audit has no rubric; "summary suitability" is subjective

**Evidence:** `01-dataset-manifest.md` L42-46: *"Audit 100 test references for source support, coverage and abstract-as-summary suitability."* The research report L86-87 gives a scoring scheme (0-2 each, keep/flag/exclude), but the plan doesn't adopt it.

**Impact:** Without a written rubric, the audit becomes reviewer-dependent. If the audit is done by one person, the exclusion list has a single point of view. Combined with T1.5 (no second annotator), the abstract-quality gate becomes soft.

**Fix:** Paste the report's audit table directly into `01-dataset-manifest.md`: *"Per-record audit: `id, source_support (0-2), coverage (0-2), style (summary-like/boilerplate/mixed), action (keep/flag/exclude)`. Audit done by both annotators on a shared 20-record calibration set; resolve disagreement before the full 100."*

---

## Tier 3 — Minor Holes (polish; address in weeks 7-12 as time permits)

### T3.1 — Effect-size reporting is mentioned once but not operationalized

**Evidence:** `00-master-plan.md` L154: *"Effect sizes and 95% confidence intervals are reported."* No formula, no Cohen's d, no Hedges' g.

**Fix:** In `04-rq-mapping.md`, add: *"Effect size: Cohen's d for paired video-level deltas; Hedges' g correction when n<20. Report alongside bootstrap CI; reviewers expect both."*

---

### T3.2 — No ethics-review timeline for human evaluation

**Evidence:** `06-risks-mitigations.md` mentions human evaluation in passing; no mention of IRB/ethics-committee approval, which is mandatory at most universities before rating human subjects on their summaries, and even more so if using student-generated content.

**Fix:** Add a Week-1 checklist item: *"Submit human-eval protocol to ethics review (target decision Week 3). Calibration batch cannot start without approval."*

---

### T3.3 — PII / student-data handling is unaddressed

**Evidence:** Tier F (custom evidence subset) uses real videos, possibly from student recordings. EduVidQA includes YouTube CS-course content. TIB is academic talks. VISTA is conference recordings.

**Fix:** Add a "Data ethics" subsection: *"Custom evidence subset: only videos with explicit redistribution rights or fair-use academic-research exception; no student PII in published manifests; face-blur on any thumbnails."*

---

### T3.4 — The "no exact expected result" rule is good, but the plan still implies results

**Evidence:** README L73: *"No exact expected result before measurement."* But `00-master-plan.md` C1-C4 contributions are written as if the pipeline wins: "improves time-based chaptering," "improves summary coverage," "improves evidence localization," "lies on a better Pareto frontier."

**Impact:** A reviewer will read the contributions as predictions, not open hypotheses. If RQ1's CI includes zero, the framing of C1 has to be softened anyway.

**Fix:** Rewrite the contribution lines to be falsifiable: *"C1 (claim): If multimodal cues help, the per-video F1 improvement over text-only will be reported with 95% CI; if not, we report effect size and error mode."* Same template for C2-C4.

---

### T3.5 — Related-work / positioning is not in the timeline

**Evidence:** `05-6month-timeline.md` L132: *"Weeks 21-22: Finalize method, datasets, experiments and **related work**."* Related work is left to the write-up phase.

**Impact:** Strong 2024-2026 papers on lecture / video summarization (e.g., Video-LLama series, VideoChat, VISTA, TimeChat, LLaVA-Video) need to be cited and positioned from Week 6 onward. Writing related work cold in Week 21 is the #1 reason papers get rejected for "missing baseline X."

**Fix:** Add a Week-6 deliverable: *"Related-work bibliography first draft (50 papers), categorized by: lecture summarization, long-video LLMs, video QA, multimodal RAG, chaptering."* Add a Week-12 deliverable: *"Related-work bibliography final draft (100+ papers), with positioning paragraph."*

---

### T3.6 — No contingency for both VISTA and TIB failing

**Evidence:** `01-dataset-manifest.md` L147-151: fallback order ends at "VT-SSum extractive only; do not claim abstractive lecture summarization."

**Impact:** This is a hard cliff. If both VISTA and TIB fail, the project loses its RQ2 (the core research task per the README L18). Six months of work reduced to RQ1 + RQ3 (which depend on the same data) is a thin paper.

**Fix:** Add a third-tier fallback now: *"If both VISTA and TIB fail, narrow to (a) RQ1 only on YTSeg, (b) RQ3 only on EduVidQA + custom evidence, (c) submit as a **short paper** rather than a **full paper** with the contribution narrowed to multimodal chaptering and evidence-grounded QA. No abstractive summarization claim."* Make this an explicit pre-registered scope-cut.

---

## Risk-Register Cross-Check

The risk register is honest. These are the entries that I think are under-stated:

| Risk | Current treatment | Under-statement |
|------|-------------------|-----------------|
| R1 (VISTA fails) | Make TIB primary | Doesn't address the 1.93 TB / 2 TB ASR cost if TIB also fails (T3.6) |
| R5 (unequal budget) | "Enforce or report scaling curve" | The escape hatch is too easy (T2.2) |
| R6 (too few samples) | Narrow claims | Doesn't address power analysis (T1.4) |
| R8 (test leakage) | Freeze hashes | No enforcement mechanism in the runner |
| R12 (baseline changes) | Keep frozen baseline | Doesn't address what to do if a much-better compact VLM appears mid-project (publish a 6-month-out-of-date result, or restart?) |
| R13 (licensing blocks release) | "Release IDs only" | Pending Decision 4 is unresolved; the project may not be releasable at all |
| R14 (API drift) | Cache outputs, pin snapshot | Doesn't mention a fixed API evaluation **date** — if the API is evaluated in Week 14 and reported in Week 25, the API has likely changed |

---

## Internal Consistency Issues

Three direct contradictions between the plan and the most recent research reports:

1. **VISTA transcripts** — `01-dataset-manifest.md` L36 says transcripts exist; `260830-1917-dataset-vlm-research.md` L9 says they do not.
2. **E4 checkpoint** — `02-benchmark-matrix.md` L113 and `00-master-plan.md` L77 say "current compact Qwen3-VL"; `260830-1917-dataset-vlm-research.md` L102 says "Qwen3-VL-4B-Instruct FP16 (frozen)."
3. **TIB multimodal source** — `01-dataset-manifest.md` L52 implies decoding video for keyframes; `260830-1917-tib-audio-visual-probe.md` L81 recommends using `tib-bench.slides` directly without video decode.

These should be reconciled by a single Week-1 decision document.

---

## Recommendations (in execution order)

**Before Week 1 ends:**
1. Resolve the 4 Pending Decisions in writing.
2. Reconcile the 3 plan-vs-report contradictions above.
3. Submit VISTA access form; record approval date as a tracked risk.
4. Submit human-eval protocol to ethics review.
5. Freeze E4 checkpoint and OCR/visual embedder revisions.

**Before Week 6 ends (M2):**
6. Add Holm-Bonferroni + Cohen's d to `04-rq-mapping.md`.
7. Run a 10-video human-eval pilot; back-solve the required n.
8. Build the related-work bibliography (50 papers).
9. Update `01-dataset-manifest.md` with the TIB `tib-bench` 80-test subset and VISTA-no-transcript reality.
10. Add the third-tier fallback to `01-dataset-manifest.md` (T3.6).

**Before Week 12 ends (M3):**
11. Freeze the C5 architecture in `02-benchmark-matrix.md`.
12. Add a code/data release plan with PII handling.
13. Promote the "Pareto C5 vs C7 is structurally asymmetric" caveat to the paper outline.

**Through M3-M6:**
14. Re-evaluate Tier 3 holes by Week 18; some will be paper-level decisions.

---

## Unresolved Questions (for the supervisor)

1. Is the primary output a graduation thesis, a conference paper, or both? (Pending Decision 1)
2. Is Vietnamese evaluation mandatory, optional, or out-of-scope? (Pending Decision 2)
3. Is a second human annotator committed for the full 26 weeks? (Pending Decision 3)
4. May the project release only IDs/features, or also source media after legal review? (Pending Decision 4)
5. Is the supervisor willing to co-author the paper and review the related-work draft in Week 6, not Week 21?
6. Is the T4 budget fixed at Colab free, or can the project use Colab Pro / institutional GPU for a 2-week crunch if needed?

---

## Closing Note

The plan is a serious, defensible research design. The holes above are the difference between "a paper that gets accepted with one revision" and "a paper that gets rejected on methodology." None of them are show-stoppers individually, and most close in a single paragraph or one new checklist item. The pattern to watch for is **ambition vs. rigor**: every week the plan tries to do more (VLM baseline, human eval, multiple datasets, custom evidence subset, paper + thesis) without adding either a buffer week or a forced scope-cut. The two-track timeline (evidence + product, per the user-memory plan under review) is the right framing — make sure the evidence track is not crowded out by product work after Week 8.

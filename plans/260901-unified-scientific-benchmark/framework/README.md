# Plan: Multimodal Lecture Understanding, Summarization & Retrieval

**Date:** 2026-08-30  
**Owner:** Tran Phi Hung  
**Status:** Draft v5 — critical gaps patched (2026-08-31)  
**Compute target:** T4 16GB-class GPU + optional API  
**Timeline:** 26 weeks

## Core thesis

Build one temporally structured multimodal representation and test whether it improves:

1. semantic chaptering;
2. hierarchical lecture summarization;
3. evidence-grounded retrieval/QA;
4. quality–latency–VRAM trade-offs.

Summarization is a core research task, not an optional product feature.

## Authoritative files

| File | Purpose |
|------|---------|
| [`00-master-plan.md`](./00-master-plan.md) | Thesis framing, architecture, contributions, scope |
| [`01-dataset-manifest.md`](./01-dataset-manifest.md) | Dataset qualification and acceptance gates |
| [`02-benchmark-matrix.md`](./02-benchmark-matrix.md) | Controlled experiments and metrics |
| [`03-colab-runbook.md`](./03-colab-runbook.md) | T4 execution and reproducibility |
| [`04-rq-mapping.md`](./04-rq-mapping.md) | Four falsifiable research questions |
| [`05-6month-timeline.md`](./05-6month-timeline.md) | 26-week execution plan |
| [`06-risks-mitigations.md`](./06-risks-mitigations.md) | Stop/go risks and scope cuts |
| [`decisions-log.md`](./decisions-log.md) | Frozen strategic decisions (updated as answers arrive) |

## Dataset strategy

| Tier | Dataset | Core use |
|------|---------|----------|
| A | **YTSeg** | Chaptering and temporal representation |
| B | **VISTA** | Primary abstractive scientific-video summarization (form-gated; must submit access request Week 1) |
| C | **TIB** | External long multimodal lecture/presentation summarization; primary if VISTA fails |
| D | **EduVidQA** | External lecture QA |
| E | **VT-SSum** | Auxiliary extractive transcript summarization/segmentation |
| F | Small custom evidence set | OCR/keyframe evidence localization only |

No single dataset needs every annotation. The same representation and controlled interfaces must transfer across datasets.

## Four research questions

- **RQ1 — Representation/chaptering:** Do transcript, acoustic, visual and OCR cues improve time-based chaptering over transcript-only?
- **RQ2 — Summarization:** Does predicted chapter/scene structure improve summary coverage and factual consistency over flat and fixed-chunk baselines?
- **RQ3 — Retrieval/QA:** Does the same hierarchy improve evidence localization and grounded QA?
- **RQ4 — Efficiency:** What quality–latency–VRAM trade-off is achieved against a current compact VLM under equal local budgets?

## Two-week dataset gate

Do not start full experiments until:

- official licenses/splits/schemas are recorded for YTSeg, VISTA, TIB and EduVidQA;
- 20 candidate media items are probed and download success is measured;
- at least 10 items run end-to-end;
- 100 VISTA/TIB references are audited for source support and summary suitability;
- chaptering metrics match `chunkseg`;
- one current compact VLM runs on the target T4 budget;
- real storage, GPU-hour, latency and API cost are measured;
- VISTA access form is submitted and approval date is logged;
- ethics-review protocol for human evaluation is submitted.

If VISTA media/license fails, TIB becomes primary summarization data. If both fail, see the three-tier fallback below.

## Three-tier summarization fallback

1. **VISTA primary + TIB external** — nominal path.
2. **TIB primary** — if VISTA fails media, license or schema gate; document exclusion; narrow abstracts to scientific presentations.
3. **No abstractive summarization** — if both VISTA and TIB fail; narrow to (a) RQ1 only on YTSeg, (b) RQ3 only on EduVidQA + custom evidence; submit as a **short paper** with contributions narrowed to multimodal chaptering and evidence-grounded QA. Do not claim abstractive lecture summarization.

## Frozen decisions

The following are locked before any full experiment run. See `decisions-log.md` for the full record.

- Summarization remains core.
- No independent ASR/OCR/scene/captioning leaderboards.
- No chapter boundary used as shot-boundary ground truth.
- No TVSum importance score used as scene-boundary ground truth.
- No eight-video paper claim.
- No exact expected result before measurement.
- Local baselines use the same hardware, frame, source-token and output-token budgets.
- Video is the statistical unit; nested chapters/questions are not independent samples.
- Learned variants use three seeds and video-level paired bootstrap confidence intervals.
- **E4 baseline = `Qwen3-VL-4B-Instruct` FP16** (pin exact HF commit hash in Week 1 after 10-video VRAM pilot). Qwen3-VL-8B-AWQ is an optional quality row only and must not be mixed into the main Pareto table.
- **C7 (RQ1 compact VLM) and E4 (RQ4 efficiency) are the same `Qwen3-VL-4B-Instruct` FP16 checkpoint.** Cached features are shared; do not re-run.
- **VISTA does not ship transcripts.** Text-only VISTA fallback requires self-ASR on raw video. This is out of scope unless VISTA media is fully downloaded and an ASR step is budgeted. See `01-dataset-manifest.md`.
- **OCR model = PaddleOCR v3 (`ch_PP-OCRv4`) confidence threshold 0.6.**
- **Visual embedding = DINOv2 ViT-S/14.**
- **Keyframe sampling = TransNetV2 scene-boundary detector; 1 fps fallback.**
- **C5 architecture = 4-layer cross-attention transformer, 256-dim hidden, text/visual/OCR projected to shared 256-dim space, 3 learned boundary tokens, binary cross-entropy on boundary positions.**
- **C6 architecture = same as C5 but with concatenation-only fusion (no cross-attention); isolates fusion-mechanism contribution.**
- **Multiple-comparison correction = Holm-Bonferroni within each RQ family (RQ1: 4 deltas, RQ2: 4 S-pairs, RQ3: 3 Q-pairs) at α = 0.05. Report both raw and corrected CIs. Effect size = Cohen's d with Hedges' g when n < 20.**
- **Human eval sample: determined by power analysis pilot (Week 13); floor is 50 videos but final n is back-solved from effect size; scope-cut list updated accordingly.**
- **TIB evaluation target = `gigant/tib-bench` test subset (80 records, zero leakage confirmed by probe 2026-08-30); multimodal evidence = `slides` column (PIL PNG 512×288).**
- Equal-token-budget enforcement is strict: if S3/S4 cannot fit the same source/output budget as S1, the run is a failure — not a separate scaling curve. See `06-risks-mitigations.md`.

## Strategic decisions (resolved 2026-08-31)

Recorded in `decisions-log.md`:

1. **Primary deliverable (D-S01):** Khóa luận tốt nghiệp kết hợp bài báo hội nghị (Thesis + Paper track).
2. **Vietnamese evaluation (D-S02):** Tiếng Anh là chính; tiếng Việt là Future Work / scope-cut #1.
3. **Annotator & Evaluation protocol (D-S03):** Không có annotator thứ 2; sử dụng 100% pre-labeled ground truth datasets + single author pilot audit + LLM-as-a-Judge.
4. **Data release (D-S04):** Chỉ phát hành IDs, manifests, precomputed features và reproduction scripts (không redistribute raw media).

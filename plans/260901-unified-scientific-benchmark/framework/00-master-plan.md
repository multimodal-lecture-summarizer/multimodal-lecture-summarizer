# Master Plan: Multimodal Lecture Understanding, Summarization & Retrieval

**Last updated:** 2026-08-31 — contributions reframed as falsifiable claims; C5/C6 architectures frozen; C7/E4 identity clarified; VISTA transcript issue resolved.

## 1. Research objective

**Vietnamese title:** Nghiên cứu và xây dựng hệ thống hiểu, tóm tắt và khai thác video bài giảng đa phương thức  
**English title:** Multimodal Lecture Video Understanding, Summarization and Knowledge Retrieval

**Thesis statement:**

> Can a compute-efficient, temporally structured multimodal representation improve lecture chaptering, hierarchical summarization and evidence-grounded QA over transcript-only baselines under controlled compute and context budgets?

The project builds one representation, not three unrelated pipelines.

## 2. Architecture

```text
Lecture video
   │
   ├── transcript + acoustic cues
   ├── sampled frames (DINOv2 ViT-S/14) + visual embeddings
   └── OCR text (PaddleOCR v3 ch_PP-OCRv4, conf >= 0.6)
          │
          ▼
Temporal multimodal encoder
  [C5: 4-layer cross-attention transformer, 256-dim hidden,
   text/visual/OCR projected to shared 256-dim space,
   3 learned boundary tokens, binary cross-entropy loss]
          │
          ├── boundary head ───────────────► semantic chapters
          │
          ▼
Video → chapter → scene → evidence hierarchy
          │
          ├──► hierarchical summarization + evidence citations
          └──► retrieval/QA + evidence timestamps
                         │
                         ▼
       controlled quality / latency / VRAM evaluation
```

**Compact VLM baseline (C7/E4):** `Qwen3-VL-4B-Instruct` FP16 — same checkpoint for RQ1 (chaptering) and RQ4 (efficiency). HF commit hash pinned Week 1. _(See `decisions-log.md` D-T01, D-T03.)_

## 3. Scientific contributions (falsifiable)

| ID | Claim | Evidence if positive | Evidence if negative |
|----|-------|----------------------|----------------------|
| C1 | Temporally structured multimodal representation combining transcript, acoustic, visual and OCR cues improves chaptering over text-only | C5 > C1 with positive 95% CI and Cohen's d after Holm correction | Report effect size and error mode; do not change metric |
| C2 | Hierarchical lecture summarization driven by predicted structure improves human factual coverage over equal-budget fixed chunking | S3 > S1 on primary human metrics with positive CI | Report hierarchy overhead and error propagation; do not switch to ROUGE |
| C3 | Shared evidence hierarchy improves grounded lecture QA retrieval over flat RAG | Q2/Q3 > Q0 on evidence and QA metrics | Report oracle gap and failure analysis |
| C4 | Resource-aware pipeline lies on a better local Pareto frontier than transcript-only and compact VLM | E3 dominates on ≥ 2 core metrics at equal budget | Report Pareto position; note C5-vs-C7 structural asymmetry |

All claims are open hypotheses. Negative results are valid results. No post-hoc metric or model selection.

## 4. Research questions

### RQ1 — Multimodal chaptering

Do multimodal cues improve time-based lecture chaptering over transcript-only?

- Primary: collar F1 at ±3 seconds.
- Secondary: ±5/10 seconds, Pk, WindowDiff.
- Data: YTSeg official split plus frozen downloadable multimodal subset.
- Statistics: paired bootstrap CI; Holm-Bonferroni within RQ1 family (4 deltas); Cohen's d.

### RQ2 — Hierarchical summarization

Does predicted chapter/scene structure improve summary coverage and factual consistency over flat and fixed-chunk baselines?

- Primary data: VISTA (no transcript shipped; raw video + self-ASR required; form-gated).
- External validation: `gigant/tib-bench` test subset (80 records, zero leakage).
- Auxiliary extractive diagnostic: VT-SSum.
- Primary evidence: blinded human factuality/coverage ratings and source-grounded QA.
- ROUGE/BERTScore remain secondary.
- Statistics: Holm-Bonferroni within RQ2 family (4 S-pairs); Cohen's d.

### RQ3 — Evidence retrieval/QA

Does the same hierarchy improve evidence retrieval and grounded QA over flat transcript RAG?

- External data: EduVidQA.
- Custom data: small evidence-first subset only for OCR/keyframe/time localization.
- Metrics: Recall@K, MRR, evidence hit/IoU, answer correctness, faithfulness.
- Statistics: Holm-Bonferroni within RQ3 family (3 Q-pairs); Cohen's d.

### RQ4 — Efficiency

What quality–latency–VRAM trade-off does the proposed pipeline achieve against transcript-only and a current compact VLM?

- Local track: same T4, frames, source tokens and output tokens.
- API track: separate, with measured request usage/cost; cloud VRAM is N/A.
- Caveat: E3 (modular pipeline) vs E4 (end-to-end VLM) is structurally asymmetric for raw accuracy; Pareto is valid for resource analysis only.

## 5. Why summarization is core

The original product promise is lecture summarization. Removing it would make the research misaligned with the system.

RQ2 remains scientifically defensible by:

1. using public scientific-video/presentation summary datasets rather than eight self-generated silver references;
2. comparing structure while fixing the summarization model and total token budget;
3. separating reference creation, system generation and judging;
4. combining automatic, QA-based and blinded human evaluation;
5. validating on more than one summarization dataset.

## 6. Controlled experiment principle

Only one factor changes per comparison:

- RQ1: modality/fusion changes; boundary head and tuning budget stay fixed.
- RQ2: structuring method changes; LLM, prompt, total input/output budget stay fixed (strictly enforced; see `decisions-log.md` D-T08).
- RQ3: retrieval structure/evidence changes; embedder, generator, top-k and context budget stay fixed.
- RQ4: system changes; test items and resource budgets stay fixed within each track.

## 7. Dataset qualification

A dataset enters the benchmark only after:

- official source, version, split and license are recorded;
- schema contains the required input and target;
- media availability and failure rate are measured;
- reference quality is audited on a predeclared sample (rubric in `01-dataset-manifest.md` §5);
- leakage/duplicate checks pass;
- unavailable items follow a frozen policy;
- statistical unit and clustering are defined.

See [`01-dataset-manifest.md`](./01-dataset-manifest.md) and [`decisions-log.md`](./decisions-log.md).

## 8. Scope

### In scope

- Frozen feature extraction: ASR/acoustic, visual (DINOv2 ViT-S/14), OCR (PaddleOCR v3).
- Learned temporal representation and chaptering (C5/C6 architectures frozen; see `decisions-log.md` D-T02).
- Hierarchical abstractive summarization with evidence citations.
- Evidence-grounded retrieval/QA.
- Same-hardware and measured API evaluation.
- Reproducible manifests, raw predictions, statistics and failure reporting.

### Out of scope

- Training a large Video-LLM from scratch.
- Independent component leaderboards.
- Product UI redesign, billing, multi-tenant deployment.
- Building a large new QA or summarization dataset.
- Treating optional Vietnamese data as a blocker for the English core study (see `decisions-log.md` D-S02).
- Post-hoc metric/model selection after seeing test results.
- Text-only VISTA fallback without self-ASR pipeline (VISTA has no transcript field; see `decisions-log.md` D-T05).

## 9. Milestones

| Milestone | Weeks | Exit |
|-----------|-------|------|
| M1 Dataset/compute pilot | 1–2 | Dataset gates pass; T4 feasibility confirmed; E4 commit hash pinned; VISTA access submitted; IRB protocol submitted |
| M2 Frozen data + runner | 3–6 | Versioned manifests, caches, metrics, stats; related-work v1 (50 papers); ethics approval received |
| M3 RQ1 representation | 7–12 | Chaptering ablations × 3 seeds with corrected CIs; related-work v2 (100+ papers) |
| M4 RQ2 summarization | 13–16 | VISTA primary + TIB external results; human eval with IAA |
| M5 RQ3/RQ4 | 17–20 | QA and efficiency tables; reproducibility package skeleton |
| M6 Writing/submission | 21–26 | Paper/thesis + reproducibility package |

## 10. Success criteria

- Dataset gates pass and missingness is reported.
- Four Pending Decisions (D-S01–D-S04) answered before full runs.
- RQ hypotheses/metrics are frozen before full runs.
- All core comparisons use equal task budgets (strictly enforced).
- Per-video predictions and failures are retained.
- Effect sizes (Cohen's d), bootstrap 95% CIs, and Holm-corrected p-values are reported for all primary comparisons.
- Negative results remain valid results; no post-hoc narrative replacement.
- IRB approval received before human annotation begins.
- Related-work bibliography completed by Week 12.

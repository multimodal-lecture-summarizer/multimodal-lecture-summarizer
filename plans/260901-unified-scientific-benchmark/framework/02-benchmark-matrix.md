# Controlled Benchmark Matrix

No full matrix runs before the two-week dataset/compute gate.

## 1. Shared controls

| Control | Rule |
|---------|------|
| Data | Official/frozen splits and item IDs |
| Local hardware | Same T4-class GPU within a comparison |
| Visual budget | Same sampled frames/minute and image resolution |
| Text budget | Same total source tokens and output tokens |
| Generation | Same LLM, prompt, decoding and revision within RQ2/RQ3 |
| Retrieval | Same embedder, top-k and context budget within RQ3 |
| Learned models | Three seeds; same search/early-stop budget |
| Statistics | Video-level aggregation and paired bootstrap 95% CI |
| Failures | Retained in denominator and reported |

## 2. RQ1 — Representation and chaptering

| ID | Variant | Modalities | Purpose |
|----|---------|------------|---------|
| C0 | Periodic/random boundary | Duration only | Sanity lower bound |
| C1 | Text-only temporal model | Transcript | Primary baseline |
| C2 | Text + acoustic | Transcript, pauses/audio embedding | Acoustic value ablation |
| C3 | Text + visual | Transcript, sampled-frame embedding (DINOv2 ViT-S/14) | Visual value ablation |
| C4 | Text + OCR | Transcript, OCR (PaddleOCR v3 ch_PP-OCRv4, conf ≥ 0.6) | On-screen text value ablation |
| C5 | Full learned fusion | All modalities | **Proposed representation** — 4-layer cross-attention transformer, 256-dim hidden, text/visual/OCR projected to shared 256-dim space, 3 learned boundary tokens, binary cross-entropy loss |
| C6 | Full late fusion | All modalities | Fusion-mechanism ablation — same as C5 but concatenation-only, no cross-attention |
| C7 | Current compact VLM | Frames + transcript prompt | End-to-end baseline — **same checkpoint as E4** (`Qwen3-VL-4B-Instruct` FP16, HF commit pinned Week 1) |

**Note:** C7 and E4 are identical checkpoints. Feature caches are shared; do not re-run. _(See `decisions-log.md` D-T01, D-T03.)_

**Keyframe sampling:** TransNetV2 scene-boundary detector; 1 fps fallback. _(See `decisions-log.md` D-T04.)_

**Training objective:** supervised boundary loss; optional contrastive alignment is auxiliary and must be ablated.

**Primary metrics:**

- collar precision/recall/F1 at ±3 seconds;
- Pk and WindowDiff;
- sensitivity at ±5/10 seconds;
- per-video failure rate and confidence interval.

**Statistics:** paired bootstrap by video, 1,000 resamples, 95% CI; Cohen's d effect size (Hedges' g when n < 20); Holm-Bonferroni within RQ1 family (4 deltas). _(See `04-rq-mapping.md` RQ1.)_

## 3. RQ2 — Hierarchical summarization

The summarization model stays fixed. Only structuring/evidence input changes.

| ID | Method | Structure | Evidence |
|----|--------|-----------|----------|
| S0 | Flat/truncated transcript | None | Transcript |
| S1 | Fixed-chunk map-reduce | Equal token chunks | Transcript |
| S2 | Oracle hierarchy diagnostic | Reference chapters where available (VISTA: N/A; TIB: confirm Week 14) | Transcript |
| S3 | Predicted hierarchy | C5 chapters/scenes | Transcript |
| S4 | Multimodal predicted hierarchy | C5 chapters/scenes | Transcript + OCR (PaddleOCR v3) + keyframes (DINOv2 ViT-S/14) |

**Datasets:**

- VISTA: primary abstractive evaluation. _(No transcript shipped; raw video required; see `01-dataset-manifest.md` §2 and `decisions-log.md` D-T05.)_
- TIB: external validation using `gigant/tib-bench` test subset (80 records, zero leakage). _(See `decisions-log.md` D-T06.)_
- VT-SSum: extractive auxiliary diagnostic.

**Fairness constraints:**

- Same generator/model revision.
- Same maximum total source tokens across S0–S4 (strictly enforced; see `decisions-log.md` D-T08).
- Same maximum output length.
- Same prompt information except the tested structure/evidence.
- Same number of generation attempts and deterministic settings where supported.
- S2 is diagnostic; only used where reference chapters exist; not mixed into primary VISTA table.

**Token budget (1 h lecture, 32K source-token cap):**

| Component | Tokens |
|-----------|--------|
| Transcript (~7–10k words) | ~10–16K |
| Frames: 150 scene keyframes @ 128–200 tok (448 px short edge) | ~13–20K |
| OCR text (slide text) | ~2–5K |
| **Total target** | **≤ 30–32K** |

Sample ~1 frame per 20–40 s; cap 150–200 frames; resize short edge 448–512 px. Lock `frames`, `source_tokens`, `output_tokens` (512) per run record and keep equal across S0–S4. _(See `decisions-log.md` D-T08.)_

**Automatic metrics:**

- ROUGE-1/2/L and BERTScore — secondary overlap metrics.
- QA-based key-point coverage.
- Unsupported-claim rate against source evidence.
- Citation/evidence precision when timestamps are generated.
- Length, compression ratio, latency and token usage.

**Human metrics:**

- Factual support, key-point coverage, coherence and concision on 0–5 scales (weighted Cohen's κ for ordinal dimensions).
- Pairwise preference: S1 vs S3 and S3 vs S4.
- Final n from Week-13 power analysis pilot (floor 50 videos); two blinded raters; agreement reported. _(See `decisions-log.md` D-T10.)_

**Statistics:** Holm-Bonferroni within RQ2 family (4 S-pairs); Cohen's d with bootstrap CI. _(See `04-rq-mapping.md` RQ2.)_

## 4. RQ3 — Evidence retrieval and QA

| ID | System | Retrieval structure | Evidence |
|----|--------|---------------------|----------|
| Q0 | Flat transcript RAG | Fixed token chunks | Transcript |
| Q1 | Oracle-chapter RAG | Reference chapter | Transcript |
| Q2 | Predicted-chapter RAG | C5 chapter → scene | Transcript |
| Q3 | Multimodal structured RAG | C5 chapter → scene | Transcript + OCR (PaddleOCR v3) + keyframe (DINOv2 ViT-S/14) |

**Datasets:** EduVidQA official splits plus frozen custom visual-evidence subset (Tier F).

**Metrics:**

- Recall@1/5 and MRR;
- evidence timestamp hit/IoU;
- answer correctness/entailment;
- faithfulness to retrieved evidence;
- latency, context tokens and measured cost;
- macro-by-video and question-type breakdown.

Q1 vs Q2 measures chaptering error propagation. Q2 vs Q3 isolates multimodal evidence value.

**Statistics:** Holm-Bonferroni within RQ3 family (3 Q-pairs); Cohen's d with bootstrap CI. _(See `04-rq-mapping.md` RQ3.)_

## 5. RQ4 — Efficiency

### Local track

| ID | System | Hardware |
|----|--------|----------|
| E1 | C1 + S1 + Q0 transcript-only | Same T4 |
| E2 | C5 + S3 + Q2 structured | Same T4 |
| E3 | C5 + S4 + Q3 multimodal structured | Same T4 |
| E4 | `Qwen3-VL-4B-Instruct` FP16 (HF commit pinned Week 1) | Same T4 |

**Note:** E4 = C7 (same checkpoint, shared cache). Optional quality row: Qwen3-VL-8B-AWQ if pilot confirms VRAM fits. Do not merge with primary Pareto table. _(See `decisions-log.md` D-T01, D-T03.)_

**Structural asymmetry caveat:** E3 (modular pipeline) vs E4 (end-to-end VLM) is not a fair accuracy comparison. The Pareto table is valid for resource trade-off analysis only; include an explicit caveat paragraph in the paper.

Report task quality, wall time, peak allocated/reserved VRAM, throughput, input/output tokens, failures and storage.

### API track

One strong API VLM on the same frozen items. Report measured usage, USD, wall time and API model snapshot. Run on a single fixed date (record in `decisions-log.md` D-T13). Do not merge local and API rows into a same-hardware claim.

## 6. Result table shells

### Chaptering

| Variant | F1 ±3s | Pk ↓ | WindowDiff ↓ | Cohen's d | 95% CI (corrected) | Failure % |
|---------|--------|------|--------------|-----------|---------------------|-----------|
| C0–C7 | TBD | TBD | TBD | TBD | TBD | TBD |

### Summarization

| Method | Human factuality ↑ | Human coverage ↑ | QA coverage ↑ | Unsupported claims ↓ | Cohen's d | 95% CI (corrected) | ROUGE-L | Tokens |
|--------|---------------------|------------------|---------------|----------------------|-----------|---------------------|---------|--------|
| S0–S4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed |

### Retrieval/QA

| System | R@5 | MRR | Evidence hit/IoU | QA correctness | Faithfulness | Cohen's d | 95% CI (corrected) | Tokens |
|--------|-----|-----|------------------|----------------|--------------|-----------|---------------------|--------|
| Q0–Q3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | fixed |

### Efficiency

| System | Track | Task quality | Wall time | Peak VRAM | Throughput | Cost |
|--------|-------|--------------|-----------|-----------|------------|------|
| E1–E4 | Local/API | TBD | TBD | TBD/N/A | TBD | measured |

## 7. Reproducibility record

Every run stores:

```json
{
  "run_id": "...",
  "rq": "RQ2",
  "variant": "S3",
  "dataset": "VISTA",
  "dataset_revision": "...",
  "item_id": "...",
  "seed": 42,
  "model_revision": "...",
  "prompt_revision": "...",
  "budget": {
    "frames": 150,
    "source_tokens": 32000,
    "output_tokens": 512,
    "frame_resolution_px": 448
  },
  "metrics": {},
  "latency_sec": 0,
  "peak_vram_gb": 0,
  "context_length": 0,
  "status": "ok",
  "git_commit": "..."
}
```

No unmeasured expected values are allowed in result tables.

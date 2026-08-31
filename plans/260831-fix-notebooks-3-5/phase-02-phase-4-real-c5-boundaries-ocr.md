---
phase: 2
title: "Phase 4: Real C5 Boundaries & OCR"
status: pending
priority: P1
dependencies: [phase-01]
---

# Phase 2: Phase 4 Notebook — Real C5 Boundaries & Real OCR/References

## Overview

The Phase 4 notebook loads real VT-SSum test JSONs (good) but then **fabricates** `c5_predicted_boundaries` (a cumulative-sum heuristic loop, not the real C5 model) and `ocr_slides` (a `"Slide {i}: {title} - Key Slide Concepts"` template). Replace with genuine values: real C5 boundary predictions from Phase 1's real-data model, and real OCR evidence — or honestly downgrade the "S4 Multimodal" comparison when real OCR is unavailable.

## Requirements

- Functional: `c5_predicted_boundaries` for each test lecture come from the **real C5 checkpoint** (Phase 1 `checkpoints/c5_real.pt`) applied to the SAME lectures' real cached features; OCR evidence is real or the S4 multimodal arm is explicitly documented as pending; **the template-fake OCR in `run_rq2_benchmark.py` is also removed**; stale fake outputs cleared.
- Non-functional: reproducible; no network download at notebook run time (OCR must already be cached); consistency with Phase 1 model & data split.

## Architecture

Current flow (fabricated boundaries + fake OCR):
```
load VT-SSum test JSONs (real transcript + real reference labels)
ocr_slides = [f"Slide {i+1}: {title} - Key Slide Concepts" ...]        # FAKE
c5_boundaries: cumulative-sum loop over slide sentence counts          # FAKE, not C5
summarizers S0/S1/S3/S4 run with fake boundaries + fake OCR
```

Target flow (honest):
```
real data path: load cached features for the same test lectures (Phase 1 test split)
c5_boundaries = real C5 model (loaded weights) forward() -> predicted_boundaries  # REAL
reference = VT-SSum ground-truth extractive labels  (already real)
OCR: read real ocr_features/transcript from cached .pt for that lecture
   - if a real OCR text stream exists for the lecture -> use it
   - else -> run S4 comparison ONLY if real OCR exists; otherwise drop S4 / mark "Real OCR pending"
```

## Related Code Files
- Modify: `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb`
  - Cell ~4 (dataset build) — replace the fake `ocr_slides` + fake `c5_boundaries` block.
  - Cell ~8 (eval loop) — pass real boundaries & real OCR.
- Modify: `benchmarks/scripts/run_rq2_benchmark.py`
  - Line ~100 `ocr_texts = [f"Slide Concept {i+1}..."...]` — replace with real OCR from cached `.pt` (or drop/absence-document).
- Reference (do not modify): `benchmarks/models/chaptering.py` (C5), `benchmarks/models/summarization.py` (S0–S4), `benchmarks/data/cached_features/*.pt`, `checkpoints/c5_real.pt`.
- Verify: whether any cached `.pt` carries slide-text/OCR to feed S4. If not, S4 must report the gap explicitly.

## Implementation Steps
1. Load the Phase 1 C5 checkpoint (`checkpoints/c5_real.pt`) into a freshly-built `C5_TemporalCrossAttentionTransformer` (same config as Phase 1: `d_ac=32`, d_model=256, etc.).
2. Map each test lecture to its cached `.pt` and get real `predicted_boundaries` from the C5 forward pass.
3. Replace `ocr_slides` in the notebook with real OCR evidence: inspect `ocr_features`/`transcript_sentences` in the cache; if real OCR text exists for the lecture, assemble it; else document `ocr_available=False`.
4. In the eval loop: pass `c5_predicted_boundaries` (real) and `ocr_slides` (real or empty). When `ocr_available=False`, **exclude S4 from the statistical family** (`S4 - S3` comparison) and add an explicit markdown/OCR-pending note — no fabricated multimodal claim.
5. In `run_rq2_benchmark.py`: replace the `"Slide Concept {i+1}..."` template OCR with real cached OCR (or, if none, drop S4's OCR contribution and note the gap) so the script matches the notebook.
6. Clear stale outputs (`outputs: []`) on notebook result cells.
7. Update markdown, dropping the "100% coverage / +61.89%" claims that came from the old fake run.

## Success Criteria
- [ ] `c5_predicted_boundaries` derived from the real C5 checkpoint, not the cumulative-sum heuristic.
- [ ] OCR is real cached evidence, or S4 is explicitly marked "OCR pending" — no template-fake OCR in notebook or `run_rq2_benchmark.py`.
- [ ] Stale RQ2 results cleared; metrics/statistics machinery intact.
- [ ] Notebook + `run_rq2_benchmark.py` both valid / no fabrication remaining.
- [ ] No fabricated "significant" conclusion remains.

## Risk Assessment
- **Risk: lecture-id mismatch between VT-SSum JSON id and cached `.pt` lecture_id** (VT-SSum uses hash ids; cache uses readable names). Mitigation: derive a matching by title substring; if no match, drop that lecture from Phase 4's eval (keep count honest) rather than fabricate.
- **Risk: no real OCR/slide-text stream in the cache** → S4 genuinely cannot run with real OCR. Mitigation: keep S4's other (text) evidence honest; apply the OCR-gap note and drop S4 from the RQ2 statistical family; do not use template OCR.
- **Risk: C5 weights not persisted / checkpoint missing** → Phase 4/5 cannot get real predicted boundaries. Mitigation: Phase 1 writes `checkpoints/c5_real.pt`; Phase 2 raises a clear error if missing (never fabricate boundaries).

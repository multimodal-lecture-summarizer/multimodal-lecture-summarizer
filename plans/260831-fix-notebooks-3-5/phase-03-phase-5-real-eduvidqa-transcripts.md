---
phase: 3
title: "Phase 5: Real EduVidQA Transcripts"
status: pending
priority: P1
dependencies: [phase-01]
---

# Phase 3: Phase 5 Notebook — Real EduVidQA Transcripts via `q_and_a.json`

## Overview

The Phase 5 notebook currently reads real EduVidQA **questions/answers** but then **fabricates transcripts** that embed the answer verbatim:
```
sents = [ ..., f"Detailed scientific explanation confirms: {ans_text}", ... ]
```
This makes Recall/Answer-F1 meaningless. **Fix confirmed by validation:** reuse the real data path already proven by `run_rq3_benchmark.py` — load questions/answers from `experiments/datasets/eduviqa/q_and_a.json` and real transcripts from `cached_data["transcript_sentences"]` (matching `video_name` to `cached_features/*.pt`). No answer text may ever enter evidence; never fabricate transcripts. Also remove the boundary-noise + template-OCR fabrication from `run_rq3_benchmark.py` so notebook and script agree.

## Requirements

- Functional: answers are NEVER injected into transcripts/evidence; retrieval & QA run on real cached transcripts matched by `video_name`; real C5 boundaries (Phase 1 checkpoint) replace `b + np.random.uniform` noise; both the notebook and `run_rq3_benchmark.py` are de-fabricated; stale fake outputs cleared.
- Non-functional: no silent fabrication; reproducible; respects RQ3 budget parity (top-k=3, context ≤1024); no YouTube download needed.

## Architecture

Current flow (fabricated, notebook + script):
```
real_world_test.csv / q_and_a.json -> real question + real answer
sents = template + [f"...confirms: {ans_text}"]          # FAKE: answer leak (notebook)
pred_boundaries = b + np.random.uniform(-10,10)          # FAKE (script)
ocr_slides = [f"Slide: {sentences[...][:40]}"]           # FAKE template (script)
Q0..Q3 -> Recall/Answer-F1  (meaningless: answer leaked / fake boundaries)
```

Target flow (real, both notebook + script):
```
q_and_a.json -> real question, real reference answer, video_name (all 20 real lectures)
cached .pt matched by video_name -> real transcript_sentences, real timestamps,
                                    real ground_truth_boundaries, real OCR/visual features
predicted boundaries = Phase 1 C5 checkpoint forward -> real predicted_boundaries
Q0..Q3 use real transcripts + real C5 boundaries + real OCR (or text-only with note)
Answer F1 / recall computed vs the real reference answer only — evidence NEVER contains a copy of it
```

## Related Code Files
- Modify: `experiments/notebooks/05_phase5_evidence_retrieval_and_qa.ipynb`
  - Cell ~4 and ~15 (duplicated data-build blocks) — replace fabricated transcript/OCR/oracle blocks with the `q_and_a.json` + cached-transcript loader; deduplicate the two blocks into ONE source.
- Modify: `benchmarks/scripts/run_rq3_benchmark.py`
  - Line ~84 `pred_boundaries = [b + np.random.uniform(-10.0,10.0) for b in gt_boundaries]` → real Phase 1 C5 boundaries (load `checkpoints/c5_real.pt`).
  - Line ~99 `ocr_slides = [f"Slide: {sentences[...][:40]}"...]` → real cached OCR or drop OCR with a noted gap.
- Reference (do not modify): `benchmarks/models/retrieval_qa.py` (Q0–Q3, already real SBERT+LLM), `benchmarks/metrics/qa_metrics.py`, `checkpoints/c5_real.pt`.
- Possibly create (only if needed): a small shared loader cell/function mapping `video_name` → cached `.pt`; keep it minimal (YAGNI).

## Implementation Steps
1. Write one item-builder that, for each entry in `q_and_a.json`: uses `video_name` to locate the matching `cached_features/*.pt`; if no match, SKIP that lecture (count honest drops). Build `transcript_sentences` from `cached_data["transcript_sentences"]`, `timestamps` from `cached_data["timestamps"]`, `gt_boundaries` from `cached_data["ground_truth_boundaries"]`.
2. Load the Phase 1 C5 checkpoint; compute `predicted_boundaries` per matched lecture. If the checkpoint is missing, raise a clear error (never fabricate boundaries).
3. Build `oracle_chapters` from real timestamps/`gt_boundaries`; `ocr` from real cached OCR when present, else empty with `ocr_available=False`. **Never read `answer` into `sents`/`ocr`/oracle.**
4. Run Q0–Q3 (existing real SBERT+LLM pipelines). Evaluate with `compute_all_qa_metrics` using `ground_truth_answer` as the reference only. When `ocr_available=False`, gate Q3's OCR contribution and note the gap.
5. In `run_rq3_benchmark.py`: replace the boundary-noise line (84) with real C5 boundaries; replace the template OCR (99) with real cached OCR or a hard gap.
6. Deduplicate cell 15 vs cell 4 — keep a SINGLE item-builder used consistently; document whether the eval uses all 20 or a representative subset.
7. Clear stale fake outputs (`outputs: []`) on all result cells; remove fake qualitative answer prints.
8. Update markdown, removing the fake Answer-F1 numbers and stating the real `q_and_a.json` + cached-transcript contract.
9. Add an LLM/SBERT availability guard: before evaluating, assert SBERT `DenseEmbedder` loaded AND the real LLM engine is configured; otherwise raise a clear error (no silent hash-fallback reported as real).

## Success Criteria
- [ ] No `ans_text`/reference answer appears inside `sents`/`ocr`/evidence construction in notebook or `run_rq3_benchmark.py`.
- [ ] Transcripts come from `cached_data["transcript_sentences"]` matched by `video_name` (not fabricated).
- [ ] `predicted_boundaries` come from the real Phase 1 C5 checkpoint (no `np.random.uniform`).
- [ ] Single source of truth for the item-builder (no duplicated divergent block).
- [ ] Stale Q0–Q3 fake outputs cleared.
- [ ] LLM/SBERT guard present (no silent hash-fallback).
- [ ] Notebook + `run_rq3_benchmark.py` valid / de-fabricated.

## Risk Assessment
- **Risk: `video_name` in `q_and_a.json` does not exactly match a cached `.pt` stem** → lecture skipped. Mitigation: substring/normalized matching (+ store `video_name`→`lecture_id` mapping); keep an honest skip count; verify the 20 names in `q_and_a.json` align with the 20 `.pt` manifest entries.
- **Risk: OCR features are multimodal tensors, not readable text** → S4/Q3 "slide text" may not be directly consumable. Mitigation: use `ocr_features` only if the benchmark contract consumes them; otherwise run Q0–Q2 (text) and mark Q3 OCR as pending; never template-fake.
- **Risk: LLM engine not configured (no API key)** → answer synthesis cannot run. Mitigation: the availability guard in Step 9 fails loudly instead of reporting hash-fallback as real.

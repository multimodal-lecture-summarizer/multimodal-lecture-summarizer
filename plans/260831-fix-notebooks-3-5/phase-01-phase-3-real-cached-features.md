---
phase: 1
title: "Phase 3: Real Cached Features"
status: pending
priority: P1
dependencies: []
---

# Phase 1: Phase 3 Notebook — Load Real Cached Features + Rebuilt Supervision

## Overview

Replace the synthetic `generate_lecture_batch()` (which builds `text/vis/ocr/acoustic` from `torch.randn` and random boundaries) with the real cached `.pt` features via `benchmarks/data/dataset.py`. Because the cached per-timestep `targets` vector is broken (near-empty, misaligned with `ground_truth_boundaries`), Phase 1 **rebuilds the training targets with a hard guard** from the cache's own real `ground_truth_boundaries` (boundary timestamps). This makes RQ1 chaptering results meaningful.

Note: initial plan referenced VT-SSum segmentation as GT, but verification showed **0 overlap** between VT-SSum test videos and the cached lectures (different datasets). The cache's own `ground_truth_boundaries` are real and present, so they are the authoritative GT.

## Requirements

- Functional: notebook loads real 20-lecture cache; acoustic dim corrected 64→32; model configs match data dims; **supervision rebuilt from the cache's real `ground_truth_boundaries` with a guard (never train on broken cached `targets`)**; stale fake outputs removed; **trained C5 checkpoint persisted for Phases 2/3**.
- Non-functional: cells run only when real data exists (clear error otherwise); no network; reproducible via the existing split helper.

## Data-Health Finding (Drives Architecture)

Verification (spot-check of `cached_features/*.pt`) shows the cache supervision is internally inconsistent:

| file | `len(ground_truth_boundaries)` | `targets.sum()` | note |
|------|------|------|------|
| CA01 | 13 | 0 | no boundary marked in targets |
| CA05 | 27 | 2 | timestamps non-monotonic `[0,0,15.96,0,...]` |
| DS12-Queue | 44 | 2 | — |
| DS11-Recursion-Stack | 44 | 1 | — |

Manifest `num_boundaries` also disagrees with `len(ground_truth_boundaries)` (CA01: manifest 0, tensor 13). So the cached `targets` are **not** a reliable supervision signal. The authoritative GT is the cache's own `ground_truth_boundaries` (real boundary timestamps). (VT-SSum was considered but rejected: 0 overlap with the cached lectures — different videos.)

## Architecture

Current flow (fabricated):
```
generate_lecture_batch()  -> torch.randn text(384) vis(384) ocr(384) ac(64) + random boundary injection
C1..C6  <- train/val/test batches (synthetic)
eval + Holm-Bonferroni  -> fake "significant" numbers
```

Target flow (real, with rebuilt supervision):
```
LectureFeatureDataset(data_dir=benchmarks/data/cached_features)
create_lecture_splits(data_dir, train_ratio, val_ratio, seed) -> (train,val,test)  # FAR 42 split
collate_lecture_batches(list_of_dicts) -> ChapteringBatch  # real features
--- NEW DATA-HEALTH GUARD + TARGETS REBUILD ---
for each sample in (train,val,test):
    check targets vs ground_truth_boundaries consistency
    rebuild targets by binning the cache's own ground_truth_boundaries onto timestamps
    raise clear error if a lecture yields no valid boundary label
C1..C6 initialized with matching input dims  (d_ac=32, not 64)
eval metrics on test set + Holm-Bonferroni  (real, meaningful)
```

Verified data shapes (from `CA01.pt`):
- `text_features [T,384]`, `visual_features [T,384]`, `ocr_features [T,384]`, `acoustic_features [T,32]`, `targets [T]`, `timestamps [T]`, `ground_truth_boundaries`, `transcript_sentences`.
- Authoritative GT source: each lecture's own `ground_truth_boundaries` (real boundary timestamps in seconds).

## Related Code Files
- Modify: `experiments/notebooks/03_phase3_representation_and_chaptering.ipynb`
  - Cell ~6 (`generate_lecture_batch`) → replace with `create_lecture_splits` + `collate_lecture_batches` + the targets-rebuild/guard cell.
  - Cell ~8 (model init) → change `C2_AcousticChapterer(d_ac=64→32)` and `C5(..., d_ac=64→32)`.
  - Cell ~13 (eval) — currently uses `test_batch.timestamps`/`targets`; adapt to the real `ChapteringBatch` (already `[B,T]` padded) and use the rebuilt supervision for gold.
  - Cell ~18 (qualitative plot) — sample index & gold from real batch.
- Reference (do not modify): `benchmarks/data/dataset.py`, `benchmarks/models/chaptering.py`, `benchmarks/scripts/run_rq1_benchmark.py`.
- Delete: any stale text claiming DINOv2/OCR extraction for the random data.

## Implementation Steps
1. Read the notebook JSON (`Read` tool) and map each code cell index to its purpose (env, data, model-init, train, eval, stats, qualitative).
2. Replace `generate_lecture_batch` cell with a loader cell:
   ```python
   from benchmarks.data.dataset import LectureFeatureDataset, collate_lecture_batches, create_lecture_splits
   DATA_DIR = PROJECT_ROOT / "benchmarks" / "data" / "cached_features"
   train_ds, val_ds, test_ds = create_lecture_splits(data_dir=str(DATA_DIR), train_ratio=0.6, val_ratio=0.2, seed=42)
   train_batch = collate_lecture_batches([train_ds[i] for i in range(len(train_ds))])
   val_batch   = collate_lecture_batches([val_ds[i] for i in range(len(val_ds))])
   test_batch  = collate_lecture_batches([test_ds[i] for i in range(len(test_ds))])
   ```
3. Guard the loader so it raises a clear `FileNotFoundError` if `DATA_DIR` empty (do not silently fabricate).
4. **Add a data-health guard + targets-rebuild cell** (runs after collate, before training):
   - For each sample, assert consistency between `targets` and `ground_truth_boundaries`; if `targets.sum()` is suspiciously low (< ~1 boundary per ~3-5 sentences), treat the cached `targets` as unreliable.
   - Rebuild `targets` by binning the cache's own `ground_truth_boundaries` boundary timestamps (in seconds) onto `timestamps`: `targets[i] = 1` for any timestamp at/near a boundary.
   - Raise a clear `ValueError` (with lecture id) if a lecture yields no valid boundary label — do not silently drop or fabricate.
5. Fix acoustic dim in model init: `C2_AcousticChapterer(d_ac=32, ...)`, `C5(..., d_ac=32, ...)`.
6. Update eval cell: derive `gold_ts` from the **rebuilt** supervision + `timestamps`; keep `compute_all_chapter_metrics(gold_ts, preds, video_duration_sec=...)`.
7. **Persist the trained C5 checkpoint**: after training, save `state_dict` to `checkpoints/c5_real.pt` (create `checkpoints/` under project root if absent) so Phase 4 (S3/S4) and Phase 5 (Q2/Q3) reuse the same real C5 weights.
8. Remove/clear stored synthetic outputs in the notebook JSON (`outputs: []` on affected cells) so a re-run produces fresh, real numbers.
9. Update markdown wording that asserts fake findings; keep metrics/statistics machinery (Holm-Bonferroni + Bootstrap) untouched.

## Success Criteria
- [ ] Notebook no longer contains `torch.randn` for text/vis/ocr/acoustic feature generation (only legit init noise remains).
- [ ] Acoustic dim is 32 everywhere in model init and data path.
- [ ] Loader uses `create_lecture_splits`/`collate_lecture_batches` and guards against empty cache.
- [ ] **Data-health guard + targets rebuild from the cache's own `ground_truth_boundaries`; no lecture trains on empty or broken cached `targets`.**
- [ ] Trained C5 `state_dict` is persisted to `checkpoints/c5_real.pt` (usable by Phases 2 & 3).
- [ ] All synthetic output cells cleared (`outputs: []`).
- [ ] Notebook cells are valid JSON (parse with `nbformat`).

## Risk Assessment
- **Risk: `weights_only=False` torch.load on Colab vs local torch version** — already used by `dataset.py`; acceptable. Mitigation: load on same runtime as data was produced (local) first.
- **Risk: cached `targets` broken (verified: CA01 sum=0, CA05 sum=2, timestamps non-monotonic)** — Mitigation: **never train on raw cached `targets`**; rebuild via the guard from the cache's own `ground_truth_boundaries`. This is the key correctness fix for RQ1.
- **Risk: `ground_truth_boundaries` have duplicate/near-duplicate timestamps** (e.g. CA05 `190,191`, CA01 `458,459`) and sparse per-sentence bins → rebuilt `targets` may stay sparse. Mitigation: bin with tolerance / dedupe near-duplicate boundaries; guard raises a clear error listing affected lectures instead of silently training on bad labels; report low-data caveat in markdown; do not over-claim.
- **Risk: sequence lengths vary** → `collate_lecture_batches` pads to max; evaluation per-sample uses masks. Ensure eval loops per real sample, not padded whole-batch compare.

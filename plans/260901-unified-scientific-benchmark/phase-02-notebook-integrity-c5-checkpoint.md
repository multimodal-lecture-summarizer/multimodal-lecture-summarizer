---
phase: 2
title: Notebook Integrity & C5 Checkpoint
status: completed
priority: P1
dependencies:
  - phase-01
---

# Phase 2: Notebook Integrity & C5 Checkpoint — 01–05 Real-Data-Only (D-T15)

## Overview
**REAL-DATA-ONLY (D-T15):** Khử **toàn bộ** fabrication khỏi `01/02/03/04/05` notebooks + `run_rq2/rq3_benchmark.py` và khôi phục 100% real-data path. Không chỉ `03/04/05` — `01` (synthetic_test) và `02` (`np.random.randn` embeddings + `np.random.uniform` C2–C6) cũng phải chuyển sang real. `cached_features/*.pt` (20 lectures, 384d text/vis/ocr + 32d acoustic) qua `LectureFeatureDataset`/`create_lecture_splits`/`collate_lecture_batches`, rebuild `targets` từ `ground_truth_boundaries` (guard), `d_ac 64→32`, và persist `checkpoints/c5_real.pt` cho P3→P5. Không chạy notebooks (static edits per user constraint).

## Requirements
- Functional: **01** real-only: `synthetic_test.csv`/`synthetic_train.csv` chỉ được load để báo cáo “synthetic excluded from RQ tables” hoặc bỏ hẳn; eval tables chỉ `real_world_test.csv` (269 QA) + 20 cached `.pt` lectures (D-T15). **02** hết mock: thay `np.random.randn(…,384/16)` visual/acoustic (`cell 7`) bằng real `cached_features/*.pt` loads (`LectureFeatureDataset` hoặc `torch.load .pt → visual_features/acoustic_features 32d`), và thay `np.random.uniform` C2–C6 simulation (`cell 15`) bằng **real `C1–C6` inference** (`benchmarks/models/chaptering.py` + `benchmarks/data/dataset.py` collate) hoặc chuyển thành placeholder markdown `Skipped — run benchmarks/scripts/run_rq1_benchmark.py for real results`; `03` hết `torch.randn` synthetic (`generate_lecture_batch` → real loader + targets-rebuild guard), `C2/C5 d_ac 32`, eval dùng rebuilt supervision + `compute_all_chapter_metrics`; `04` `c5_predicted_boundaries` từ real C5 checkpoint (không cumulative-sum heuristic), OCR real từ cache hoặc `S4` gated `OCR pending`; `05` transcripts từ `q_and_a.json`+`cached_data["transcript_sentences"]` matched `video_name`, no answer-leak, `pred_boundaries` từ checkpoint (không `np.random.uniform`), single item-builder deduplicated; stale outputs cleared; `run_rq2:100` template OCR → real, `run_rq3:84` noise + `:99` template OCR → real. **Toàn bộ `01–05` grep mock `randn|uniform.*gold_boundaries|Slide Concept|ans_text` =0 (research cells).**
- Non-functional: guards fail-loud (empty cache, missing checkpoint, `video_name` mismatch, LLM/SBERT missing → `ValueError` không silent); reproducible `seed 42`; valid `nbformat` JSON; `run_rq1_benchmark.py` untouched; `pilot_qualification_runner.py` synthetic sanity giữ lại nhưng không vào RQ tables (ghi rõ comment `sanity only`).

## Architecture
```
Before (fabricated):
  03: generate_lecture_batch() torch.randn 384/384/384/64 + random injection → C1-C6 → fake Holm
  04: VT-SSum JSON (real) + ocr_slides=f"Slide {i}: {title} - Key Slide Concepts" + cumsum boundaries → S0-S4 fake
  05: real Q/A + sents=f"...confirms: {ans_text}" + pred=b+uniform(-10,10) + ocr=f"Slide: {sent[:40]}" → Q0-Q3 fake
  run_rq2:100 template OCR, run_rq3:84 noise + :99 template

After (real):
  LectureFeatureDataset(cached_features) → create_lecture_splits(seed 42) → collate_lecture_batches
    → guard: rebuild targets by binning ground_truth_boundaries (sec) onto timestamps (sec)
    → C1..C6 (d_ac 32) → train → checkpoints/c5_real.pt (persist)
  04: load C5 checkpoint → real predicted_boundaries per test lecture + real ocr_features/transcript → S0-S4
  05: q_and_a.json (20, video_name matches .pt) → cached transcript_sentences/timestamps/gt_boundaries
      → C5 checkpoint → real predicted_boundaries → Q0-Q3 (SBERT+LLM) vs real reference answer
```

Data-health finding (drives guard): `CA01 13 boundaries→targets.sum 0`, `CA05 27→2`, `DS12 44→2`, `DS11 44→1`, `timestamps` non-monotonic `[0,0,15.96,0,…]`, `manifest num_boundaries` mismatch. VT-SSum 0 overlap với cached lectures → rejected as GT.

## Related Code Files
- Modify: `experiments/notebooks/01_phase1_qualification_and_pilot.ipynb` (cell 4 synthetic_test → real-only note), `experiments/notebooks/02_phase2_frozen_data_and_runner.ipynb` (cell 7 randn embeddings → real cached_features loads; cell 15 uniform C2–C6 → real C1–C6 inference or placeholder), `experiments/notebooks/03_phase3_representation_and_chaptering.ipynb` (cells ~6,8,13,18 + markdown), `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb` (cells ~4,~8 + markdown), `experiments/notebooks/05_phase5_evidence_retrieval_and_qa.ipynb` (cells ~4,~15 deduplicated + markdown + LLM/SBERT guard), `benchmarks/scripts/run_rq2_benchmark.py:100`, `benchmarks/scripts/run_rq3_benchmark.py:84,99`
- Reference: `benchmarks/data/dataset.py:63 create_lecture_splits, :105 collate_lecture_batches`, `benchmarks/models/chaptering.py:279 C5_TemporalCrossAttentionTransformer`, `benchmarks/models/summarization.py:266 S4`, `benchmarks/metrics/qa_metrics.py:100 compute_all_qa_metrics`, `benchmarks/data/cached_features/*.pt` (20), `experiments/datasets/eduviqa/q_and_a.json` (20), `checkpoints/c5_real.pt` (new)
- Delete: stale `outputs: []` fabrication remnants; `generate_lecture_batch` cell; `02` mock `np.random.randn/uniform` blocks
- Verify untouched (except D-T15 allowlist): `benchmarks/scripts/run_rq1_benchmark.py` (0 randn research), `benchmarks/models/chaptering.py:279` init `torch.randn` allowed, `benchmarks/metrics/statistics.py` bootstrap RNG allowed, `tests/` mocks allowed

## Implementation Steps
1. **Map notebook cells:** `Read` 5 notebooks JSON, map `env/data/model-init/train/eval/stats/qualitative` cell indices. Đặc biệt `01 cell 4`, `02 cell 7` (randn), `02 cell 15` (uniform C2–C6).
2. **01 real-only (cell 4):** `eduvidqa_dir / "synthetic_test.csv"` → chỉ báo cáo `df_real_test` (269 QA) cho RQ tables; nếu giữ `df_syn_*` phải thêm markdown `> **D-T15:** Synthetic splits excluded from all RQ evaluation; shown for provenance only.` và không dùng `synthetic_test` cho bất kỳ metric nào. Xóa `print` synthetic khỏi eval path.
3. **02 real embeddings (cell 7):** Thay
   ```python
   visual_embeddings = np.random.randn(num_slides, 384)
   acoustic_embeddings = np.random.randn(num_slides, 16)
   ```
   bằng real loads:
   ```python
   # REAL-DATA-ONLY D-T15: không randn mock
   from benchmarks.data.dataset import LectureFeatureDataset
   _ds = LectureFeatureDataset(str(PROJECT_ROOT/"benchmarks/data/cached_features"))
   # map real lecture by id or pick first entry; hoặc torch.load .pt trực tiếp
   _sample = _ds[0]  # real cached lecture
   visual_embeddings = _sample["visual_features"].numpy() if hasattr(_sample["visual_features"],"numpy") else _sample["visual_features"]
   acoustic_embeddings = _sample["acoustic_features"].numpy() if hasattr(_sample["acoustic_features"],"numpy") else _sample["acoustic_features"]
   assert visual_embeddings.shape[1]==384 and acoustic_embeddings.shape[1]==32, f"Expected 384/32, got {visual_embeddings.shape}/{acoustic_embeddings.shape}"
   ```
   Nếu `LectureFeatureDataset` trả về dict khác, dùng `torch.load(PT_PATH, map_location="cpu")` và đọc `visual_features`/`acoustic_features` (32d, không phải 16d — sửa dim). Guard fail-loud nếu cache trống.
4. **02 real C1–C6 (cell 15):** Thay toàn bộ `pred_c2..c6 = [b + np.random.uniform(...)]` bằng **real inference** hoặc placeholder an toàn:
   ```python
   # REAL-DATA-ONLY D-T15: thay random simulation bằng real model hoặc skip
   from benchmarks.models.chaptering import C1_TextOnlyChapterer, C5_TemporalCrossAttentionTransformer
   from benchmarks.data.dataset import LectureFeatureDataset, collate_lecture_batches, create_lecture_splits
   # Option A (preferred): load cached C5 checkpoint if exists, else train 1 epoch demo on real cached_features, run collate batch through models C1..C6, collect predicted_boundaries
   # Option B (nếu chưa có checkpoint): skip simulated table, ghi markdown "C2-C6 simulation removed per D-T15; run `python -m benchmarks.scripts.run_rq1_benchmark --manifest benchmarks/manifests/frozen_manifest_v1.json` for real Holm table"
   ```
   Không để lại `np.random.seed`/`uniform`/`rand` nào trong research table cell. Xóa `np.random.seed(len(...))` dòng.
5. **03 loader + guard:** Replace `generate_lecture_batch` cell:
   ```python
   from benchmarks.data.dataset import LectureFeatureDataset, collate_lecture_batches, create_lecture_splits
   DATA_DIR = PROJECT_ROOT / "benchmarks" / "data" / "cached_features"
   if not any(DATA_DIR.glob("*.pt")): raise FileNotFoundError(f"Empty cache: {DATA_DIR}")
   train_ds, val_ds, test_ds = create_lecture_splits(data_dir=str(DATA_DIR), train_ratio=0.6, val_ratio=0.2, seed=42)
   train_batch = collate_lecture_batches([train_ds[i] for i in range(len(train_ds))])
   # ... val/test likewise
   ```
   Add guard+rebuild cell (before train): for each sample check `targets.sum()` vs `len(ground_truth_boundaries)`; if suspiciously low (<1 per 3-5 sents) rebuild `targets[i]=1` where `timestamps[i]` near `ground_truth_boundaries` (sec) with tolerance/dedupe; `raise ValueError(lecture_id)` if no valid label.
6. **03 model dims + checkpoint:** Fix `C2_AcousticChapterer(d_ac=32)`, `C5(..., d_ac=32)`; after train save `torch.save(model.state_dict(), PROJECT_ROOT/"checkpoints/c5_real.pt")` (mkdir if needed); update eval cell `gold_ts` from rebuilt targets + `timestamps`; clear `outputs: []`.
7. **04 real boundaries+OCR:** Load `checkpoints/c5_real.pt` into `C5_TemporalCrossAttentionTransformer(d_ac=32, d_model=256…)`; for each test lecture map to cached `.pt` (title substring) → `model.forward()` → `predicted_boundaries`; replace `ocr_slides` template with real `ocr_features`/transcript (if none, set `ocr_available=False`, exclude S4 from RQ2 Holm family, add markdown `OCR pending`); patch `run_rq2_benchmark.py:100` likewise; clear outputs, drop fake `+61.89%` claims.
8. **05 real transcripts (de-duplicate):** Single item-builder for cells 4 & 15:
   ```python
   # for each entry in q_and_a.json: match video_name → cached .pt → transcript_sentences/timestamps/gt_boundaries
   # predicted_boundaries = C5 checkpoint forward; ocr = real cached or empty+flag; NEVER read answer into sents/ocr/oracle
   # oracle_chapters from timestamps/gt_boundaries
   ```
   Patch `run_rq3_benchmark.py:84` noise → real checkpoint boundaries, `:99` template → real OCR; add `DenseEmbedder`+LLM availability guard (`assert sbert_loaded and llm_configured else raise`); clear fake `Answer-F1` outputs; update markdown to `q_and_a.json`+cached-transcript contract.
9. **Markdown + outputs sweep (01–05):** Remove fabricated "significant" language; keep Holm+bootstrap machinery; set all affected cells `outputs: []`, `execution_count: null`. `01` thêm D-T15 notice; `02` xóa `np.random.seed(42)` mock seed dòng.
10. **Static verification (no run):** `nbformat.read` 5 notebooks parse ok; `python -m json.tool` passes; `ast.parse` patched cells; `grep -R` qua `experiments/notebooks/01* 02* 03* 04* 05*` cho `randn`/`np.random.uniform.*gold_boundaries`/`Slide Concept`/`ans_text` → 0 (trừ allowlist `chaptering.py` init); confirm `d_ac=64→32`, `seed 42` + `checkpoints/c5_real.pt` consistent, `run_rq1_benchmark.py` diff chỉ init.

## Success Criteria
- [ ] **01** real-only: không `synthetic_test.csv` trong RQ tables (chỉ `real_world_test.csv` 269 QA); `grep` synthetic trong research cells =0 hoặc có D-T15 notice.
- [ ] **02** no `np.random.randn` embeddings (thay bằng real `cached_features` 384/32) và no `np.random.uniform` C2–C6 (thay bằng real `C1–C6` inference hoặc placeholder skip); `np.random.seed(42)` mock dòng đã xóa; acoustic dim 32 (không 16). **Hybrid+checkpoint verified.**
- [ ] `03` no `torch.randn` for features (init noise only), `d_ac 32`, loader `create_lecture_splits(seed 42)` + targets-rebuild guard (no lecture trains on empty/broken `targets`).
- [ ] `checkpoints/c5_real.pt` persisted and referenced by `04`/`05` + `run_rq2/rq3` (no fabrication fallback).
- [ ] `04`: `c5_predicted_boundaries` from checkpoint (not heuristic), OCR real or `S4` gated+markdown noted, no template OCR in notebook or `run_rq2`.
- [ ] `05`: no `ans_text` in `sents`/`ocr`/evidence, transcripts from `cached_data["transcript_sentences"]` matched `video_name`, `pred_boundaries` from checkpoint (no uniform noise), single item-builder, LLM/SBERT guard present, no template OCR in notebook or `run_rq3`, stale outputs cleared.
- [ ] **01–05** notebooks valid `nbformat` (5 files), 2 scripts `py_compile` ok, `run_rq1` only allowlist `torch.randn` init, `grep -R randn|uniform.*gold_boundaries|Slide Concept|ans_text` trên `experiments/notebooks/0*` =0.

## Colab Free Execution (Approach A, 2026-09-01 — keep 5 notebooks, >60' with checkpoint)

- **02 Phase 2 notebook** `02_phase2_frozen_data_and_runner.ipynb` (19c) — **8–12'** on T4 after D-T15 patch (cell 7 real PT load + cell 15 fail-loud, no random compute).
- **Hybrid cache:** `cell 7` tries `DRIVE_CACHE` then `benchmarks/data/cached_features/*.pt` (20 real), `weights_only=False` with dim assert 384/32; pads/crops to `num_slides` if VT-SSum slide count mismatch. No `np.random` fallback.
- **Runner:** `ResumableExperimentRunner` checkpoint `cache/runner_real_checkpoint.json` per item (20 VT-SSum test). Resume `run_variant(..., resume=True)` skips done IDs. Progress bar per item.
- **Stats cell 15:** Mock C2–C6 removed — now fail-loud with instruction `python -m benchmarks.scripts.run_rq1_benchmark --manifest ...` for real Holm table. Prevents silent fake stats in report.
- **Full vs demo:** Phase 2 stays 20 VT-SSum lectures (8–12'), not 300 — keeps <15' while P5 does full 300 with sharding. User approved `>60'` with checkpoint, so no need to split 02.

## Risk Assessment
- **R weights_only / torch version:** `torch.load` with `weights_only=False` already used in `dataset.py`; acceptable if same runtime; else add `map_location` guard.
- **R ground_truth_boundaries duplicates** (`190,191`, `458,459`): bin with tolerance/dedupe; guard raises clear error listing lectures, report low-data caveat.
- **R VT-SSum vs cache ID mismatch** (hash vs readable): substring/normalized matching + honest drop count, not fabrication.
- **R no OCR stream:** Expected — `S4/Q3` multimodal genuinely pending; markdown must state `Mode: Reveal-in-Packaging` so paper doesn't over-claim.
- **R padded batches:** `collate_lecture_batches` pads to max; eval per-sample via masks, not whole-batch compare.

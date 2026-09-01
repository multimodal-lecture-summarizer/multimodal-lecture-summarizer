# Notebooks 01–05 — Mức độ sửa & Hành động

**Ngày đánh giá:** 2026-09-01 (sau hợp nhất `260901-unified-scientific-benchmark`, sau commits `6111e97`, `4013290`, `4ab8963`)
**Phạm vi:** `experiments/notebooks/01..05` trong `multimodal-lecture-summarizer/experiments/notebooks/`

## Kết luận nhanh

| Notebook | Vai trò | Tình trạng hiện tại | Mức sửa còn lại | Ước lượng |
|----------|---------|---------------------|-----------------|-----------|
| **01** `01_phase1_qualification_and_pilot` | Dataset qualification & pilot (14 cells) | Visual EDA cho EduVidQA/VT-SSum/TIB/YTSeg, đã dùng **real** `probes/cache` + `candidate_media_20.json` + `chapter_metrics` — không có fabrication | **Nhẹ** — chỉ alias đường dẫn | 2 cells, ~10 dòng |
| **02** `02_phase2_frozen_data_and_runner` | Frozen manifests, runner, statistics (19 cells) | Đã dùng real `frozen_manifest_v1` + VT-SSum cache + `ResumableExperimentRunner` + `statistics` — không fabrication | **Nhẹ** — alias đường dẫn | 2 cells, ~10 dòng |
| **03** `03_phase3_representation_and_chaptering` | RQ1 C1–C6 (20 cells) | **Đã sửa nặng** ở `6111e97`+`phase-02`: `generate_lecture_batch`→`LectureFeatureDataset/create_lecture_splits/collate_lecture_batches`, `d_ac 64→32`, guard rebuild `targets` từ `ground_truth_boundaries`, `checkpoints/c5_real.pt` | **Xong** — chỉ cần clear stale outputs (11 cells còn outputs cũ) | 0 code, chỉ clear outputs |
| **04** `04_phase4_hierarchical_summarization` | RQ2 S0–S4 (16 cells) | **Đã sửa** ở `4013290`+`4ab8963`+`phase-03`: `LLM_PREFERENCE=auto` + `get_llm_engine` singleton, `USE_CACHED_FALLBACK`/`vtssum_clone_target`, 0 outputs (đã clear) | **Xong** — chỉ patch alias nếu đổi probe path | 0–5 dòng |
| **05** `05_phase5_evidence_retrieval_and_qa` | RQ3 Q0–Q3 (16 cells) | **Đã sửa** ở `6111e97`+`3b0cb82`: dùng `q_and_a.json` + `cached transcript`, `c5_real.pt`, `ground_truth_boundaries`, no `ans_text` leak, 6 cells còn outputs | **Xong** — clear stale outputs | 0 code |

**Tổng:** `03/04/05` đã được sửa triệt để, không còn `torch.randn`/`ans_text`/`Slide template`/`np.random.uniform` (scan 2026-09-01: 0 hits). `01/02` chưa từng bị fabrication. Việc còn lại là **đồng bộ đường dẫn** sau khi hợp nhất.

## Chi tiết từng notebook

### 01 — Qualification & Pilot (14 cells, 8 có outputs)
- Dùng real data: `plans/260830-1917-scientific-benchmark/probes/cache/eduvidqa/data/real_world_test.csv` (`cell 4`), `vtssum/test/22axpf7xhjwrzdzw7w77yc7mukreba37.json` (`cell 8`), `manifests/candidate_media_20.json` (`cell 11`), `03-colab-runbook` chapter metrics (`cell 13`).
- **Vấn đề sau hợp nhất:** `plans/260830-1917-scientific-benchmark/` đã xóa (giữ shim `manifests/`+`probes/output/` via `git checkout`), nhưng `probes/cache/vtssum` là **untracked cache** đã mất khi `Remove-Item`. `cell 8` sẽ `FileNotFoundError` nếu chưa regenerate.
- **Fix đề xuất (phase-01, 10 dòng):** Thêm helper đầu notebook:
  ```python
  LEGACY = PROJECT_ROOT / "plans" / "260830-1917-scientific-benchmark"
  UNIFIED = PROJECT_ROOT / "plans" / "260901-unified-scientific-benchmark"
  def resolve_legacy(*parts):
      for base in [LEGACY, UNIFIED / "framework", PROJECT_ROOT]:
          p = base.joinpath(*parts)
          if p.exists(): return p
      return LEGACY.joinpath(*parts)
  vt_sample_file = resolve_legacy("probes","cache","vtssum","test","22axpf7xhjwrzdzw7w77yc7mukreba37.json")
  ```
  Hoặc chạy `python plans/260830-1917-scientific-benchmark/probes/probe_vtssum.py` để regenerate cache.

### 02 — Frozen Data & Runner (19 cells, 11 có outputs)
- Real: `benchmarks/manifests/frozen_manifest_v1.json` (`cell 4`), `probes/cache/vtssum/test` (`cells 7,12`), `candidate_media_20.json` (`cell 18`), runner/statistics.
- **Fix:** Tương tự 01 — `cells 7,12` cần `resolve_legacy` cho `probes/cache/vtssum/test`. `cell 4` đã đúng (`benchmarks/manifests/` không đổi). Clear outputs nếu muốn `01`/`02` chạy lại sạch.

### 03 — Representation & Chaptering (20 cells, 11 có outputs)
- Scan: `LectureFeatureDataset` ✓, `create_lecture_splits` ✓, `c5_real.pt` ✓, `ground_truth_boundaries` ✓, `d_ac 32` ✓, **0 fabrication** (đã xóa `generate_lecture_batch`/`torch.randn`).
- **Còn lại:** 11 cells còn `outputs` cũ từ run synthetic trước `6111e97`. Theo `phase-02` yêu cầu `outputs: []` — chỉ cần `Clear All Outputs` trong Jupyter, không sửa code. Đã có guard rebuild `targets`.

### 04 — Hierarchical Summarization (16 cells, 0 outputs) ✅
- Scan: `get_llm_engine` ✓, `LLM_PREFERENCE` ✓, **0 fabrication**, 0 outputs (đã clear ở `4013290`).
- Đã có `USE_CACHED_FALLBACK` + `vtssum_clone_target` cho Colab. Nếu đổi probe path, thêm `resolve_legacy` như 01/02.
- Code đã rate-limit-free (`Qwen2.5-1.5B`/`Deterministic` via `benchmarks/models/llm_engine.py:110`).

### 05 — Evidence Retrieval & QA (16 cells, 6 có outputs)
- Scan: `q_and_a.json` ✓, `cached transcript` ✓, `c5_real.pt` ✓, `d_ac 32` ✓, **0 fabrication** (không còn `ans_text` leak hay `np.random.uniform`).
- **Còn lại:** 6 cells còn outputs fake `Answer-F1` cũ — cần clear. Đã có `DenseEmbedder`+LLM guard.

## So sánh với unified phases

- `01/02` → **Phase 1** Foundation (không code mới, chỉ freeze & probe gate W1-2)
- `03` → **Phase 2** Notebook Integrity (đã xong 90%, còn clear outputs)
- `04` → **Phase 3** RateLimit-Free Engine (đã xong, code ở `4013290`)
- `05` → **Phase 5** Retrieval Scale (đã xong phần de-fabricate, còn scale 30→5,252 QA ở `phase-05`)

## Hành động tiếp theo (để notebooks chạy lại)

1. **Compat shim (đã làm 2026-09-01):** `git checkout -- plans/260830-1917-scientific-benchmark/manifests/candidate_media_20.json plans/260830-1917-scientific-benchmark/manifests/frozen_manifest_v1.json plans/260830-1917-scientific-benchmark/probes/output` — giữ `plans/260830-...` như shim cho `01/02/04` đọc `manifests` & `probes/output` (untracked `probes/cache` cần regenerate riêng).
2. **Regenerate VT-SSum cache (nếu cần chạy 01/02):** `python plans/260830-1917-scientific-benchmark/probes/probe_vtssum.py` hoặc giữ fallback `vtssum_clone_target` trong `04`.
3. **Clear outputs:** Mở `03` và `05` → `Kernel → Restart & Clear All Outputs` → save (JSON `outputs: []`).
4. **(Tùy chọn) Thêm `resolve_legacy` helper** vào `01` cell 2 và `02` cell 2 để hỗ trợ cả hai đường dẫn `260830` và `260901-unified` — snippet ở trên, ~10 dòng/cell.
5. **Không sửa `run_rq1_benchmark.py`** (đã clean) — `phase-06` sẽ verify untouched.

## Rủi ro nếu không sửa

- `01/02` sẽ `FileNotFoundError` tại `probes/cache/vtssum` nếu cache chưa regenerate — không ảnh hưởng unified execution (probes là derived data), nhưng EDA visuals sẽ trống.
- `03/05` stale outputs có thể gây hiểu nhầm "significant" cũ — cần clear trước khi share notebooks.

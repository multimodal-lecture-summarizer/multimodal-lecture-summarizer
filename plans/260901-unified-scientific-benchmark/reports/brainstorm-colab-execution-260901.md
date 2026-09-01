# Brainstorm: Colab Free Execution cho Unified Benchmark (5 Notebooks, Real-Data-Only)

**Date:** 2026-09-01
**Plan:** `plans/260901-unified-scientific-benchmark/plan.md` (6 phases, D-T01…D-T15)
**Request:** Notebook phải thực thi tốt trên Colab Free, không quá dài, có thể chia nhỏ nếu cần — bổ sung chi tiết vào plan.
**Decisions:** `>60' với checkpoint chấp nhận`, `giữ 5 notebooks`, `Hybrid cache (precomputed + auto-extract)`, `Full 300/300`.

---

## 1. Scout Summary (what I found)

- **Project:** `multimodal-lecture-summarizer` — Python 3.11, PyTorch 2.5.1+cu124, `transformers 4.57.6`, `faster-whisper 1.1.0`, `paddlepaddle 2.6.2`, `chromadb 1.5.9`. No `package.json` — pure PyTorch backend + `experiments/notebooks/*.ipynb`.
- **Current notebooks 01–05:** 01(14c/7outs), 02(19c/9outs), 03(20c/11outs), 04(16c/0outs), 05(16c/6outs). All already patched to real-data-only per D-T15 (grep `randn|uniform.*gold` =0). 03/04/05 previously used `torch.randn`/`Slide Concept`/`ans_text` — removed via `fix-notebooks` + 2026-09-01 patches.
- **Existing infra:** `benchmarks/` has `FrozenManifestManager` (frozen_manifest_v1.json, 53k records), `FeatureCache`, `ResumableExperimentRunner`, `statistics.py` (Holm+bootstrap), `cached_features/*.pt` (20 lectures, 384/32d), `llm_engine.py` (Qwen2.5-1.5B singleton + Deterministic v2). Already supports sharding and checkpoint.
- **Plan constraints:** D-T09 `HF_HOME=/root/.cache/huggingface` (SSD) vs Drive for features, D-T08 strict 32k/512/200f, D-T15 no mock, Colab T4 15GB/12h kill, VISTA 300 ~10h ASR (2'/video), YTSeg 300 ~6h download+features, `04` 100 vids <30' with cache `>80%` hit.

---

## 2. Exact Requirements (5 mandatory)

1. **Expected output:** 5 notebooks `01–05` vẫn là 5 files, mỗi file chạy end-to-end trên Colab Free T4 **>60' chấp nhận** nếu có checkpoint/resume, không cần <15'. Full 300/300 scale retained.
2. **Acceptance criteria:** (a) Mỗi notebook: `pip install <2'`, model load `<60s`, `hybrid cache` thử `Drive/cache` trước, nếu miss tự extract DINOv2/PaddleOCR/Whisper và lưu cache; (b) Resume sau Colab kill không mất kết quả (cache hit `>80%`); (c) Full 300/300 hoàn thành với ≤2 lần resume (12h×2); (d) Grep D-T15 mock =0; (e) `04`/`05` 0 Gemini khi `auto`.
3. **Scope boundary OUT:** Không tách 5→7–10 notebooks; không train Video-LLM mới; không thêm UI/billing; không redistribute video; không downgrade sang demo-only 20 videos.
4. **Non-negotiable constraints:** Real-data-only D-T15, D-T08 strict, D-T09 SSD vs Drive, C5/C6 frozen D-T02, same T4/hardware per comparison, 3 seeds (42,1337,2026), per-video predictions retained.
5. **Touchpoints:** `experiments/notebooks/01–05`, `benchmarks/core/feature_store.py` (shard/resume), `benchmarks/models/llm_engine.py`, `benchmarks/scripts/fetch_ytseg_subset.py` + `vista_subset_asr.py` (new), `benchmarks/manifests/frozen_manifest_v1.json`, `cache/feature_store_real`, `outputs/phase4_cache`, `plans/260901-unified-scientific-benchmark/phase-0*`.

---

## 3. Approaches Evaluated

### Approach A — Keep 5 Notebooks + Hybrid Cache + Checkpoint Resume (Recommended, KISS/YAGNI)

- **Giữ** `01–05` nguyên 5 files; mỗi notebook thêm: (i) header `RUN_MODE` + `USE_CACHE` toggle, (ii) `hybrid_load()` thử `Drive/feature_store` → fallback `extract_and_save()`, (iii) `ResumableExperimentRunner` + `outputs/*_cache` + `feature_store --shard --resume` với progress bar và `torch.cuda.empty_cache()` per 10 items, (iv) `>60'` split nội bộ bằng `shard` loops (50 videos/shard) — mỗi shard là 1 cell, có thể `Run > Restart & Run All` và tự skip shards done.
- **Pros:** Ít xáo trộn nhất (diff <100 LOC/notebook), giữ 5-file mental model, reuse infra sẵn có, YAGNI không tạo orchestrator mới, mỗi notebook tự-contained để share Colab link riêng.
- **Cons:** Mỗi notebook dài (>60') — user phải hiểu resume (chấp nhận per discovery). Không có dashboard tập trung.
- **Risk:** VISTA 10h vượt 12h nếu không sharded — mitigated bằng 6 shards ×50 (mỗi shard ~1.7h, checkpoint per video, xóa raw mp4 sau shard).

### Approach B — Keep 5 + Run Mode Toggle (Demo 20 vs Full 300)

- Thêm `DEMO_MODE = os.getenv("DEMO", "false")` → `n=20` nhanh <15' cho review, `FULL` chạy full 300 với resume. Cùng Hybrid + checkpoint như A.
- **Pros:** Review nhanh, thesis vẫn full 300 khi cần.
- **Cons:** Thêm branch logic, risk demo/full divergence (cần CI đảm bảo cùng code path). User đã chọn Full-only nên extra toggle là YAGNI nếu không yêu cầu.
- **Verdict:** Để A làm default, B là optional enhancement (1 flag, ~5 dòng) có thể thêm sau nếu cần demo.

### Approach C — 5 + 1 Orchestrator Notebook (`00_colab_setup.ipynb`)

- Thêm `00` mount Drive, `pip install --quiet`, `huggingface-cli login`, check `HF_HOME` vs `Drive`, verify `frozen_manifest_v1`, rồi link tới 01–05.
- **Pros:** Onboarding mượt, tập trung setup.
- **Cons:** Thêm file mới, duplicate `connect_drive.ipynb` đã tồn tại, tăng maintenance. User chọn giữ 5 nên C là overhead.
- **Verdict:** Không làm — cải thiện `01 cell 1` setup thay vì new notebook.

**Recommendation:** **Approach A** — giữ 5 notebooks, bổ sung chi tiết Hybrid + Checkpoint vào plan, mỗi notebook `>60'` chấp nhận với shard resume.

---

## 4. Final Design — Approach A Detail

### Per-Notebook Execution Budget (Colab Free T4, D-T15 real-data-only)

| Notebook | Nội dung chính | Hybrid Cache | Shard/Resume | Thời gian ước tính (T4) | Cells thêm |
|----------|----------------|--------------|--------------|-------------------------|------------|
| **01** | Qualification & pilot (EDA 5 datasets, 53k manifest, chapter metrics) | Thử `plans/.../framework` + `probes/cache` (real), nếu miss chạy `probe_* --simulate` (<2') | Không cần (chỉ viz) | **5–8'** | +1 header cell (RUN_MODE, cache check, D-T15 note) |
| **02** | Frozen manifests + Feature Store + Runner + Stats (D-T15 patched) | `cached_features/*.pt` (20) → `FeatureCache` validate; visual/acoustic từ `torch.load .pt` real 384/32 (đã patch `cell 7`); Runner 20 VT-SSum test | Runner checkpoint `cache/runner_real_checkpoint.json` per item, resume | **8–12'** | Đã patch `cell7`+`cell15`, thêm progress bar |
| **03** | RQ1 C1–C6 (train + collar F1/Pk) | `create_lecture_splits(seed 42)` + `collate_lecture_batches` real; `targets` rebuild guard | Train checkpoint `checkpoints/c5_real.pt` per epoch, resume | **20–40'** (1 epoch demo) / **60–90'** (3 seeds nếu chạy full) — chia 3 cells (1/seed) để resume per seed |
| **04** | RQ2 S0–S4 (hierarchical summarization) | `get_llm_engine(auto)` Hybrid (Qwen/Deterministic), `outputs/phase4_cache` `sha1(prompt+boundaries)` | 6 shards ×~17 vids, batch 4 chapters, `empty_cache` per 10 lectures | **20–30'** (100 vids, cache hit >80% second run <5') |
| **05** | RQ3 Q0–Q3 (retrieval QA) | `q_and_a.json` 5,252 QA → `DenseEmbedder` SBERT singleton + BM25, `outputs/phase5_cache` | Per-question cache, `top_k=3` (600tok) | **15–25'** (5k QA, SBERT offline) |
| **VISTA ASR (part of 04/05 prep, Phase 4)** | `vista_subset_asr.py` 300 videos | `snapshot_download` sharded 6×50 → `whisper-small` per mp4 → `probes/cache/vista_subset/*.json` | 6 shards ×50, `rm` raw mp4 sau shard, resume per video | **~10h** total, mỗi shard **~1.7h** — khuyến nghị chạy riêng 6 lần `!python -m ... --shard X` hoặc 1 notebook cell per shard với checkpoint per video |

**Tổng cho full thesis (300/300):** ~12h nếu chạy liền (vượt 12h kill) → **cần 2 sessions với resume** (đã thiết kế idempotent). Đây là lý do `>60'` chấp nhận là key.

### Hybrid Cache Pattern (to be added to each notebook header)

```python
# Hybrid D-T15 + D-T09
CACHE_DIR = PROJECT_ROOT / "cache" / "feature_store_real"
DRIVE_CACHE = Path("/content/drive/MyDrive/feature_stores") if Path("/content/drive").exists() else None
USE_CACHE = os.getenv("USE_CACHE", "auto")  # auto|force_refresh|offline

def hybrid_load(video_id: str):
    for base in [DRIVE_CACHE, CACHE_DIR]:
        if base and (base / f"{video_id}.pt").exists():
            return torch.load(str(base / f"{video_id}.pt"), map_location="cpu", weights_only=False)
    # Fallback: extract thật (DINOv2+PaddleOCR+Whisper) và lưu cache — không mock
    return extract_and_save(video_id, save_to=DRIVE_CACHE or CACHE_DIR)
```

### Checkpoint Pattern (per phase)

- **Feature store:** `python -m benchmarks.core.feature_store --shard 0/6 --resume --data-dir cache/feature_store_real` (skips done via `manifest.json`).
- **VISTA:** `python -m benchmarks.scripts.vista_subset_asr --shard 0 --limit 50 --resume` (per-video JSON, `rm` mp4).
- **Summarization/QA:** `outputs/phase4_cache/{sha1}.json` + `outputs/phase5_cache/{hash}.json` — second run `llm.generate` count 0 for hit.
- **Training:** `checkpoints/c5_real.pt` per epoch, `Trainer(resume_from_checkpoint=True)`.

### Plan Patch Checklist (Approach A)

- [ ] `phase-01` — thêm `Hybrid + checkpoint` note vào Implementation Steps (đã có `probes/gate_w1.json` simulate).
- [ ] `phase-02` — đã patch `01 cell4` + `02 cell7/15` D-T15; thêm execution budget table + `hybrid_load` snippet vào phase file.
- [ ] `phase-03` — thêm `cache_key = sha1(chapter+boundaries)` + `batch 4` + `empty_cache` per 10 lectures + shard resume detail.
- [ ] `phase-04` — thêm 6-shard breakdown (50/shard, 1.7h/shard, resume per video, `rm` mp4) + WER gate `<30%` fail-loud.
- [ ] `phase-05` — thêm YTSeg 6-shard + retrieval per-question cache + hybrid SBERT/BM25 weight freeze.
- [ ] `phase-06` — thêm D-T15 grep gate `experiments/notebooks/0*` =0 vào consistency sweep.

---

## 5. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Colab 12h kill giữa VISTA 10h | High | 6 shards, checkpoint per video, resume `--resume`, xóa raw mp4 per shard để tránh Drive quota 30GB blowup |
| Drive sync corrupt model weights | Medium | D-T09: `HF_HOME` SSD only, Drive chỉ cho `feature_store`/`outputs` |
| Hybrid cache miss → phải extract DINOv2 (slow) | Medium | Warm cache từ `benchmarks/data/cached_features` 20 mẫu; shard streaming; `pip install` <2' |
| `04` VRAM OOM Qwen 1.5B FP16 | Medium | Fallback `Deterministic` per remaining lectures + log, không crash |
| `>60'` gây timeout perception | Low | Progress bar per shard + `print(f"Shard {i}/6 done — resume with --shard {i+1}")` |

---

## 6. Success Metrics

- Mỗi notebook `01–05` có header `RUN_MODE/USE_CACHE` + hybrid + checkpoint snippet, diff `<100 LOC`.
- `01` <8', `02` <12', `03` 1 seed <40' (3 seeds <90' với per-seed resume), `04` 100 vids <30' (second run <5'), `05` 5k QA <25', VISTA 300 ~10h với ≤2 resumes (6 shards).
- `grep -R randn|uniform.*gold` trên `experiments/notebooks/0*` =0 (D-T15) — already 0 after 2026-09-01 patches.
- `ck plan validate plans/260901-unified-scientific-benchmark/plan.md` pass, `ck plan status` 6 phases `pending→done` via checkpoint.

---

## 7. Next Steps

1. Patch unified plan phases 01–06 với bảng execution budget + hybrid + shard resume (Approach A detail trên).
2. Commit `docs(brainstorm): colab execution 5 notebooks hybrid+checkpoint`.
3. Handoff to `/ck:plan` — no new plan needed (patch existing `260901-unified-scientific-benchmark`), or run `ck plan check 1 --start`.

---

## Appendix — Alternatives Rejected

- **Tách 7–10 notebooks:** Over-engineering for `>60'` chấp nhận; giữ 5 đủ, thêm shard loops nội bộ là đủ.
- **Tự extract mọi lần (no cache):** Vi phạm YAGNI, chậm gấp 3×, không cần khi đã có real `cached_features` 20 mẫu.

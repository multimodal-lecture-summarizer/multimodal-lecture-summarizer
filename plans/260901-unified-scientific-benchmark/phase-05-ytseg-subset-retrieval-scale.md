---
phase: 5
title: YTSeg Subset & Retrieval Scale
status: completed
priority: P1
dependencies:
  - phase-02
  - phase-03
---

# Phase 5: YTSeg Subset & Retrieval Scale

## Overview
Scale RQ1 (YTSeg) từ `n=20→300` lecture/science để đạt power cho `C5>C1` @ `d=0.3` (D-T10 cần ≥100) và scale RQ3 retrieval/QA từ 30-item demo lên full EduVidQA `5,252` QA offline. Gộp `hardening` P1 (YTSeg fetch + feature_store sharding) và P3 (retrieval hardening) thành 1 phase với `frozen_manifest_v2.json`, `feature_store` sharded tới Drive, `DenseEmbedder all-MiniLM-L6-v2` SBERT+BM25 hybrid `top_k=3`, chunked `outputs/phase4_cache` + `outputs/phase5_cache` resume.

## Requirements
- Functional: `benchmarks/scripts/fetch_ytseg_subset.py` HF `retkowski/ytseg` filter `category==Education` + `duration>600s` + `chapters≥3`, over-fetch 500 survive 30-40% attrition → `n=300`, `yt-dlp` retries 3, `TransNetV2` fallback 1fps, `feature_store.py` (`--shard 0..5 ×50`, `--resume`) sharded tới `FEATURE_STORE=Drive/feature_stores` (HF weights ở `HF_HOME` SSD per D-T09), `FrozenManifestManager` freeze `manifests/frozen_manifest_v2.json` (`version 2.0.0-frozen-ytseg`, seed 42, 0.6/0.2/0.2, `verify_split_leakage passed`, `targets` rebuilt per P2 guard). `05` retrieval: full `q_and_a.json` 5,252 QA (not 30), `DenseEmbedder` SBERT singleton + BM25 hybrid, `top_k=3` (600 tokens) D-T08, per-question cache `outputs/phase5_cache`, `04` chunked `outputs/phase4_cache` (sha1 chapter+boundaries hash), single `llm` from P3.
- Non-functional: Idempotent, resume không duplicate manifest entries, 100-video fetch+features `<6h` T4, `04` 100 vids `<30 min` + `05` 5k QA `<30 min` offline, 0 Gemini khi `auto`, cache hit `>80%` on resume, per-video predictions retained.

## Architecture
```
P5 YTSeg branch:
  HF retkowski/ytseg 19,299 → filter Education/lecture (duration>600s, chapters≥3)
    → yt-dlp 500 (--simulate probe first, retries 3) → ~300 with chapters (40% attrition)
    → feature_store.py --shard 0..5 (each 50) → Drive/feature_stores shards
        ASR whisper-small (32d acoustic), DINOv2 ViT-S/14 (384d vis), PaddleOCR v3 0.6 (384d ocr)
        embeddings + targets rebuilt from ground_truth_boundaries (P2 guard)
    → manifests/frozen_manifest_v2.json (version 2.0.0-frozen-ytseg, seed 42, 0.6/0.2/0.2, leakage 0)
    → C5/C6 training (frozen D-T02) ×3 seeds (42,1337,2026) on v2 → collar F1±3/5/10s, Pk, WindowDiff

P5 Retrieval branch (offline):
  EduVidQA full 5,252 QA (real 270 QA/99 vids + synthetic 4982/197, MIT) → match video_name→cached .pt/vista_subset transcript
    → DenseEmbedder all-MiniLM-L6-v2 SBERT singleton + BM25 hybrid → top_k=3 (600 tokens, D-T08)
    → llm.generate (P3 auto→Qwen/Deterministic) → answer → outputs/phase5_cache/{q_hash}.json
    → cache per lecture batch, torch.cuda.empty_cache() per 10 lectures

Cache invalidation: cache_key = sha1(chapter_text+variant_id+predicted_boundaries_hash)
```

## Related Code Files
- Create: `benchmarks/scripts/fetch_ytseg_subset.py`, `manifests/frozen_manifest_v2.json`, `cache/feature_store_real` shards (Drive), `outputs/phase4_cache/`, `outputs/phase5_cache/`, `tests/test_ytseg_manifest.py` + `tests/test_phase4_cache.py` + `tests/test_retrieval_scale.py` (post-hoc)
- Modify: `benchmarks/core/feature_store.py` (add `--shard`/`--resume`, skip done videos via `manifest.json`), `benchmarks/data/dataset.py` (support v2 manifest path), `experiments/notebooks/05_phase5_evidence_retrieval_and_qa.ipynb` (MAX_QUESTIONS removal, full load), `benchmarks/models/retrieval_qa.py` (DenseEmbedder singleton, hybrid BM25), `experiments/notebooks/04_phase4_hierarchical_summarization.ipynb` (cache resume batch already in P3)
- Delete: (none)
- Reference: `benchmarks/core/feature_store.py`, `benchmarks/data/dataset.py:63/105`, `benchmarks/models/chaptering.py:279`, `benchmarks/models/llm_engine.py` (P3), `experiments/datasets/eduviqa/q_and_a.json` (5252), `benchmarks/models/retrieval_qa.py`, `benchmarks/metrics/chapter_metrics.py`

## Implementation Steps
1. **Implement `fetch_ytseg_subset.py`:**
   ```python
   # load_dataset("retkowski/ytseg", split="test", trust_remote_code=True)
   # filter: category==Education or title contains lecture/science/course, duration>600, len(chapters)≥3
   # over-fetch 500 → yt-dlp --simulate probe → keep ~300 with downloadable media
   # for shard in 0..5: yt-dlp downloads (retries 3, format best) → feature_store per video (ASR+DINOv2+PaddleOCR per D-T04, timestamps non-monotonic guard)
   # write shards to cache/feature_store_real (Drive) with provenance.json (checksum, extractor revision, sampling rate)
   # call FrozenManifestManager.freeze(v2, seed 42, split 0.6/0.2/0.2) → manifests/frozen_manifest_v2.json
   # assert verify_split_leakage().passed==True, len(items)≥300 (fallback gate: if 100-300 acceptable per hardening, but unified locked 300)
   ```
   Check Drive quota 30 GB before start (`rclone` or `du`), fail early if insufficient.
2. **Shard `feature_store.py`:** Add `python -m benchmarks.core.feature_store --shard 0/6 --resume --data-dir cache/feature_store_real` reads existing `manifest.json` in shard and skips done videos. `HF_HOME` stays local SSD (model weights), only finished `.pt` sync to Drive. Keep `torch.hub DINOv2` on SSD.
3. **Rebuild supervision (reuse P2 guard):** After each `.pt` write, assert `targets` rebuilt from `ground_truth_boundaries` (bin boundary timestamps sec onto `timestamps` sec with tolerance, dedupe near-duplicates like `190,191`). No lecture with `targets.sum()==0`.
4. **Scale `05` retrieval:** In `05` notebook + `retrieval_qa.py` remove `MAX_QUESTIONS_PER_LECTURE` demo cap, load full `q_and_a.json` (5252), `build_qa_items()` should produce `≥1000` (full) not 30; hybrid `DenseEmbedder` (`sentence-transformers all-MiniLM-L6-v2` singleton) + `rank_bm25` → `top_k=3` (600 tokens) strict D-T08; per-question `outputs/phase5_cache/{hash}.json` (key=question+chapter hash).
5. **Wire P3 engine cache:** Ensure `04` uses P3 `get_llm_engine(auto)` singleton: `test_04_uses_deterministic_on_cpu` (mock `torch.cuda.is_available=False` → `Deterministic`), `test_04_cache_resume` (2 dummy chapters → write → re-run `generate` count 0), `test_04_budget_enforced` (`max_source 32k output ≤512`), batch 4 chapters/call, share `summarizers["S0"].llm is summarizers["S3"].llm`.
6. **Idempotency & verification:** Run `fetch_ytseg_subset.py --dry-run --shard 0 --limit 5` twice, assert no duplicate manifest entries; `grep` `frozen_manifest_v2.json` `n=300` `version 2.0.0-frozen-ytseg` leakage 0; `test_chaptering_nms` + `test_statistics` no regression on C5/C6 D-T02; `05` offline no `google.generativeai` import when `auto`.

## Success Criteria
- [ ] `fetch_ytseg_subset.py` tồn tại, `frozen_manifest_v2.json` `n=300` (hoặc `≥100` nếu attrition cao hơn 40%), `version 2.0.0-frozen-ytseg`, `zero leakage`, `seed 42`, per-video `ground_truth_boundaries` non-empty. **Sharded 6×50, Hybrid.**
- [ ] `cache/feature_store_real` có `n` `.pt` với `text 384d, visual 384d, ocr 384d, acoustic 32d`, `targets` rebuilt sum>0, sharded idempotent `--resume` no duplicates. **Hybrid: precomputed 20 + 280 new.**
- [ ] `04` `auto` 0 Gemini, 100 vids `<30 min` (25 `<10 min`) / 300 vids `~60'` with shard resume, second run cache hit `>80%`, `D-T08` obeyed, single `llm` instance.
- [ ] `05` full EduVidQA `≥1000` QA (target 5252) offline (**Hybrid SBERT cache**), no Gemini import, `Q0-Q3` complete, `Recall@3/MRR` computed, `test_retrieval_scale.py` post-hoc pass (`test_05_full_eduviqa_not_30`, `test_05_offline_no_gemini`). **Per-question cache makes 5k QA <25'.**
- [ ] `test_ytseg_manifest.py` 3/3 post-hoc pass: `has_n_ge_100_and_leakage_zero`, `targets_rebuilt_not_empty`, `feature_store_sharded_idempotent`.

## Colab Free Execution (Approach A, 2026-09-01 — keep 5 notebooks, >60' with checkpoint, Hybrid)

- **YTSeg branch:** `fetch_ytseg_subset.py` 6 shards `over-fetch 500 → 300` survives 40% attrition. Mỗi shard 50 videos: `yt-dlp --simulate` probe (<1') → download → `feature_store.py --shard X --resume` (DINOv2+PaddleOCR+whisper-small 32d). **Time:** ~1h/shard → **~6h** total for 300. Resume per video (skip if `.pt` exists). Check Drive quota 30GB before start, `rm` raw mp4 per shard.
- **Hybrid:** `03`/`05` notebooks `hybrid_load(video_id)` tries `DRIVE_CACHE/cache/feature_store_real/*.pt` → fallback `extract_and_save()` real extraction and uploads to Drive. Precomputed 20 `benchmarks/data/cached_features/*.pt` gives instant demo; 280 new extracted progressively.
- **Retrieval:** `05` Q0–Q3 `DenseEmbedder all-MiniLM-L6-v2` singleton cached in `HF_HOME` SSD (1.6GB) — load once `<60s`. Per-question `outputs/phase5_cache/{q_hash}.json` → second run <2'. 300-notebook full `~60'` chấp nhận per user choice.
- **Budget:** D-T08 strict per shard too — `assert_budget` per shard, not just final table.

## Risk Assessment
- **R media attrition 40%:** Over-fetch 500 + `yt-dlp --simulate` probe first, log attrition per `01-dataset-manifest.md`; nếu còn `<300` thì document attrition rate và dùng actual `n` (nhưng warn power giảm).
- **R T4 disk 100 GB + Drive 30 GB:** Shard + stream to Drive, check quota before start; mỗi shard xong `rm` local raw mp4.
- **R leakage:** Reuse `FrozenManifestManager.verify_split_leakage()`; fail-loud nếu `passed==False`.
- **R HF OOM per 50 vids:** `auto` catch `CUDA OOM` → fallback `Deterministic` for remaining lectures + log; cache invalidation phải include `predicted_boundaries` hash.
- **R SBERT vs BM25 hybrid balance:** Hybrid phải có weight fixed trước run; đổi weight = new variant, không silent tuning.

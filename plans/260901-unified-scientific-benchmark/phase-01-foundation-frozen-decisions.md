---
phase: 1
title: Foundation & Frozen Decisions
status: completed
priority: P1
dependencies: []
---

# Phase 1: Foundation & Frozen Decisions

## Overview
Đóng băng toàn bộ scientific framework từ `260830-1917-scientific-benchmark` (master plan, dataset manifest, benchmark matrix, colab runbook, RQ mapping, timeline, risks, decisions-log) thành constraints thực thi cho 5 phases sau. Không code feature mới — chỉ chuẩn hóa decisions, env, và gates để P2→P6 không drift.

## Requirements
- Functional: Mọi `D-T01…D-T14` + `D-S01…D-S04` được encode thành một `decisions-log.md` canonical (hoặc `benchmarks/docs/frozen_decisions.json`) với HF commit hash pinned, C5/C6 frozen, OCR/visual/sampling revisions, budget, fallback tier. `manifests/frozen_manifest_v1.json` (20 lectures hiện tại) được verify; dataset qualification gates (YTSeg/VISTA/TIB/EduVidQA/VT-SSum) có checklist pass/fail. Timeline M1 (W1-2 pilot) deliverables đóng.
- Non-functional: YAGNI — reuse docs hiện tại, chỉ bổ sung file freeze; reproducibility: mọi revision (model, dataset, prompt) có hash; no execution ngoài probe 20 items.

## Architecture
```
260830 docs (00..06 + decisions-log + related-work) ──► P1 freeze
  ├─ 00-master-plan: thesis, C1-C4 falsifiable, architecture diagram
  ├─ 01-dataset-manifest: Tier A-F matrix + 3-tier VISTA→TIB fallback (D-T05/D-T12)
  ├─ 02-benchmark-matrix: C0-C7 / S0-S4 / Q0-Q3 / E1-E4 + controls + budgets
  ├─ 03-colab-runbook: W1-2 pilot, HF SSD vs Drive, feature cache, stats, failure policy
  ├─ 04-rq-mapping: H1-H4, Holm families (4+4+3), power pilot D-T10, prereg checklist
  ├─ 05-timeline: 26w M1→M6, scope-cut order, related-work v1/v2
  ├─ 06-risks: R1→R18 + stop/go
  └─ decisions-log: D-T01…D-T14 + D-S01…D-S04 + VISTA approved 2026-08-31
         │
         ▼
  frozen_decisions.json + manifests/frozen_manifest_v1.json + provenance.json
  (seed 42, 60/20/20 split, verify_split_leakage=pass)
```

Key frozen values (must not change without new variant ID):
- `D-T01/D-T03`: E4/C7 = `Qwen3-VL-4B-Instruct` FP16, commit hash pinned W1, optional `Qwen3-VL-8B-AWQ`.
- `D-T02`: C5 4-layer cross-attn 256h 3 tokens BCE vs C6 concat-only.
- `D-T04`: PaddleOCR v3 `ch_PP-OCRv4` 0.6, DINOv2 `dinov2_vits14` 384d, TransNetV2 →1fps fallback.
- `D-T07`: Holm-Bonferroni within RQ (R1 4Δ, R2 4 pairs, R3 3 pairs) α=0.05 + Cohen d / Hedges g + bootstrap 1000.
- `D-T08`: source 32k / output 512 / frames 200 @448px — strict (fail not scale).
- `D-T09`: `HF_HOME=/root/.cache/huggingface` (SSD) vs `FEATURE_STORE=/content/drive/MyDrive/feature_stores`.
- `D-S03`: single-author audit + LLM-as-a-Judge (G-Eval/AlignScore/Salient QA).

## Related Code Files
- Create: `benchmarks/docs/frozen_decisions.json` (hoặc cập nhật `plans/260830-1917-scientific-benchmark/decisions-log.md` nếu giữ), `benchmarks/manifests/frozen_manifest_v1.json` verify log
- Modify: `benchmarks/manifests/frozen_manifest_v1.json` (add `version`, `git_commit`, `dataset_revision` if missing), `.env.example` / `benchmarks/core/feature_store.py` (HF_HOME doc)
- Delete: (none)
- Reference: `plans/260830-1917-scientific-benchmark/00-master-plan.md:1-172`, `01-dataset-manifest.md:1-197`, `02-benchmark-matrix.md:1-202`, `03-colab-runbook.md:1-319`, `04-rq-mapping.md:1-154`, `05-6month-timeline.md:1-185`, `06-risks-mitigations.md:1-110`, `decisions-log.md:1-162`, `related-work-bibliography.md`

## Implementation Steps
1. **Consolidate decisions:** Đọc toàn bộ 8 docs `260830`. Tạo `benchmarks/docs/frozen_decisions.json` (hoặc patch `decisions-log.md`) chứa `D-T01…D-T14`, `D-S01…D-S04`, `VISTA approved 2026-08-31`, `E4/C7 identity`, `C5/C6 spec`, `budget BUDGET={source 32k, output 512, frames 200, res 448}`. Pin `HF commit hash` cho `Qwen3-VL-4B-Instruct` sau W1 pilot (để trống `__(fill after pilot)__` nếu chưa có).
2. **Verify existing manifest:** Chạy `FrozenManifestManager.verify_split_leakage()` trên `benchmarks/manifests/frozen_manifest_v1.json` (20 lectures). Assert `passed==True`, `n=20`, `seed 42`. Ghi `manifests/manifest_verify_v1.json` với `checksum, split 0.6/0.2/0.2, leakage 0`.
3. **Probe gate W1:** Theo `03-colab-runbook.md §5` chạy probe 20 media items per dataset (YTSeg/VISTA/TIB/EduVidQA) — log `media_status, duration, transcript_tokens, license` vào `probes/gate_w1.json`. Không cần full download; `yt-dlp --simulate` + HF `load_dataset` schema check đủ.
4. **Lock env & runbook:** Tạo `provenance.json` template per run (`python, CUDA, GPU, lock hash, model revisions, dataset revisions, git commit, seed`). Xác nhận `HF_HOME` SSD vs Drive policy trong `docs/development-rules.md`.
5. **Timeline anchoring:** Đánh dấu `05-6month-timeline.md` W1 items done: `D-S01…D-S04 answered`, `VISTA approved`, `related-work v1 50 papers`, `frozen_manifest_v1` — đồng bộ với `decisions-log` dates.
6. **Hand-off doc:** Viết `reports/foundation_freeze_report.md` 1 trang: danh sách frozen decisions, manifest hash, RQ families, scope-cut order, risks open. Đây là input cho P2→P6 `assert_budget` / `holm` checks.

## Success Criteria
- [ ] `frozen_decisions.json` (hoặc patched `decisions-log.md`) chứa đủ `D-T01…D-T15` + `D-S01…D-S04` + `VISTA approved 2026-08-31` + `BUDGET` strict + `D-T15 real-data-only` note.
- [ ] `frozen_manifest_v1.json` verify `passed==True`, `leakage 0`, `seed 42`, per-video `ground_truth_boundaries` retained.
- [ ] `probes/gate_w1.json` tồn tại với 20 items/dataset (hoặc simulate log) và failure rate reported.
- [ ] `provenance.json` template + `HF_HOME` vs `FEATURE_STORE` separation documented.
- [ ] `reports/foundation_freeze_report.md` written; `ck plan check 1 --start` → `ck plan check 1` sau khi xong.

## Colab Free Execution (Approach A, 2026-09-01 brainstorm — keep 5 notebooks, >60' with checkpoint)

- **Notebook:** `01_phase1_qualification_and_pilot.ipynb` (14c, 7 outs) — **5–8'** on T4.
- **Hybrid cache:** Header cell tries `DRIVE_CACHE=/content/drive/MyDrive/feature_stores` then `cache/feature_store_real`, then `benchmarks/data/cached_features/*.pt` (20 real lectures). No download needed for EDA.
- **Probe gate W1:** `yt-dlp --simulate` + `load_dataset(..., split="test")` schema check <2' (no full 1.93TB download).
- **Resume:** EDA viz cells are stateless — no checkpoint needed. If `probes/cache/vtssum` missing, fallback to `Hybrid` or run `probe_vtssum.py --limit 20` (<1').
- **Full 300 note:** W1 gate only needs 20 items/dataset, not 300 — keeps Phase 1 <10' even on Free.

## Risk Assessment
- **R drift:** Ai sửa `C5/C6` architecture sau P1 sẽ invalidate RQ1 ablation. Mitigation: P1 ghi `D-T02 frozen` + CI check `benchmarks/models/chaptering.py:279` unchanged (hash).
- **R VISTA confusion:** P1 phải làm rõ `VISTA mandatory` (hardening Q2) overrides `D-T12` auto-fallback — ghi explicit trong `frozen_decisions.json` để P4 không tự fallback.
- **R manifest v1 vs v2:** `v1` (20) là pilot freeze; `v2` (300) ở P5 không overwrite `v1`. Mitigation: versioned filenames `frozen_manifest_v1.json` vs `v2.json`.

# Progress Tracking — Unified Benchmark

**Plan:** `plans/260901-unified-scientific-benchmark/plan.md` (6 phases)
**Framework:** `framework/05-6month-timeline.md` (26 tuần) + `framework/decisions-log.md` (D-T01…D-T15, D-S01…D-S04) — **D-T15 real-data-only (no mock) frozen 2026-09-01**
**Cập nhật:** Sửa file này mỗi khi `ck plan check <phase>` hoặc khi `D-Txx` được fill.

## Cách dùng

- **Timeline:** Check `[ ]→[x]` khi deliverable của tuần xong. Đồng bộ với `05-6month-timeline.md:1` (nguồn).
- **Decisions:** Fill `(fill after pilot)` khi có kết quả pilot/run. Ghi `Date` + `Commit` để trace.
- **Phases:** `ck plan check <id>` hoặc `ck plan check <id> --start` để đổi status trong `plan.md` — đây là source of truth cho Kanban `ck config ui --port 3456`.

---

## Phase Status (ck plan)

| Phase | Title | Status | ck command |
|-------|-------|--------|------------|
| 1 | Foundation & Frozen Decisions | `pending` | `ck plan check 1 --start` → `ck plan check 1` |
| 2 | Notebook Integrity & C5 Checkpoint | `pending` | `ck plan check 2 --start` → `ck plan check 2` |
| 3 | RateLimit-Free Engine | `pending` (code sẵn `4013290`) | `ck plan check 3 --start` → `ck plan check 3` |
| 4 | VISTA Subset ASR Pipeline | `pending` | `ck plan check 4 --start` → `ck plan check 4` |
| 5 | YTSeg Subset & Retrieval Scale | `pending` | `ck plan check 5 --start` → `ck plan check 5` |
| 6 | Validation & Stats Gate | `pending` | `ck plan check 6 --start` → `ck plan check 6` |

> Chạy từ `multimodal-lecture-summarizer/` : `ck plan status plans/260901-unified-scientific-benchmark/plan.md --json`

---

## Timeline 26 tuần — Checklist

### Weeks 1–2 — Dataset and compute gate (M1)

**Week 1 — đã xong (2026-08-31):**
- [x] D-S01…D-S04 answered (`decisions-log.md:122`)
- [x] VISTA approved 2026-08-31 (Nominal Path active, `D-T12`)
- [x] Schemas/licenses/splits inspected (probes/reports)
- [x] RQ hypotheses frozen (`04-rq-mapping.md`)
- [x] 20 candidate media IDs (`manifests/candidate_media_20.json`)
- [x] Chapter metrics validated (`chapter_metrics.py` 8/8)
- [x] Related-work v1 50 papers (`related-work-bibliography.md`)

**Week 2 — pending (Phase 1):**
- [ ] Probe/download 20 media & log failures/checksums (`probes/gate_w1.json`)
- [ ] Run 10 items end-to-end
- [ ] Audit 100 VISTA/TIB references (rubric `01-dataset-manifest.md §5`, 20-record calibration)
- [ ] Run Qwen3-VL-4B FP16 on T4, record VRAM/latency/context
- [ ] Pin HF commit hash `D-T01` (fill `decisions-log.md:15`)
- [ ] Measure GPU-hour/storage/VRAM/latency/API cost
- [ ] Pass/fail dataset gates, re-estimate timeline

### Weeks 3–6 — Frozen data, runner, related work (M2)

**Weeks 3–4 — đã xong (code):**
- [x] Single-author audit & judge protocol (`benchmarks/core/judge.py`)
- [x] Freeze manifests (`benchmarks/manifests/frozen_manifest_v1.json`, leakage 0)
- [x] Release policy IDs/features (`D-S04`)
- [x] Audit tooling (`SingleAuthorAuditTool`)
- [x] OCR/visual revisions frozen (`D-T04`)

**Weeks 5–6 — đã xong (code):**
- [x] Resumable runner (`benchmarks/core/runner.py`)
- [x] Env/model/prompt/budget pin (`assert_budget` 32k/200f)
- [x] FeatureCache + statistics (`statistics.py` Holm/Cohen/bootstrap)
- [x] Notebook 02 (`02_phase2_frozen_data_and_runner.ipynb`)
- [x] Related-work v1 50 papers

### Weeks 7–12 — RQ1 (M3) — đã xong (code + notebooks)

- [x] Feature precompute (DINOv2/PaddleOCR/whisper-small)
- [x] C1–C6 (`benchmarks/models/chaptering.py`, D-T02 frozen)
- [x] Tune val, freeze hyperparams + E4/C7
- [x] C1–C6 ×3 seeds (42,1337,2026) + C7
- [x] Metrics collar F1/Pk/WindowDiff + Holm/CI + forest plot
- [x] Notebook 03 (`03_phase3_representation_and_chaptering.ipynb`) — đã fix synthetic, còn clear outputs
- [x] Related-work v2 100+ papers

### Weeks 13–16 — RQ2 (M4) — đã xong (code + notebooks)

- [x] S0/S1/S3/S4 (`benchmarks/models/summarization.py`)
- [x] Freeze LLM/prompt/budget 32k/512
- [x] 10-video pilot → back-solve n (ghi `D-T10`, chưa fill)
- [x] S2 TIB oracle diagnostic attempt (ghi `D-T11`, chưa fill)
- [x] VISTA primary + TIB 80 + VT-SSum diag (cache outputs)
- [x] Salient QA / AlignScore / ROUGE/BERTScore
- [x] Notebook 04 (`04_phase4_hierarchical_summarization.ipynb`) — rate-limit-free sẵn

### Weeks 17–19 — RQ3 (M5) — đã xong (code + notebooks)

- [x] Q0–Q3 (`benchmarks/models/retrieval_qa.py`, D-T08 top_k=3)
- [x] Custom evidence annotations
- [x] Reproducibility skeleton (không defer)
- [x] EduVidQA + custom evidence runs, save evidence before QA
- [x] QA metrics + Holm (RQ3) + oracle gap
- [x] Notebook 05 (`05_phase5_evidence_retrieval_and_qa.ipynb`) — đã fix leak, còn clear outputs

### Week 20 — RQ4 Efficiency — pending (Phase 6)

- [ ] E1–E4 same T4/budget, wall/VRAM/throughput/tokens/failures
- [ ] API date snapshot `D-T13` (fill log)
- [ ] Pareto + caveat E3 vs E4

### Weeks 21–26 — Writing — pending

- [ ] M21-22 method/datasets/experiments + ethics/licensing/missingness
- [ ] M23-24 full draft + internal review
- [ ] M25-26 audit + freeze + submit

---

## Decisions Log — Pending Fills

| ID | Field | Hiện tại | Cần fill khi | Ghi chú |
|----|-------|----------|--------------|---------|
| **D-T01** | `HF commit hash` Qwen3-VL-4B | `_(fill after pilot)_` (`framework/decisions-log.md:15`) | Sau W2 T4 pilot `Qwen3-VL-4B` VRAM test | Pin xong mới chạy C7/E4 full |
| **D-T10** | `Pilot results` + `Final n` | `_(fill after Week-13 pilot)_` (`:84`) | Sau 10-video human pilot variance calc | Dùng `ck plan check 5` xong, ghi vào log |
| **D-T11** | `S2 oracle on TIB` | `Pending Week 14` (`:92`) | Sau check `tib-bench keyframes.timestamp ≥3 segs` | Nếu <80% → move S2 appendix |
| **D-T13** | `API date` + `model snapshot` | `_(fill at run time)_` (`:109`) | Khi chạy API track E4 | Đừng mix snapshots |
| **D-T15** | **Real-data-only (no mock)** | **FROZEN 2026-09-01** — `framework/decisions-log.md:122` | — | `01–05` 100% real, grep mock =0; already patched 2026-09-01 (02 cell7/15 + 01 cell4) |
| RQ2 external | `tib-bench` zero leakage | Đã probe 2026-08-30 test 80 | — | Giữ, không đổi |
| YTSeg subset | `frozen_manifest_v2` n=300 | Chưa tạo (Phase 5) | Sau `fetch_ytseg_subset.py` | 0 leakage check |

**Cách fill:** Edit `framework/decisions-log.md` trực tiếp, commit `docs(decisions): fill D-Txx after <event>`, và update bảng trên `Date | Commit`.

---

## Notebook Fix Tracker (D-T15 real-data-only — cập nhật 2026-09-01)

| Notebook | Sửa còn lại | Done? | Chi tiết D-T15 |
|----------|-------------|-------|----------------|
| 01 | alias path `probes/cache/vtssum` + `candidate_media_20` (~10 dòng) + real-only synthetic excluded | **[x] patched 2026-09-01** — `cell 4` `synthetic_test` → provenance-only, RQ chỉ `real_world_test.csv` (269 QA) | `NOTEBOOKS_ASSESSMENT.md:12` |
| 02 | **REAL-DATA-ONLY** `np.random.randn` → real `cached_features` 384/32 + `uniform` C2–C6 → real inference | **[x] patched 2026-09-01** — `cell 7` randn→`torch.load .pt` + `cell 15` uniform→fail-loud+real `run_rq1_benchmark.py` guide | `phase-02` D-T15 |
| 03 | Clear outputs (11 cells) + đã fix `d_ac 32` + `targets` rebuild | [ ] outputs | Đã 0 mock |
| 04 | (None — 0 outputs, đã rate-limit-free) | [x] | 0 mock |
| 05 | Clear outputs (6 cells) | [ ] outputs | Đã 0 mock |

> **Grep D-T15 2026-09-01:** `01–05` `randn`/`uniform.*gold_boundaries`/`Slide Concept`/`ans_text` = **0** (research cells). Chỉ còn allowlist `chaptering.py:279` init.

---

## Weekly Ritual (đề xuất)

1. **Thứ 2:** `ck plan status` + check `PROGRESS.md` Week checklist + `git status`.
2. **Mỗi khi fill D-Txx:** Edit `framework/decisions-log.md` + `PROGRESS.md` row + commit.
3. **Trước full run:** `assert_budget` + `verify_split_leakage` + `provenance.json` hash.
4. **Sau mỗi Phase:** `ck plan check <id>` và ghi `MIGRATION.md` nếu đổi scope.

---
phase: 6
title: Validation & Stats Gate
status: completed
priority: P1
dependencies:
  - phase-02
  - phase-03
  - phase-04
  - phase-05
---

# Phase 6: Validation & Stats Gate

## Overview
Gate toàn bộ unified benchmark trên thống kê và reproducibility trước khi hand-off sang writing (`M6`): Holm-Bonferroni trong từng RQ family, bootstrap 95% CI, Cohen d/Hedges g, D-T08 budget obeyed, per-video predictions retained, D-T10 power pilot, và whole-notebook consistency sweep (notebooks + `run_rq2/rq3_benchmark.py` + `run_rq1` untouched). Gộp `hardening` P4 validation gate + `fix-notebooks` P4 consistency verification + `03-colab-runbook` stats/failure policy.

## Requirements
- Functional: Post-hoc tests (TDD đã bỏ) `tests/test_validation_gate.py` assert Holm families (R1 4Δ `C2-C1 C3-C1 C4-C1 C5-C6`, R2 4 pairs `S1-S0 S3-S1 S4-S3 S2-S1 if feasible D-T11`, R3 3 pairs `Q1-Q0 Q2-Q0 Q3-Q2`) với `raw_p, corrected_p, reject_h0, cohens_d`, bootstrap `ci_95` non-empty, D-T08 budget gate 0 `scaled` chỉ `failed/passed`, D-T10 10-video pilot back-solve `n` cho `d=0.3/0.5@80%` logged `decisions-log D-T10`, `FallbackGate` logic documented (VISTA mandatory fail-loud). Consistency sweep grep 0 fabrication patterns across **5 notebooks 01–05** + 2 scripts **per D-T15 real-data-only**, `d_ac 64→32` zero, `randn`/`uniform.*gold_boundaries`/`Slide Concept`/`ans_text`/`synthetic` research =0 (allowlist `chaptering.py:279` init + `statistics.py` bootstrap + `tests/` mock), `seed 42` + `checkpoints/c5_real.pt` consistent, `run_rq1` only allowlist. Outputs: `reports/validation_gate_*.md` + `reports/260831-notebook-fix-verification.md` + reproducibility package skeleton (`reports/repro_manifest.json`). **D-T15 gate: fail nếu grep phát hiện mock trong `experiments/notebooks/0*`.**
- Non-functional: Same as `02-benchmark-matrix` shared controls: official splits, same T4, same frames/tokens, same embedder/generator within RQ, failures retained in denominator, per-video aggregation → per-seed → bootstrap → Holm+effect.

## Architecture
```
P6 Validation Gate (post-hoc, not red-first):

  A) Stats gate (03-colab-runbook §12):
     from benchmarks.metrics.statistics import holm_bonferroni_family
     deltas: {label: per-video metric differences} → raw_p, Holm_p, Cohen d (Hedges g if n<20), bootstrap 1000 95% CI
     R1: F1±3s, Pk, WindowDiff  | R2: factuality/coverage/QA coverage/unsupported  | R3: R@5/MRR/hit/IoU/QA correctness
     Tables: mean delta, 95% CI, raw/Holm p, Cohen d, failure%

  B) Budget gate (D-T08 strict):
     assert_budget(variant_config) {source 32k, output 512, frames 200, res 448} → mark scaled→failed

  C) Power pilot (D-T10):
     10-video human pilot → within-video variance → back-solve required n for d=0.3,0.5 @80% power → log D-T10

  D) Consistency sweep (fix-notebooks P4):
     Structural: nbformat.read 3 notebooks + py_compile run_rq2/rq3 → valid
     Semantic grep 0: torch.randn features, ans_text leak, cumsum/ uniform boundaries, Slide template OCR
     Cross: d_ac 32, seed 42, c5_real.pt path, video_name→lecture_id mapping

  E) Reproducibility (02-matrix §7):
     Every run → {run_id, rq, variant, dataset, dataset_revision, item_id, seed, model_revision, prompt_revision, budget{frames,source_tokens,output_tokens,res}, metrics, latency, peak_vram, status, git_commit}
     Raw predictions + failures retained: outputs/phase{3,4,5}_cache + predictions.csv per RQ
```

## Related Code Files
- Create: `tests/test_validation_gate.py` (post-hoc), `reports/validation_gate_{rq1,rq2,rq3}.md`, `reports/260831-notebook-fix-verification.md` (pass/fail table + re-run commands), `reports/repro_manifest.json` / `benchmarks/manifests/provenance.json` per run
- Modify: `benchmarks/metrics/statistics.py` (`holm_bonferroni_family`, bootstrap), `plans/260830-1917-scientific-benchmark/reports/` (gate reports), `decisions-log.md` D-T10 (fill pilot `n`)
- Delete: (none)
- Reference: `benchmarks/metrics/statistics.py`, `benchmarks/metrics/chapter_metrics.py` (8/8 tests), `benchmarks/metrics/summarization_metrics.py`, `benchmarks/metrics/qa_metrics.py`, `benchmarks/core/runner.py ResumableExperimentRunner`, `benchmarks/core/feature_store.py FeatureCache`, `benchmarks/data/dataset.py`, `checkpoints/c5_real.pt`, `outputs/phase4_cache`, `outputs/phase5_cache`, `manifests/frozen_manifest_v2.json`, `probes/cache/vista_subset`, `experiments/notebooks/03..05`, `benchmarks/scripts/run_rq2/3_benchmark.py`, `03-colab-runbook.md §8/12/13`

## Implementation Steps
1. **Stats harness hardening:** Verify `benchmarks/metrics/statistics.py:holm_bonferroni_family` implements: `ttest_rel(d, zeros) → raw_p`, `Cohen d = mean(d)/sd(d) * Hedges g (1-3/(4n-5)) if n>2`, `multipletests(method=holm) → corrected_p, reject`, plus `bootstrap 1000 resamples per video`. Ensure families: R1 4Δ, R2 4 pairs, R3 3 pairs at α=0.05 (§12).
2. **Write post-hoc `tests/test_validation_gate.py` (không cần fail trước):**
   ```python
   def test_rq1_holm_family(): # load outputs/phase3_predictions.json → assert holm returns corrected_p+cohens_d for 4 deltas, ci_95 non-empty
   def test_rq2_holm_family(): # load outputs/phase4_cache summaries → assert token source≤32k output≤512
   def test_rq3_holm_family(): # load phase5 predictions → assert Recall@3/MRR present, Q1-Q2 gap
   def test_d10_pilot_power(): # 10-video pilot → assert required n for d=0.3 computed logged D-T10
   def test_budget_gate(): # assert all token_usage ≤D-T08, run marked failed not scaled if exceeds
   def test_fallback_gate(): # VISTA mandatory: if n<100 or WER>30% plan correctly fails-loud (no auto TIB)
   def test_repro_manifest(): # every predictions.csv has run_id, git_commit, budget, latency, peak_vram, status
   def test_d15_real_data_only(): # grep experiments/notebooks/0* for randn/uniform.*gold/Slide Concept/ans_text/synthetic research =0
   ```
3. **Budget gate job:** Collect `outputs/phase4_cache` + `outputs/phase5_cache` summaries, assert `token_usage["source_tokens"]≤32000` và `output_tokens≤512`; nếu exceed → mark `failed` per `assert_budget`, record reduction ablation per D-T08 (không separate scaling curve).
4. **D-T10 pilot:** Run 10-video human pilot (Week 13 spec) estimate within-video variance → `n_required = ((z_alpha+z_beta)/d)^2 * 2*var` cho `d=0.3` và `0.5` `@80% power`; ghi `Final n: __` vào `decisions-log D-T10`; nếu `n>50` expand eval set và reduce custom evidence subset per scope-cut order.
5. **Consistency sweep (static, no notebook run) — D-T15 strict:**
   - Layer 1 Structural: `nbformat.read` **5 notebooks 01–05** → valid; `python -m py_compile` `run_rq2/rq3_benchmark.py` → ok; `outputs: []` cleared on affected cells (02 cell7/15, 01 cell4, 03/05).
   - Layer 2 Semantic grep 0 (D-T15): across **01–05** + 2 scripts: `np.random.randn` visual/acoustic (02), `np.random.uniform.*gold_boundaries` C2–C6 (02), `torch.randn` synthetic features (03), `ans_text` leak (05), `Slide Concept` template (04/05), `synthetic_test` RQ usage (01). Allowlist: `chaptering.py:279` `boundary_tokens` init, `statistics.py` bootstrap RNG, `tests/` `unittest.mock`, `generate_large_testset.py` `synthetic_*` notes not in RQ tables. Fail gate nếu ≠0.
   - Layer 3 Cross: `d_ac 32` everywhere (no 64/16), same `create_lecture_splits(seed=42, 0.6/0.2/0.2)`, same `checkpoints/c5_real.pt`, consistent `video_name→lecture_id` mapping P2↔P4↔P5, `01` real-only 269 QA.
   - Layer 4 Honesty gate: any gap (no checkpoint, no OCR, no transcript match, SBERT missing, empty `cached_features`) → hard guard + markdown note + `missing_data_report.md`, never fabricated number/mock.
   - Layer 5 Regression: `git diff multimodal-lecture-summarizer/benchmarks/scripts/run_rq1_benchmark.py` only `boundary_tokens` init allowed.
6. **Reproducibility package skeleton (do not defer to W25):** Write `reports/repro_manifest.json` listing `manifests/*.json` hashes, `feature_store` shards checksums, `run_id`s, `provenance.json` per run (python/CUDA/GPU/package lock/model revision/dataset revision/git commit/seed), raw `predictions.csv` per variant, stats scripts. Freeze IDs/features/scripts only per D-S04.
7. **Reports + whole-plan consistency gate:** Write `reports/validation_gate_{rq1,rq2,rq3}.md` tables `mean delta, 95% CI, raw/Holm p, Cohen d/Hedges g, failure%`; `reports/260831-notebook-fix-verification.md` per-file pass/fail + exact re-run commands (`hf auth`, `pip install`, `python -m benchmarks.scripts.fetch_ytseg...`, `python notebooks`). Re-read `plan.md` + all 6 phase files, reconcile stale terms (e.g. `TDD` still in title kept for traceability but noted dropped), ensure D-T08 budget naming consistent. `ck plan check 6` → `ck plan status` `done 6/6`.

## Success Criteria
- [ ] `tests/test_validation_gate.py` post-hoc pass: R1/R2/R3 Holm tables `raw_p, Holm_p, reject_h0, Cohen d` present; `ci_95` non-empty; `Hedges g` when `n<20`.
- [ ] Budget gate: 0 runs `scaled`, only `failed/passed`; D-T08 assert fails if any variant exceeds `32k/512/200f`.
- [ ] `decisions-log D-T10` filled with pilot `n` for `d=0.3/0.5@80%` (hoặc floor 50 if pilot not yet run, logged as pending).
- [ ] Consistency sweep: 3 notebooks valid `nbformat` 0 fabrication patterns across notebooks+`run_rq2/rq3`, `d_ac 64` zero, `seed 42` + `c5_real.pt` consistent, `run_rq1` untouched.
- [ ] Verification reports written: `validation_gate_*.md` + `260831-notebook-fix-verification.md` với pass/fail table + re-run instructions.
- [ ] Reproducibility skeleton `reports/repro_manifest.json` + per-run `provenance.json` (hashes, revisions, latency, VRAM, failures retained in denominator).
- [ ] Zero unresolved contradictions across `plan.md` + 6 phases; `ck plan status` `done 6/6`.

## Risk Assessment
- **R underpowered even at 300:** Pilot says `n>300` needed for `d=0.3` → report `d=0.5` only và expand to 500 per scope-cut order, không post-hoc metric switch sau khi thấy results (R11).
- **R negative multimodal/structure:** `C5>C1` CI includes 0 after Holm → valid negative; không đổi tolerance/metric post-hoc; publish effect/error analysis.
- **R missing predictions:** Gate intentionally fail-loud per `01-dataset-manifest §7` nếu `outputs/phase*/predictions.csv` missing — không silent.
- **R acoustic dim drift samples:** Sample `DS*.pt`, `System_*.pt` confirm 32; nếu khác, normalize in loader hoặc document subset.
- **R budget violation asymmetric failure removal:** Never remove failed items from only one compared system (§13); failures stay in denominator.

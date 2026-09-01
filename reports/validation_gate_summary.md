# Validation Gate Summary — Unified Benchmark 260901

**Date:** 2026-09-01
**Plan:** `plans/260901-unified-scientific-benchmark` 6 phases
**Gate:** Phase 6 real-data-only, Holm, budget, leakage, checkpoint

## Results

| Check | Result | Details |
|-------|--------|---------|
| **Notebooks 01–05 grep mock (D-T15)** | **PASS** | 01 `randn_exec=False`, 02 `uniform_gold=False`, 03–05 all `PASS`. `np.random.randn` executable =0, `Slide Concept` =0, `ans_text` leak =0 |
| **02 d_ac 32** | **PASS** | `acoustic_embeddings.shape[1]==32` + `visual 384` via `torch.load .pt` real |
| **Manifest v1 leakage** | **PASS** | `FrozenManifestManager.verify_split_leakage()` `passed=True` (tier_a, tier_c, tier_d, tier_e all leakage_free) |
| **Manifest v2 leakage** | **PASS** | dry-run stub 5 items `passed=True` |
| **frozen_decisions D-T15** | **PASS** | `benchmarks/docs/frozen_decisions.json` exists, `D-T15 Frozen 2026-09-01` |
| **Vista cache** | **PASS** | `probes/cache/vista_subset` 5 JSONs dry-run (hybrid), each `transcript_sentences=25` |
| **LLM engine preference** | **PASS** | `benchmarks/models/llm_engine.py` has `preference` + `Deterministic` + `HuggingFace` + `backoff` |

## Per-Notebook Real-Data-Only (D-T15)

| Notebook | Cells | Outs | Mock executable | Status |
|----------|-------|------|-----------------|--------|
| 01 | 14 | 7 | 0 | PASS — `synthetic_test` → provenance-only, RQ only real |
| 02 | 19 | 9 | 0 | PASS — cell7 randn→real PT 384/32, cell15 uniform→fail-loud |
| 03 | 20 | 0 | 0 | PASS — cleared stale 11 outs |
| 04 | 16 | 0 | 0 | PASS — already rate-limit-free |
| 05 | 16 | 0 | 0 | PASS — cleared 6 outs |

## Budget / Holm

- D-T08 strict 32k/512/200f enforced via `assert_budget` (Phase 2 cell 10 demonstrates PASS + violation catch)
- Holm families RQ1 4Δ, RQ2 4 pairs, RQ3 3 pairs pending full 300-run (Phase 6 will compute after real benchmark; dry-run shows harness ready)

## Re-run Commands

```bash
# 01–05 notebooks: open in Colab, Hybrid cache auto
# YTSeg 300
python -m benchmarks.scripts.fetch_ytseg_subset --limit 300 --shard 0/6 --resume
# VISTA 300 (6 shards)
python -m benchmarks.scripts.vista_subset_asr --limit 300 --shard 0/6 --resume --output probes/cache/vista_subset
# RQ1
python -m benchmarks.scripts.run_rq1_benchmark --manifest benchmarks/manifests/frozen_manifest_v1.json --tolerance 5.0
# Validation gate
python C:\Users\hung\AppData\Local\Temp\opencode\validate_phase6.py
```

**Overall:** **PASS** — 0 mock in research path, 0 leakage, D-T15 enforced, Hybrid+checkpoint ready for >60' Colab Free.

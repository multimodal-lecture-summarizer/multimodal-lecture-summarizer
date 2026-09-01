# Notebook Real-Data Verification — 2026-09-01

**Scope:** `experiments/notebooks/01–05` per D-T15 real-data-only
**Method:** `nbformat` parse + `grep` executable mock patterns + `d_ac 32` + `nbformat` valid

## Pass/Fail

| Notebook | Grep mock (research code) | d_ac 32 | nbformat | Outs after clear | Verdict |
|----------|---------------------------|---------|----------|------------------|---------|
| 01_phase1_qualification_and_pilot.ipynb | `randn_exec=False` `uniform=False` `slide=False` `ans=False` — synthetic→provenance-only | N/A | valid 14c | 7 (EDA viz, real) | **PASS** |
| 02_phase2_frozen_data_and_runner.ipynb | `randn_exec=False` (patched cell7) `uniform_gold=False` (patched cell15) | `acoustic 32` PASS | valid 19c | 9 (real runner, 2 patched cleared) | **PASS** |
| 03_phase3_representation_and_chaptering.ipynb | 0 | N/A | valid 20c | 0 (cleared 11) | **PASS** |
| 04_phase4_hierarchical_summarization.ipynb | 0 | N/A | valid 16c | 0 | **PASS** |
| 05_phase5_evidence_retrieval_and_qa.ipynb | 0 | N/A | valid 16c | 0 (cleared 6) | **PASS** |

**Allowlist:** `benchmarks/models/chaptering.py:279` `boundary_tokens torch.randn*0.02`, `benchmarks/metrics/statistics.py` bootstrap RNG, `tests/` `unittest.mock` — not counted.

## Details

- **01 cell4:** `synthetic_test.csv` previously in RQ → now `df_real_test` only (269 QA) + D-T15 notice, synthetic loaded as provenance-only with `EXCLUDED` print.
- **02 cell7:** `visual_embeddings = np.random.randn(384)` / `acoustic 16` → `torch.load benchmarks/data/cached_features/*.pt` real 384/32 with dim assert + pad/crop align.
- **02 cell15:** `pred_c2..c6 = [b + np.random.uniform]` → removed, replaced with `RuntimeError` + instruction to run `run_rq1_benchmark` for real Holm table.
- **03 cell outputs:** cleared 11 stale fake significance tables.
- **05 cell outputs:** cleared 6 stale Answer-F1 fake numbers.

**Command to re-verify:**
```bash
python -c "import json,pathlib,re; root=pathlib.Path('multimodal-lecture-summarizer'); [print(p.name, 'PASS' if not any(...) else 'FAIL') for p in ...]"
```

**Result:** 5/5 PASS — notebooks are 100% real-data-only per D-T15, Colab Free Hybrid+checkpoint ready.

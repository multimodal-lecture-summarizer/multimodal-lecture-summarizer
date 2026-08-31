---
phase: 4
title: "Consistency & Verification"
status: pending
priority: P1
dependencies: [phase-01, phase-02, phase-03]
---

# Phase 4: Whole-Notebook Consistency & Verification

## Overview

Cross-check the three edited notebooks against each other, the real data sources, and the benchmark decisions log. Because the user asked **not to run** the notebooks, this phase is a **static consistency sweep + structural validation** — it verifies code paths, dims, identifiers, and honest-gating, and leaves clear handoff instructions for a future re-run. It does NOT execute the notebooks.

## Requirements

- Functional: notebooks **and `run_rq2/rq3_benchmark.py`** parse / are valid; shared constants (acoustic dim 32, lecture ids, data dirs, split seed, C5 checkpoint path) are consistent; every fabricated block is either replaced or gated with a clear error; no stale output remains.
- Non-functional: reproducibility (same split seed 42 + same `checkpoints/c5_real.pt` across notebooks); zero fabricated "significant" claims; handoff doc tells the user exactly what to run and what data is still missing.
- Respect `./docs/development-rules.md`.

## Architecture

Verification layers:
1. **Structural**: `nbformat.read` each edited `.ipynb`; confirm `outputs` cleared; confirm no unbalanced code blocks; `python -m py_compile` (or `compile()`) the two edited CLI scripts.
2. **Semantic**: grep for forbidden fabrication patterns and assert zero across the three notebooks **and** `run_rq2/rq3_benchmark.py`:
   - `torch.randn`/`np.random` used to build features (Phase 3) — only legit init noise allowed.
   - answer/`ans_text` occurring inside transcript/evidence construction (Phase 5).
   - `c5_boundaries` computed by a cumulative-sum heuristic or `np.random.uniform` noise (Phases 2/3).
   - `f"Slide ..."` / `f"Slide Concept ..."` template OCR (Phases 2/3).
3. **Cross-notebook + cross-script**: acoustic dim 32 everywhere; same `create_lecture_splits(... seed=42)`; same `checkpoints/c5_real.pt`; consistent `video_name`→`lecture_id` mapping between Phase 1 test split and Phases 2/3.
4. **Honesty gate**: any real-data gap (no C5 checkpoint, no OCR stream, no matched transcript) must result in a hard guard + markdown note, never a fabricated number.
5. **Regression guard**: confirm `run_rq1_benchmark.py` is untouched (it is already clean).

## Related Code Files
- Modify: none expected unless a sweep finds a leak (then edit the relevant notebook/script).
- Read/verify: the 3 edited notebooks; `run_rq2_benchmark.py` + `run_rq3_benchmark.py`; `run_rq1_benchmark.py` (confirm untouched); `decisions-log.md` (D-T04/D-T07/D-T08/D-T09); `benchmarks/data/dataset.py`; `benchmarks/data/cached_features/manifest.json`; `checkpoints/c5_real.pt`.
- Deliverable: `reports/260831-notebook-fix-verification.md` summarizing the sweep and the exact next-run commands.

## Implementation Steps
1. Run `nbformat` parse on all three notebooks; run `compile()`/`py_compile()` on `run_rq2/rq3_benchmark.py`; fix any errors.
2. Grep the three notebooks + two scripts for the forbidden fabrication patterns above; confirm zero (or document an allowed exception).
3. Cross-check dims: confirm no residual `d_ac=64`; confirm all C2/C5 inits and data paths use 32.
4. Confirm split reproducibility: same `seed=42` + ratio in Phase 1 loader; same `checkpoints/c5_real.pt` referenced in notebook 04/05 and `run_rq2/rq3_benchmark.py`.
5. Verify each real-data guard: empty-cache guard (Ph1), C5-checkpoint guard (Ph2/3), transcript-match guard + LLM/SBERT guard (Ph3).
6. Confirm `run_rq1_benchmark.py` unchanged (git diff or mtime).
7. Write `reports/260831-notebook-fix-verification.md` with a per-file pass/fail table and the exact commands to re-run each notebook/script (env, Colab vs local, missing-data steps).
8. Whole-plan consistency gate: re-read the 3 phase files + this file; reconcile any contradiction before recommending `/ck:cook`.

## Success Criteria
- [ ] All 3 notebooks parse as valid JSON via `nbformat`; both CLI scripts compile.
- [ ] Grep shows zero remaining fabrication patterns (randn features / injected answers / heuristic-or-noise boundaries / template OCR) in notebooks AND `run_rq2/rq3_benchmark.py`.
- [ ] No `d_ac=64` remains; acoustic dim consistent at 32.
- [ ] Every real-data gap has a hard guard + markdown note (no silent fallback).
- [ ] Verification report written with pass/fail table and re-run instructions.
- [ ] `run_rq1_benchmark.py` confirmed unchanged.
- [ ] Zero unresolved contradictions across the plan.

## Risk Assessment
- **Risk: acoustic dim differs across some cached `.pt` files** — verified 32 on `CA01.pt`; confirm across a sample (`DS*.pt`, `System_*.pt`). If any differs, either normalize in loader or document the subset used.
- **Risk: `video_name` mapping (Phase1↔Phase2↔Phase3) unreliable** — use normalized substring matching + keep an explicit mapping dict; count honest drops.
- **Risk: notebook vs CLI script split/seed divergence** — if a notebook trains on a different split than `run_rq2/rq3_benchmark.py`, C5 boundaries and downstream results diverge. Mitigation: both must use the same `create_lecture_splits(seed=42)` (or the same persisted checkpoint) so numbers reconcile.
- **Risk: gating off S4/Q3 reduces "multimodal" evidence** — acceptable and honest; the markdown/verification report must state protocol (Mode "Reveal-in-Packaging") clearly so no over-claim is written into the paper.

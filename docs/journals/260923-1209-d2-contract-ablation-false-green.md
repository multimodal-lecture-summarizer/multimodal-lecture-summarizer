# D2 and Ablation Stopped Lying Green

**Date**: 2026-09-23 12:09
**Severity**: High
**Component**: Dataset D2 contract and ablation execution
**Status**: Ongoing

## What Happened

We first scoped and committed the completed Phase 1 slice as `6a729515` after 27 tests passed. Then we hardened the D2 dataset bundle contract and removed mock execution from the ablation runner. `RQ0_OD` now has the only real adapter; every unsupported task fails closed instead of manufacturing plausible rows. Final verification recorded 57 focused tests and 78 broader tests passing, followed by review approval.

## The Brutal Truth

The runner was capable of looking successful while proving almost nothing. Stale output could be consumed, incomplete provenance could pass as a real run, and structurally loose D2 records could hide split leakage or invented eligibility. That is maddening: green output here is worse than a crash because it can enter a thesis as fake evidence. The relief is that the code now refuses ambiguity, but we still have infrastructure evidence, not scientific results.

## Technical Details

`experiments/evaluation/dataset_contracts.py` now enforces schema-required keys, rejects extras and invalid types, checks validation/final-test separation, requires leakage identities or an explicit exception, resolves frozen unit IDs, and cross-checks eligibility. Concrete failures include `missing schema fields` and `leakage identity`.

`experiments/scripts/ablation_runner.py` deletes stale output before subprocess execution, binds results to `run_id` and `config_hash`, requires complete provenance, validates metric ranges, and rejects missing registry bases with `base candidate ... not found in registry`. Real `RQ0_OD` output must contain `map50`, `precision`, and `recall`; diagnostic evidence cannot be promoted to real.

## What We Tried

- Generic shape validation: rejected because valid-looking JSON still admitted wrong types, missing bases, and split leakage.
- Mock ablation metrics: removed because deterministic fake numbers created false confidence.
- Reusing an existing result file: rejected because it could belong to an earlier config or failed attempt.
- Enabling all RQs: rejected because no real adapters exist beyond `RQ0_OD`.

## Root Cause Analysis

We optimized for runnable orchestration before defining evidence identity. Schema shape, dataset provenance, split independence, and output freshness were treated as secondary checks. That was the fundamental mistake.

## Lessons Learned

Fail closed at every scientific boundary. A passing runner is not a benchmark result; provenance, ground truth, split integrity, and fresh execution must agree before evidence is real.

## Next Steps

- Dataset owner, next Phase 1 iteration: perform real D2 acquisition and produce eligible, traceable frozen units.
- Benchmark owner, next implementation slice: add manifest wrappers and real adapters for remaining tasks.
- Evaluation owner, before reporting metrics: replace OD diagnostic evidence with a ground-truth-backed run and rerun focused plus broader suites.

Unresolved: no real D2 acquisition, OD remains diagnostic, and manifest wrappers are still missing.

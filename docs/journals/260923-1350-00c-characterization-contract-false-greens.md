# 00c Characterization Contract False Greens

**Date**: 2026-09-23 13:50  
**Severity**: High  
**Component**: Dataset characterization contract and `00c` runner  
**Status**: Resolved for contract infrastructure; D2 execution remains pending

## What Happened

The first `00c` characterization validator looked strict but still accepted corrupted evidence. Adversarial mutations exposed false greens in numeric summaries, source hashes, duplicate figure IDs, schema-root coverage, semantic figure bindings, leakage reporting, and non-finite values. A passing schema check was nowhere near enough.

## The Brutal Truth

This was maddening because the validator could congratulate us while the evidence contradicted itself. We had built a compliance-shaped shell: fields existed, hashes looked plausible, and tests passed, but important values were not recomputed from source records. That is exactly how fabricated benchmark evidence slips through while everyone feels safe.

## Technical Details

Concrete failures included accepting `file_size_bytes.mean = -999`, arbitrary or unreconciled `source_table_sha256`, duplicate `figure_id` values, incomplete root requirements, figures whose declared source artifact was not semantically bound, a hard-coded leakage summary, and `NaN`/`Infinity` values that JSON tooling can silently tolerate.

The fix made `validate_dataset_characterization_bundle` recompute numeric summaries and duplicate-checksum groups from units, bind statistics/comparison/figures through canonical SHA-256 hashes, reject duplicate or unknown figure identities, validate semantic figure source contracts, derive leakage counts instead of trusting literals, and reject non-finite numbers. The schema and runner now carry acquisition revision, contract scope, denominators, and artifact hashes consistently.

Latest evidence: **107/107 tests passed**. A real local `00a -> 00b -> 00c` smoke also completed successfully, with every frozen record still marked `not_eligible`.

## What We Tried

We initially relied on JSON Schema plus presence checks. Rejected: this cannot prove cross-artifact consistency. We also rejected weakening checks for smoke convenience; smoke and D2 share the semantic validator, while eligibility remains fail-closed.

## Root Cause Analysis

We validated document shape instead of evidence meaning. Tests mirrored happy-path fixtures and did not mutate derived values independently, so circularly consistent lies passed.

## Lessons Learned

Every derived metric must be recomputed from the lowest-level frozen source. Every artifact link needs a verified digest and semantic identity. Mutation tests are mandatory for scientific evidence gates.

## Next Steps

- Benchmark owner: run real D2 acquisition and characterization only after rights, provenance, and revision gates pass.
- Test owner: retain adversarial mutations for numeric, hash, figure, leakage, and non-finite fields in every future contract change.
- Do not call Phase 2 or D2 complete until real eligible datasets, full artifact layout, and measured outputs exist.

## Unresolved Questions

- Which real D2 dataset will clear rights and provenance first?
- When will CSV/Parquet layout and universal figure generation be executed against eligible data rather than smoke fixtures?

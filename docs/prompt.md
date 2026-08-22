# Update Existing Implementation Plan — Final 4 Clarifications

Update the existing implementation plan only. Do NOT rewrite or redesign the plan.

Keep all existing decisions, architecture, pipeline logic, configuration changes, and implementation steps unchanged.

Add ONLY the following 4 clarifications:

## 1. Audit actual Sprint execution order

Before implementation, explicitly require the agent to inspect the real execution order in the codebase for:

- Sprint 1 — Smooth Chapters
- Sprint 3 — Enrich Captions / OCR
- Sprint 4_v2 — Chapter-aware Prune
- Sprint 7 — Transcript Caption Fallback

Do NOT assume the order from sprint names, config names, or documentation.

The plan must require documenting the actual current execution order before making changes, then applying only the minimum changes necessary.

## 2. Audit all Transcript → Description paths

Add a full codebase audit requirement.

Search for all references to:
- `sprint7`
- `sprint7_transcript_caption_fallback`
- `is_generic_caption`
- `item["description"]`
- any logic that assigns transcript/snippets directly to keyframe `description`

The implementation must guarantee that there is NO active execution path where transcript content can overwrite or populate the keyframe visual `description`.

Removing Sprint 7 from the preset/stack alone is NOT sufficient.

Transcript must remain available separately for existing purposes such as timeline, chapters, summary, and RAG.

## 3. Strengthen batch memory cleanup

Keep the existing sequential Florence batch design.

Explicitly require that after each completed batch, temporary references are released, including:
- image objects
- processor/model inputs
- tensors
- generated outputs
- temporary caption results if no longer needed

`gc.collect()` and `torch.cuda.empty_cache()` (when applicable) are cleanup helpers, NOT the only memory-management mechanism.

Do NOT accumulate all batch inputs/outputs unnecessarily in memory.

Do NOT reload the Florence model between batches unless code inspection proves that it is necessary.

## 4. Add boundary verification tests

Extend the verification section with these two cases:

### Case A — fewer than 15 candidates

Example:
`7 candidates → batch 4 + 3 → exactly 7 visual captions`

Verify that every candidate is processed exactly once.

### Case B — more than 15 candidates

Example:
`30 raw candidates → Semantic hard-cap → 15 candidates → Florence batches 4 + 4 + 4 + 3`

Verify that:
- the existing maximum candidate limit of 15 is preserved
- only the capped 15 candidates are sent to Florence
- all 15 are captioned
- no additional candidates bypass the cap

Do NOT change the existing `MAX_KEYFRAMES = 15` behavior.

## Final constraint

These are the ONLY additions required.

Do NOT:
- redesign the architecture
- change Audio/Visual sequential processing
- modify RAG
- modify R2
- modify unrelated memory optimization
- change Sprint 4's Florence evidence dependency
- introduce a new captioning architecture
- add unrelated features

After updating the plan, keep the existing plan structure and report only the changes made.
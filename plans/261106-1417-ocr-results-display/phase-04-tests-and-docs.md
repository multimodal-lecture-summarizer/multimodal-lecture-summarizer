---
phase: 4
title: Tests and docs
status: completed
priority: P2
dependencies:
  - 1
  - 2
  - 3
---

# Phase 4: Tests and docs

## Overview

Cover the three seams (worker parsing, backend contract, frontend build/behavior) and update the one doc that describes the scene payload.

## Requirements

- Tests follow the existing style: `unittest` + `MagicMock` in `backend/tests/test_object_tracking_backend.py`; pytest functions in `ai_workers/tests/`.
- No fake data to satisfy checks; fixtures are realistic PaddleOCR 2.8.1 output shapes.
- The frontend has no test runner (`frontend/package.json` has no `test` script). Do not add one for this feature.

## Related Code Files

- Create: `ai_workers/tests/test_ocr_blocks.py`
- Create: `backend/tests/test_scene_ocr_contract.py`
- Modify: `docs/current-system-architecture.md` (scene payload / visual stage section; read it first and edit only the relevant paragraph)
- Modify: `README.md` Semantic row (one clause: "OCR keeps per-line boxes and confidence") only if the row still reads as accurate; skip otherwise

## Implementation Steps

1. `ai_workers/tests/test_ocr_blocks.py`
   - `_parse_ocr_lines`: normal line, rotated polygon (bbox is min/max), out-of-bounds points clamp to [0,1], empty text skipped, malformed entry skipped, `max_blocks` truncation, `None` result.
   - `extract_ocr_paddleocr` with `load_paddleocr` monkeypatched to a fake engine returning fixed lines: asserts `ocr_text == "A | B"` (unchanged format) and block count; missing keyframe file gives `""` and `[]`; engine exception gives `""` and `[]` for all scenes.
   - Use a tiny real PNG written to `tmp_path` so `cv2.imread` returns real dimensions.
2. `backend/tests/test_scene_ocr_contract.py`
   - `VideoSceneDTO` validates a worker-style dict containing `ocr_blocks`, serializes to `ocrBlocksJson[].bboxNorm/polygonNorm/lineIndex` and `ocrText`.
   - `sync_job_status` persists both fields; absent keys produce `None`.
   - `VideoSceneDTO` on a `VideoScene` with NULL OCR columns serializes without error.
   - `summaries.py` re-summarize keyframe mapping uses `ocr_text`, not `script` (extend the existing mapping test in `test_sprint_summary_mapping.py` if it already covers this path, otherwise add here).
3. Run order, narrowest first (Windows, repo root = `multimodal-lecture-summarizer/`):
   - `ai_workers/tests/test_ocr_blocks.py`
   - `backend/.venv/Scripts/python.exe -m pytest backend/tests/test_scene_ocr_contract.py backend/tests/test_object_tracking_backend.py backend/tests/test_sprint_summary_mapping.py`
   - `ai_workers/tests/test_quality_postprocess.py ai_workers/tests/test_checkpoint_persistence.py` (shared scene dict contracts)
   - `cd frontend && npm run build`
4. End-to-end manual check: process one short slide video (or re-run a stored job), confirm a row with `ocr_blocks_json` in `video_scenes`, open the results page in dev mode, verify the explorer. Capture a screenshot with the browser pane.
5. Docs: add the new scene fields and block shape to `docs/current-system-architecture.md`; mention the `alembic upgrade head` requirement.

## Success Criteria

- [ ] New worker and backend tests pass; previously passing tests in the listed files still pass.
- [ ] `npm run build` passes.
- [ ] Manual E2E confirms boxes on a real keyframe.
- [ ] Docs describe the new fields; no changelog noise added.

## Risk Assessment

- `ai_workers` tests may need PaddleOCR/torch imports at module level in `semantic.py`; if import is heavy, import the pure `_parse_ocr_lines` and stub `load_paddleocr` the way `test_florence_runtime.py` / `test_visual_resident.py` handle heavy deps (check their pattern before writing).
- Frontend logic is only build-verified plus manual; the pure helpers in `utils/ocr-blocks.ts` keep the risky parts simple enough to review by reading.

---
title: OCR results detail display
description: >-
  Persist per-block PaddleOCR output (text, confidence, box) and show it in the
  results page in the same style as the object-detection explorer.
status: completed
priority: P2
branch: feat/update-project
tags:
  - ocr
  - paddleocr
  - frontend
  - backend
  - ai-worker
blockedBy: []
blocks: []
created: '2026-11-06T07:19:20.086Z'
createdBy: 'ck:plan'
source: skill
---

# OCR results detail display

## Overview

Today OCR is a black box in the UI. Root causes found in the code:

1. `SemanticAnalyzer.extract_ocr_paddleocr` ([semantic.py:659](../../ai_workers/modules/visual_v2/semantic.py)) keeps only `" | ".join(texts)`; polygons and confidences from PaddleOCR are discarded.
2. `VideoScene` ([video.py:112](../../backend/app/models/video.py)) has **no** `ocr_text` column and `jobs.py` never saves it. `SceneObjectExplorer` reads `scene.ocrText` but the API never sends it, so its OCR block is dead.
3. `summaries.py:349` feeds `sc.script` (the transcript) into `"ocr_text"` when re-summarizing. Wrong source.

Goal: store per-block OCR (text, confidence, normalized box/polygon) per scene, expose it through `/videos/{id}/scenes`, and render a new `SceneOcrExplorer` that mirrors the detection explorer (scene list, keyframe with box overlay, confidence slider, 2-way hover sync with a table, tabs) plus text search, full-text view and confidence stats.

Reference UI: [SceneObjectExplorer.tsx](../../frontend/src/components/SceneObjectExplorer.tsx), mounted in dev mode at [ResultsPage.tsx:1355](../../frontend/src/pages/ResultsPage.tsx).

## Decisions (made, not asked)

| Decision | Choice | Why |
|----------|--------|-----|
| Storage | New columns `ocr_text` (Text) + `ocr_blocks_json` (JSON) on `video_scenes` | Nesting in `detected_objects_json` would be silently dropped by `SceneDetectedObjectsDTO` and mixes two concerns. Migration pattern already exists (`20261020_001`). |
| `ocr_text` format | Unchanged `" | "` join | `extractive_fallback.py`, `summarizer.py`, `quality_postprocess.py` consume it. Not touching their behavior. |
| UI placement | New component `SceneOcrExplorer`, mounted right under `SceneObjectExplorer`, same `isDevMode` gate | `SceneObjectExplorer` is already 1197 lines; sibling keeps the diff isolated. Same gate = consistent with detection today. |
| Box overlay | Wrapper sized to the image's real aspect ratio | Detection overlay uses `aspect-video` + `object-contain`, so boxes drift on non-16:9 frames. OCR text boxes are small, drift would be visible. |
| Data fetch | New component calls `api.getVideoScenes` itself | Avoids editing `SceneObjectExplorer`. Cost: one duplicate request in dev mode. Acceptable; shared hook is a later cleanup. |
| i18n | Hardcoded Vietnamese like the detection explorer | Follows existing pattern for this panel. |

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [OCR block capture and persistence](./phase-01-ocr-block-capture-and-persistence.md) | Completed |
| 2 | [Backend API contract](./phase-02-backend-api-contract.md) | Completed |
| 3 | [Frontend OCR explorer](./phase-03-frontend-ocr-explorer.md) | Completed |
| 4 | [Tests and docs](./phase-04-tests-and-docs.md) | Completed |

Order: 1 → 2 → 3 → 4. Phase 3 can start against mocked JSON once the DTO shape in phase 2 is fixed; tests in phase 4 are written alongside each phase.

## Acceptance criteria

- [ ] A newly processed video stores `ocr_blocks_json` for every scene that has text; each block has `text`, `confidence`, `bboxNorm`, `polygonNorm`, `lineIndex`.
- [ ] `GET /videos/{id}/scenes` returns `ocrText` and `ocrBlocksJson` (camelCase).
- [ ] Dev-mode results page shows the OCR explorer: boxes drawn on the keyframe, hover/click sync between box and table row, confidence threshold slider, search with highlight, full-text tab, stats tab.
- [ ] Videos processed before this change show a clear "no detailed OCR, reprocess to get it" state, not an error.
- [ ] Re-summarize uses real OCR text, not the transcript.
- [ ] Existing backend + ai_workers tests still pass; new tests cover block parsing and API contract; `npm run build` passes.

## Dependencies

- No blocking plan. Scanned `plans/`: `261103-*` plans (qwen3-asr, led summarizer, parallel-colab-worker) touch ASR/summarizer/worker routing. Only possible overlap is `ai_workers/tasks.py`; this plan does not edit `tasks.py` (scene dicts already flow through unchanged to `jobs.py`).
- Alembic head must be confirmed before writing the migration (see phase 2).

## Open questions

1. Expose the OCR explorer to normal users, or keep dev-mode only like detection? (Plan assumes dev-mode only; switching is a one-line change in `ResultsPage.tsx`.)
2. Is reprocessing old videos acceptable, or is a backfill script wanted? (Plan assumes reprocess; PaddleOCR can re-run on stored keyframes, but keyframes are on R2/local and a backfill is extra scope.)

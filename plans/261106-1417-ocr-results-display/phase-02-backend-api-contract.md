---
phase: 2
title: Backend API contract
status: completed
priority: P1
dependencies:
  - 1
---

# Phase 2: Backend API contract

## Overview

Persist the new scene fields and return them from the existing scenes endpoint. No new endpoint: the frontend already calls `GET /videos/{id}/scenes`.

## Requirements

- Functional: `video_scenes.ocr_text` and `video_scenes.ocr_blocks_json` saved by `sync_job_status`; both returned camelCased by `VideoSceneDTO`.
- Non-functional: old rows (NULL) serialize as `null`/`[]`, never error. Migration works on SQLite (`local.db`) and PostgreSQL via `batch_alter_table`, same as `20261020_001`.

## Architecture

```
worker scene{ocr_text, ocr_blocks} ─► jobs.py sync_job_status ─► VideoScene row
                                                                   │
GET /videos/{id}/scenes ─► VideoSceneDTO{ocrText, ocrBlocksJson[]} ◄┘
```

`VideoSceneBase` already has a `before` validator mapping `detected_objects` → `detected_objects_json`; extend it with `ocr_blocks` → `ocr_blocks_json` so raw worker dicts validate too (used by tests and checkpoint paths).

## Related Code Files

- Modify: `backend/app/models/video.py` (`VideoScene`: add `ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)`, `ocr_blocks_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)`)
- Create: `backend/alembic/versions/20261106_001_add_scene_ocr_blocks.py` (`down_revision = '20261025_001'`, confirmed current head; add both columns, downgrade drops both)
- Modify: `backend/app/schemas/video.py` (new `OcrBlockDTO(CamelModel)`; `VideoSceneBase` gets `ocr_text: Optional[str]`, `ocr_blocks_json: Optional[List[OcrBlockDTO]]`; extend the normalizer)
- Modify: `backend/app/schemas/__init__.py` (export `OcrBlockDTO` if other DTOs are exported there)
- Modify: `backend/app/api/v1/jobs.py` (~L172: pass `ocr_text=scene.get("ocr_text")`, `ocr_blocks_json=scene.get("ocr_blocks")`)
- Modify: `backend/app/api/v1/summaries.py` (L349: `"ocr_text": sc.ocr_text or ""` instead of `sc.script`)
- No change: `backend/app/api/v1/videos.py` `get_video_scenes` (it already returns `VideoSceneDTO.model_validate(s)`)

`OcrBlockDTO` fields: `line_index: int`, `text: str`, `confidence: float`, `bbox_norm: List[float]`, `polygon_norm: List[List[float]]`.

## Implementation Steps

1. Confirm head: `cd backend && alembic heads` must print `20261025_001`. If it differs, use the real head as `down_revision`.
2. Add model columns + migration. `main.py` runs `Base.metadata.create_all` so fresh DBs get columns without alembic; existing DBs need `alembic upgrade head`.
3. Add DTOs and extend the validator. Keep `bbox_norm` as `List[float]` like `KeyframeBBoxDTO`.
4. Update `jobs.py` to persist both fields. Use `scene.get(...)` so results from older workers (no `ocr_blocks`, e.g. a Colab worker not yet updated) still save.
5. Fix `summaries.py:349`. Behavior change: re-summarize previously received the transcript labeled as OCR; it will now receive real OCR text, or `""` for old videos (columns were NULL). Old videos therefore lose the duplicated transcript text in that field. This is the correct input and is called out in the risk section.
6. Document the DB upgrade step in `HUONG_DAN_CHAY_DU_AN.md` only if it already lists an `alembic upgrade` step (check first; do not add if absent).

## Success Criteria

- [ ] `alembic upgrade head` and `downgrade -1` both succeed on SQLite and Postgres.
- [ ] `sync_job_status` with a scene carrying `ocr_text` + `ocr_blocks` stores them; without those keys stores NULL.
- [ ] `VideoSceneDTO` JSON contains `ocrText` and `ocrBlocksJson` with camelCase block fields (`bboxNorm`, `polygonNorm`, `lineIndex`).
- [ ] Re-summarize path reads `sc.ocr_text`.

## Risk Assessment

- Colab/remote workers running old code return scenes without `ocr_blocks`: handled by `.get`; UI shows the "no detailed OCR" state.
- The `summaries.py` fix changes LLM input for re-summarize on existing videos (transcript no longer duplicated into the OCR slot). Verified by reading L349 that `ocr_text: sc.script` is the only occurrence; intended, but flag in the PR description.
- Rollback: downgrade migration; remove the two schema fields.

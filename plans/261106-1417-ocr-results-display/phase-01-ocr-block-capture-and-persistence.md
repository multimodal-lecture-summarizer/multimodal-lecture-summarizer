---
phase: 1
title: OCR block capture and persistence
status: completed
priority: P1
dependencies: []
---

# Phase 1: OCR block capture and persistence

## Overview

Make the worker keep what PaddleOCR already returns: per text line, the 4-point polygon and confidence, normalized to 0..1 so the UI is resolution independent (same convention as `bbox_norm` in `object_tracker.py`).

## Requirements

- Functional: each scene dict gets `ocr_blocks: list[OcrBlock]` next to the existing `ocr_text`.
- Non-functional: `ocr_text` string stays byte-identical to today (downstream summarizer code depends on it). A failure in block parsing for one image must not drop `ocr_text` for the others.

Block shape (snake_case in worker, camelCased by the API later):

```json
{
  "line_index": 0,
  "text": "Gradient descent",
  "confidence": 0.9731,
  "bbox_norm": [0.12, 0.08, 0.61, 0.15],
  "polygon_norm": [[0.12,0.08],[0.61,0.08],[0.61,0.15],[0.12,0.15]]
}
```

## Architecture

PaddleOCR 2.8.1 (pinned in `ai_workers/requirements.txt`) `ocr.ocr(path, cls=True)` returns `[ [ [poly(4x[x,y]), (text, conf)], ... ] ]`, with `result[0]` being `None` for an empty image. Image size comes from the keyframe file (`cv2.imread` is already a dependency in `object_tracker.py`).

```
keyframe png ──► ocr.ocr() ──► lines ──► _parse_ocr_lines(lines, w, h)
                                           ├─ ocr_text  = " | ".join(texts)   (unchanged)
                                           └─ ocr_blocks = [...normalized...]
scene dict ──► (existing) result["scenes"] ──► jobs.py ──► DB   (phase 2)
```

PaddleOCR already drops lines under its internal `drop_score` (0.5 default), so blocks below 0.5 never appear. The UI slider range must start at 0.5.

## Related Code Files

- Modify: `ai_workers/modules/visual_v2/semantic.py` (`extract_ocr_paddleocr`, add module-level `_parse_ocr_lines`)
- Modify: `ai_workers/core/config.py` (add `OCR_MAX_BLOCKS_PER_SCENE: int = 200` beside `PADDLEOCR_LANG`)
- Create: `ai_workers/tests/test_ocr_blocks.py`
- No change: `ai_workers/tasks.py` (scene dict is passed through; both the parallel branch at ~L483 and the sequential path call `extract_ocr_paddleocr`)
- Not touched: `ai_workers/modules/visual/semantic.py` (v1, not imported by `tasks.py`)

## Implementation Steps

1. Add `_parse_ocr_lines(lines, img_w, img_h, max_blocks)` as a pure function (no PaddleOCR import, easy to test):
   - skip malformed entries (`len(line) < 2`, non-4-point polygon, empty text after `strip()`);
   - normalize each point `x/img_w`, `y/img_h`, clamp to `[0,1]`, round to 4 decimals;
   - `bbox_norm` = min/max over the 4 points (handles rotated text);
   - `confidence` = `float(line[1][1])`, rounded to 4 decimals;
   - `line_index` = index in PaddleOCR output order (keep engine order; do not re-sort);
   - truncate to `max_blocks`.
2. In `extract_ocr_paddleocr`: read image size once per scene (`cv2.imread`; if it returns `None`, set `ocr_blocks = []` but still keep `ocr_text`). Build `text_blocks` from the same parsed lines so `ocr_text` is derived identically to before (`" | ".join(texts).strip()`).
3. Set `scene["ocr_blocks"] = []` in every early-return/exception branch that sets `scene["ocr_text"] = ""` so the key is always present.
4. Add `OCR_MAX_BLOCKS_PER_SCENE` to worker settings and pass it in.
5. Log one line per run: scenes with text / total blocks (helps spot an empty-OCR regression).

## Success Criteria

- [ ] `_parse_ocr_lines` returns normalized, clamped blocks; rotated polygons give a correct axis-aligned `bbox_norm`.
- [ ] Empty/`None` OCR result gives `ocr_text == ""` and `ocr_blocks == []`, no exception.
- [ ] `ocr_text` output equals the previous implementation for the same fixture.
- [ ] A scene whose image cannot be read does not break other scenes.

## Risk Assessment

- Payload growth: ~150 B/block × ≤200 blocks × scenes. Typical slide ~30 blocks, 30 scenes ≈ 135 KB per video. Fine for the JSON column and Celery result; cap protects pathological images.
- PaddleOCR 3.x changes the result format (`predict`). Pinned to 2.8.1; `_parse_ocr_lines` guards shape and skips bad entries rather than crashing.
- Rollback: revert `semantic.py`; extra scene key is ignored by every consumer.

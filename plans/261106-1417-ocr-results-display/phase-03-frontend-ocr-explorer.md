---
phase: 3
title: Frontend OCR explorer
status: completed
priority: P1
dependencies:
  - 2
---

# Phase 3: Frontend OCR explorer

## Overview

New `SceneOcrExplorer` panel that reuses the layout language of `SceneObjectExplorer` (header pills, filter bar, 4/8 column workspace, dark keyframe canvas with overlay, tabbed detail panel) but for OCR blocks.

## Requirements

- Functional: everything in "UI spec" below.
- Non-functional: Tailwind classes and tokens already used by the detection panel (`bg-primary`, `font-heading`, `font-mono-data`, `custom-scrollbar`); no new dependencies (lucide-react already imported there); keyboard accessible controls; works at mobile width (grid collapses to one column like the detection panel).

## UI spec

Mapping from the detection panel to OCR:

| Detection explorer | OCR explorer |
|--------------------|--------------|
| Header pills: unique tracks, total scenes | Pills: scenes with text / total, total blocks, total words, avg confidence |
| Label chip filter bar | Search box with match count + chips "Tất cả / Có chữ / Không có chữ" |
| Scene list w/ thumbnail + label count badges | Scene list w/ thumbnail + `N khối`, avg confidence badge, first ~60 chars of text, match highlight |
| Keyframe canvas + bbox overlay (colour per label) | Keyframe canvas + box overlay (colour per confidence band) |
| Confidence slider (0.1–0.9) | Confidence slider **0.5–0.95**, default 0.5 (PaddleOCR drops <0.5 at source) |
| Show/hide boxes, Tua video | Show/hide boxes, show/hide text labels on boxes, Tua video |
| Hover/click track row ↔ box | Hover/click table row ↔ box (by `lineIndex`), auto-scroll row into view |
| Tab "Danh sách đối tượng", tab "Thông tin phân cảnh" | Tab "Khối văn bản (N)", tab "Văn bản đầy đủ", tab "Thống kê" |

Details:

- **Box colour bands**: ≥0.90 emerald, 0.75–0.90 amber, <0.75 rose. Legend under the canvas. Labels on boxes show `text` truncated to ~24 chars + `NN%`; full text in a hover tooltip. Search matches get a ring so they stand out; non-matching boxes dim (same dim/highlight pattern as detection).
- **Table tab columns**: `#` (lineIndex+1), Văn bản (matched substring highlighted), Độ tin cậy (percentage + thin bar), Vị trí (`x,y %` and `w×h %`), Ký tự, copy button. Click row selects (ring on box), second click clears.
- **Full text tab**: reading-order text, one block per line, with "Sao chép" button; below it the raw stored `ocrText` (the `" | "` join) collapsed in a `<details>` so users can see exactly what the summarizer received. Shows count of lines hidden by the confidence slider.
- **Stats tab**: blocks kept/hidden by threshold, chars, words, avg/min/max confidence, 5-bucket confidence histogram built with plain CSS bars (0.5–0.6 … 0.9–1.0), share of the frame area covered by text boxes.
- **Header actions**: "Sao chép văn bản cảnh" per scene is in the full-text tab; "Xuất JSON" (client-side `Blob` of all scenes' OCR) in the header. Export is optional/P3: drop it if time is tight.
- **States**: loading skeleton like the detection panel; error with "Thử lại"; empty (no scene has `ocrBlocksJson` and no `ocrText`) → card "Chưa có dữ liệu OCR chi tiết. Video được xử lý trước khi hệ thống lưu OCR; hãy xử lý lại để xem."; scene with zero blocks → "Cảnh này không có chữ ≥ ngưỡng".
- **Overlay geometry (deliberate difference from detection panel)**: render the `<img>` in a `relative` wrapper that shrinks to the image's own size (`max-h-[360px] w-auto`), overlay `absolute inset-0` with percentage boxes. Do not use `aspect-video` + `object-contain` + container percentages, which misalign on non-16:9 frames. Use `polygon_norm` via an inline `<svg viewBox="0 0 1 1" preserveAspectRatio="none">` for rotated text, `bbox_norm` for hit areas/labels.

## Architecture

```
ResultsPage (isDevMode)
 ├─ SceneObjectExplorer          (unchanged)
 └─ SceneOcrExplorer(videoId, onSeekVideo)
      api.getVideoScenes(videoId) ─► VideoScene[]
      useMemo: normalize snake/camel → OcrScene[] {index,start,end,keyframeUrl,text,blocks[]}
      state: selectedSceneIndex, minConf, search, showBoxes, showLabels, activeTab,
             hoveredLine, selectedLine
```

Normalization helper handles both `ocrBlocksJson` and `ocr_blocks_json` (the API sends camelCase; the detection panel defensively reads both, so mirror that).

## Related Code Files

- Create: `frontend/src/components/SceneOcrExplorer.tsx` (aim ≤ 500 lines; split pure helpers out if larger)
- Create: `frontend/src/utils/ocr-blocks.ts` (pure functions: `normalizeOcrScene`, `confidenceBand`, `highlightParts(text, query)`, `computeOcrStats(blocks, minConf)`, `buildSceneText(blocks)`)
- Create: `frontend/src/utils/scene-format.ts` (`formatSeconds`, `resolveKeyframeUrl`; same behavior as `SceneObjectExplorer.tsx:181` and the `CONFIG.API_BASE_URL` prefix logic at L718-722)
- Modify: `frontend/src/types/index.ts` (add `OcrBlock` interface with snake+camel variants like `KeyframeBBox`; add `ocrBlocksJson?` / `ocr_blocks_json?` to `VideoScene`; `ocrText` / `ocr_text` already exist at L215-216)
- Modify: `frontend/src/pages/ResultsPage.tsx` (import and mount `SceneOcrExplorer` directly after `SceneObjectExplorer`, ~L1363, inside the existing `videoId &&` guard)
- No change: `SceneObjectExplorer.tsx` (its dead OCR block starts working automatically once the API returns `ocrText`; leave it)

## Implementation Steps

1. Types: add `OcrBlock`, extend `VideoScene`.
2. Write `ocr-blocks.ts` and `scene-format.ts` as pure, importable modules.
3. Build the component shell: fetch, loading/error/empty states, header pills.
4. Scene list + search + filter chips.
5. Canvas: wrapper sizing, SVG polygons, labels, legend, toolbar (slider, toggles, Tua video).
6. Tabs: blocks table with 2-way sync (refs per `lineIndex`, same approach as `trackRowRefs`), full text, stats.
7. Mount in `ResultsPage.tsx`.
8. Optional: JSON export.
9. Manual check in the browser pane against a video that has OCR blocks (dev mode on): hover sync, slider, search, mobile width, dark keyframes, empty-state with an old video.

## Success Criteria

- [ ] Boxes align with the rendered text on 16:9 and non-16:9 keyframes.
- [ ] Hovering a box highlights its row and scrolls it into view, and vice versa.
- [ ] Slider changes the box count, table rows, scene-list badges and stats consistently.
- [ ] Search highlights matches in list, table and canvas; clearing restores everything.
- [ ] Old-video empty state renders; no console errors.
- [ ] `npm run build` (tsc + vite) passes with no new type errors.

## Risk Assessment

- Large component: keep pure logic in `utils/` to keep it testable and the file short.
- Duplicate `getVideoScenes` request in dev mode: accepted; note as follow-up to share a hook.
- Keyframe URL may be relative or R2: reuse the exact prefix logic from the detection panel.
- Rollback: delete the component and the mount line.

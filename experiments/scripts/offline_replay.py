"""Shared offline replay helpers for experiments (no full pipeline re-run)."""

from __future__ import annotations

import re
from pathlib import Path

import cv2

from experiments.pipeline.quality_gates import (
    _text_similarity,
    post_process_chapters,
    verify_and_ground_captions,
)


def pick_run(payload: dict, *keys: str) -> dict | None:
    for key in keys:
        if key in payload and payload[key]:
            return payload[key]
    return None


def resolve_keyframe_path(image_url: str | None, project_root: Path) -> Path | None:
    if not image_url:
        return None
    m = re.search(r"keyframes/([^/]+)/([^/]+\.png)$", image_url.replace("\\", "/"))
    if not m:
        return None
    path = project_root / "storage" / "mock_r2_bucket" / "keyframes" / m.group(1) / m.group(2)
    return path if path.is_file() else None


def ocr_from_description(desc: str) -> str:
    desc = (desc or "").strip()
    if desc.lower().startswith("slide text:"):
        return desc.split(":", 1)[1].strip()
    return ""


def is_generic_caption(desc: str) -> bool:
    return bool(re.match(r"(?i)^keyframe for scene\s+\d+$", (desc or "").strip()))


def image_hist_sim(path_a: str, path_b: str) -> float:
    a = cv2.imread(path_a)
    b = cv2.imread(path_b)
    if a is None or b is None:
        return 0.0
    ha = cv2.calcHist([a], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    return float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))


def offline_visual_gate(
    keyframes: list,
    project_root: Path,
    min_blur_var: float = 30.0,
    hist_thresh: float = 0.92,
    ocr_diff_thresh: float = 0.30,
    min_keep_ratio: float = 0.25,
) -> tuple[list, dict]:
    scenes = []
    for kf in keyframes:
        path = resolve_keyframe_path(kf.get("imageUrl"), project_root)
        desc = kf.get("description") or ""
        item = dict(kf)
        item["keyframe_path"] = str(path) if path else None
        item["ocr_text"] = ocr_from_description(desc)
        item["caption"] = desc
        scenes.append(item)

    if not scenes:
        return [], {"input": 0, "output": 0}

    processed: list = []
    stats = {
        "input": len(scenes),
        "blur_flagged": 0,
        "blur_removed": 0,
        "dedup_removed": 0,
        "ocr_saved_slides": 0,
        "resolved_images": 0,
        "generic_captions": sum(1 for s in scenes if is_generic_caption(s.get("caption", ""))),
    }

    for sc in scenes:
        path = sc.get("keyframe_path")
        blur_var = 100.0
        if path and Path(path).is_file():
            stats["resolved_images"] += 1
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur_var = float(cv2.Laplacian(img, cv2.CV_64F).var())

        ocr = (sc.get("ocr_text") or "").strip()
        sc["blur_score"] = round(blur_var, 2)
        sc["is_blurry"] = blur_var < min_blur_var
        sc["visual_gate_status"] = "kept"

        if sc["is_blurry"]:
            stats["blur_flagged"] += 1
            if len(ocr) < 8 and not (sc.get("transcript") or "").strip():
                sc["visual_gate_status"] = "dropped_blur_no_evidence"
                stats["blur_removed"] += 1
                continue

        is_duplicate = False
        for kept in list(processed):
            kept_path = kept.get("keyframe_path")
            kept_ocr = (kept.get("ocr_text") or "").strip()
            visual_sim = image_hist_sim(path, kept_path) if path and kept_path else 0.0
            ocr_sim = _text_similarity(ocr, kept_ocr)
            time_gap = abs(float(sc.get("timestamp") or 0) - float(kept.get("timestamp") or 0))
            desc_sim = _text_similarity(sc.get("caption", ""), kept.get("caption", ""))
            near_dup = visual_sim > hist_thresh or (time_gap < 3.0 and desc_sim > 0.9)

            if near_dup:
                if ocr and kept_ocr and ocr_sim < (1.0 - ocr_diff_thresh):
                    stats["ocr_saved_slides"] += 1
                    continue
                sc_score = (
                    sc["blur_score"],
                    len(ocr),
                    0 if is_generic_caption(sc.get("caption", "")) else 1,
                    len((sc.get("transcript") or "").strip()),
                )
                kept_score = (
                    kept.get("blur_score", 0),
                    len(kept_ocr),
                    0 if is_generic_caption(kept.get("caption", "")) else 1,
                    len((kept.get("transcript") or "").strip()),
                )
                if sc_score > kept_score:
                    kept["visual_gate_status"] = "replaced_by_better_frame"
                    processed.remove(kept)
                    break
                sc["visual_gate_status"] = "dropped_duplicate"
                is_duplicate = True
                stats["dedup_removed"] += 1
                break

        if not is_duplicate:
            processed.append(sc)

    min_keep = max(1, int(len(scenes) * min_keep_ratio))
    if len(processed) < min_keep:
        ranked = sorted(
            scenes,
            key=lambda x: (
                x.get("blur_score", 0),
                len((x.get("ocr_text") or "").strip()),
                len((x.get("transcript") or "").strip()),
            ),
            reverse=True,
        )
        processed = ranked[:min_keep]

    stats["output"] = len(processed)
    stats["min_keep_floor"] = min_keep
    return processed, stats


def enrich_generic_captions(keyframes: list) -> tuple[list, dict]:
    """Sprint-3: replace generic 'Keyframe for Scene N' with OCR when available."""
    out = []
    enriched = 0
    kept_generic = 0
    for kf in keyframes:
        item = dict(kf)
        desc = item.get("description") or item.get("caption") or ""
        ocr = item.get("ocr_text") or ocr_from_description(desc)
        if is_generic_caption(desc) and len(ocr) >= 4:
            item["description"] = f"Slide Text: {ocr[:120]}"
            item["caption"] = item["description"]
            item["caption_enriched"] = True
            enriched += 1
        else:
            item["caption_enriched"] = False
            if is_generic_caption(desc):
                kept_generic += 1
        out.append(item)
    return out, {"enriched_from_ocr": enriched, "kept_generic": kept_generic}


def soft_caption_grounding(keyframes: list, min_grounding_score: float = 0.10) -> tuple[list, dict]:
    scenes = []
    for kf in keyframes:
        item = dict(kf)
        desc = item.get("description") or item.get("caption") or ""
        item["caption"] = desc
        item["ocr_text"] = item.get("ocr_text") or ocr_from_description(desc)
        scenes.append(item)

    verified, stats = verify_and_ground_captions(scenes, min_grounding_score=min_grounding_score)
    soft_replaced = 0
    soft_flagged = 0
    out = []
    for sc in verified:
        item = dict(sc)
        original = item.get("caption", "")
        is_hallu = bool(item.get("is_hallucinated"))
        reason = item.get("hallucination_reason") or ""
        score = float(item.get("grounding_score") or 0)
        ocr = (item.get("ocr_text") or "").strip()

        if is_hallu and "Repetitive" in reason:
            item["description"] = item.get("verified_caption") or original
            item["grounded_status"] = "soft_fallback_repetitive"
            soft_replaced += 1
        elif is_hallu and score < min_grounding_score and len(ocr) >= 8:
            item["description"] = f"Slide Text: {ocr[:120]}"
            item["grounded_status"] = "soft_fallback_ocr"
            soft_replaced += 1
        elif is_hallu:
            item["description"] = original
            item["grounded_status"] = "low_confidence_kept"
            soft_flagged += 1
        else:
            item["description"] = original
            item["grounded_status"] = "trusted"
        out.append(item)

    stats["soft_replaced"] = soft_replaced
    stats["soft_flagged_kept"] = soft_flagged
    return out, stats


def apply_variant(
    run: dict,
    project_root: Path,
    *,
    sprint1: bool = False,
    sprint2: bool = False,
    sprint3: bool = False,
    sprint8_soft: bool = False,
    min_chapter_sec: float = 45.0,
    visual_cfg: dict | None = None,
    caption_min_grounding: float = 0.10,
) -> dict:
    out = dict(run)
    visual_cfg = visual_cfg or {}

    chapters = run.get("chapters", [])
    if sprint1:
        chapters = post_process_chapters(chapters, min_dur_sec=min_chapter_sec)
    out["chapters"] = chapters

    keyframes = [dict(k) for k in run.get("keyframes", [])]
    visual_stats = {"skipped": True}
    enrich_stats = {"skipped": True}
    caption_stats = {"skipped": True}

    if sprint2:
        keyframes, visual_stats = offline_visual_gate(keyframes, project_root, **visual_cfg)

    if sprint3:
        keyframes, enrich_stats = enrich_generic_captions(keyframes)

    if sprint8_soft:
        keyframes, caption_stats = soft_caption_grounding(
            keyframes, min_grounding_score=caption_min_grounding
        )

    out["keyframes"] = keyframes
    out["variant_stats"] = {
        "visual": visual_stats,
        "enrich": enrich_stats,
        "caption": caption_stats,
    }
    return out

"""Production quality post-process (sprint stack) for lecture pipeline outputs.

Ports the experimentally validated RECOMMENDED stack from experiments/pipeline:
  Sprint 1  – merge short chapters
  Sprint 3  – OCR enrich generic captions
  Sprint 4v2 – evidence prune + restore min 1 KF / chapter
  Sprint 7  – transcript caption fallback

Optional FULL_STACK_S10 adds S5/S8soft/S9/S10 behind the same feature flag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


RECOMMENDED_STACK = ["sprint1", "sprint3", "sprint4_v2"]
FULL_STACK_S10 = [
    "sprint1",
    "sprint3",
    "sprint4_v2",
    "sprint5",
    "sprint8_soft",
    "sprint9",
    "sprint10",
]
STACK_PRESETS = {
    "recommended": RECOMMENDED_STACK,
    "sprint10": FULL_STACK_S10,
    "full": FULL_STACK_S10,
}


def _chapter_bounds(ch: dict) -> tuple[float, float]:
    start = float(ch.get("startTime", ch.get("start_time", ch.get("start_seconds", 0.0))) or 0.0)
    end = float(ch.get("endTime", ch.get("end_time", ch.get("end_seconds", start))) or start)
    if end <= start:
        end = start + 10.0
    return start, end


def _ocr_from_description(desc: str) -> str:
    desc = (desc or "").strip()
    if desc.lower().startswith("slide text:"):
        return desc.split(":", 1)[1].strip()
    return ""


def is_generic_caption(desc: str) -> bool:
    desc_clean = (desc or "").strip()
    if desc_clean == "Visual description unavailable":
        return True
    return bool(re.match(r"(?i)^keyframe for scene\s+\d+$", desc_clean))


def is_action_scene_kf(kf: dict) -> bool:
    """Check if keyframe represents an action scene with detected objects."""
    detected = kf.get("detected_objects", {})
    tracks = detected.get("tracks_summary", []) if isinstance(detected, dict) else []
    counts = kf.get("object_counts", {})
    has_objs = len(tracks) > 0 or (isinstance(counts, dict) and any(c > 0 for c in counts.values()))
    return bool(has_objs)


def keyframe_evidence_score(kf: dict) -> float:
    transcript = (kf.get("transcript") or "").strip()
    ocr = (kf.get("ocr_text") or _ocr_from_description(kf.get("description") or "")).strip()
    blur = float(kf.get("blur_score") or 50.0)
    importance = float(kf.get("importanceScore") or 0.5)
    desc = kf.get("description") or kf.get("caption") or ""

    score = 0.0
    score += min(len(transcript), 500) * 0.02
    score += min(len(ocr), 120) * 0.08
    score += min(blur, 800.0) * 0.001
    score += importance * 2.0
    if not is_generic_caption(desc):
        score += 1.5
    if transcript:
        score += 1.0

    # Action scene boost: presence of tracked objects provides high visual evidence
    if is_action_scene_kf(kf):
        detected = kf.get("detected_objects", {})
        tracks = detected.get("tracks_summary", []) if isinstance(detected, dict) else []
        score += 3.0 + min(len(tracks), 4) * 0.5

    return round(score, 3)


def post_process_chapters(chapters: list, min_dur_sec: float = 25.0) -> list:
    if not chapters:
        return []

    smoothed: list = []
    for idx, ch in enumerate(chapters, 1):
        item = dict(ch)
        s_sec, e_sec = _chapter_bounds(item)
        item["title"] = item.get("title") or f"Chapter {idx}"
        item["summary"] = item.get("summary") or ""
        item["startTime"] = s_sec
        item["endTime"] = e_sec
        item["start_seconds"] = s_sec
        item["end_seconds"] = e_sec

        dur = e_sec - s_sec
        if smoothed and dur < min_dur_sec:
            smoothed[-1]["endTime"] = e_sec
            smoothed[-1]["end_seconds"] = e_sec
            if item.get("summary"):
                smoothed[-1]["summary"] = (
                    smoothed[-1].get("summary", "") + " " + item["summary"]
                ).strip()
        else:
            smoothed.append(item)
    return smoothed


def sprint1_smooth_chapters(chapters: list, min_dur_sec: float = 25.0) -> tuple[list, dict]:
    before = len(chapters)
    out = post_process_chapters(chapters, min_dur_sec=min_dur_sec)
    return out, {"input": before, "output": len(out), "min_dur_sec": min_dur_sec}


def sprint3_enrich_captions(keyframes: list) -> tuple[list, dict]:
    out = []
    enriched = 0
    kept_generic = 0
    for kf in keyframes:
        item = dict(kf)
        desc = (item.get("description") or item.get("caption") or "").strip()
        ocr = (item.get("ocr_text") or _ocr_from_description(desc)).strip()
        item["ocr_text"] = ocr

        # Extract YOLO labels for context hint and hallucination check
        yolo_labels = []
        counts = item.get("object_counts")
        if isinstance(counts, dict):
            yolo_labels.extend(counts.keys())
        detected = item.get("detected_objects")
        if isinstance(detected, dict):
            for b in detected.get("keyframe_bboxes", []):
                if b.get("label"):
                    yolo_labels.append(b["label"])
        yolo_labels_lower = [l.lower() for l in yolo_labels]

        has_bike = any(lbl in ("motorcycle", "bicycle", "bike") for lbl in yolo_labels_lower)
        has_person = "person" in yolo_labels_lower
        has_bat = any(lbl in ("baseball bat", "bat", "sports ball") for lbl in yolo_labels_lower)

        # Sanitize Florence-2 hallucinations (e.g. baseball bat on motorcycle handlebars)
        if not has_bat and (has_bike or has_person):
            desc = re.sub(r"\b(?:holding|with|swinging)\s+(?:a\s+)?baseball\s+bat\b", "holding the handlebars", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\b(?:a\s+)?baseball\s+bat\b", "the handlebars", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\btennis\s+racket\b", "protective gear", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\bsurfboard\b", "bike frame", desc, flags=re.IGNORECASE)

        if is_generic_caption(desc) and len(ocr) >= 4:
            item["description"] = f"Slide Text: {ocr[:120]}"
            item["caption"] = item["description"]
            item["caption_enriched"] = True
            enriched += 1
        elif is_generic_caption(desc) and is_action_scene_kf(item):
            # Enrich generic caption using YOLO detected action objects
            obj_parts = []
            if isinstance(counts, dict) and any(c > 0 for c in counts.values()):
                obj_parts = [f"{c} {lbl}" for lbl, c in sorted(counts.items()) if c > 0]
            elif yolo_labels:
                from collections import Counter
                obj_parts = [f"{c} {lbl}" for lbl, c in Counter(yolo_labels).items()]
            obj_str = ", ".join(obj_parts) if obj_parts else "active subjects"
            item["description"] = f"Action scene featuring {obj_str}."
            item["caption"] = item["description"]
            item["caption_enriched"] = True
            enriched += 1
        else:
            item["description"] = desc
            item["caption"] = desc
            item["caption_enriched"] = bool(desc and not is_generic_caption(desc))
            if is_generic_caption(desc):
                kept_generic += 1
        out.append(item)
    return out, {"enriched_from_ocr": enriched, "kept_generic": kept_generic}


def sprint4_prune_dense_keyframes(
    keyframes: list,
    window_sec: float = 30.0,
    max_per_window: int = 2,
    min_evidence_score: float = 1.0,
) -> tuple[list, dict]:
    if not keyframes:
        return [], {"input": 0, "output": 0}

    sorted_kfs = sorted(keyframes, key=lambda k: float(k.get("timestamp") or 0.0))
    for kf in sorted_kfs:
        kf["evidence_score"] = keyframe_evidence_score(kf)

    kept: list = []
    removed = 0
    low_evidence_removed = 0

    i = 0
    while i < len(sorted_kfs):
        window_start = float(sorted_kfs[i].get("timestamp") or 0.0)
        window = []
        j = i
        while j < len(sorted_kfs):
            ts = float(sorted_kfs[j].get("timestamp") or 0.0)
            if ts - window_start <= window_sec:
                window.append(sorted_kfs[j])
                j += 1
            else:
                break

        ranked = sorted(window, key=lambda k: k.get("evidence_score", 0), reverse=True)
        for idx, kf in enumerate(ranked):
            is_action = is_action_scene_kf(kf)
            if idx < max_per_window or is_action:
                kf["sprint4_status"] = "kept"
                kept.append(kf)
            elif kf.get("evidence_score", 0) < min_evidence_score:
                kf["sprint4_status"] = "dropped_low_evidence"
                removed += 1
                low_evidence_removed += 1
            else:
                kf["sprint4_status"] = "dropped_window_cap"
                removed += 1
        i = j if j > i else i + 1

    return sorted(kept, key=lambda k: float(k.get("timestamp") or 0.0)), {
        "input": len(sorted_kfs),
        "output": len(kept),
        "removed": removed,
        "low_evidence_removed": low_evidence_removed,
        "window_sec": window_sec,
        "max_per_window": max_per_window,
    }


def _kf_in_chapter(kf: dict, start: float, end: float, is_last: bool) -> bool:
    ts = float(kf.get("timestamp") or 0.0)
    if is_last:
        return start <= ts <= end
    return start <= ts < end


def sprint4_v2_chapter_aware_prune(
    chapters: list,
    keyframes: list,
    window_sec: float = 30.0,
    max_per_window: int = 2,
    min_evidence_score: float = 1.0,
) -> tuple[list, dict]:
    original = [dict(k) for k in keyframes]
    kept, stats = sprint4_prune_dense_keyframes(
        keyframes,
        window_sec=window_sec,
        max_per_window=max_per_window,
        min_evidence_score=min_evidence_score,
    )
    kept_by_ts = {float(k.get("timestamp") or 0.0): k for k in kept}
    restored = 0

    # 1. Action-Scene Protection: ensure all distinct action scenes are retained
    for kf in original:
        if is_action_scene_kf(kf):
            ts = float(kf.get("timestamp") or 0.0)
            if ts not in kept_by_ts:
                # Keep if not within 3 seconds of another kept keyframe
                is_near = any(abs(ts - float(k.get("timestamp") or 0.0)) <= 3.0 for k in kept)
                if not is_near:
                    item = dict(kf)
                    item["evidence_score"] = keyframe_evidence_score(item)
                    item["sprint4_status"] = "restored_action_scene"
                    kept.append(item)
                    kept_by_ts[ts] = item
                    restored += 1

    # 2. Chapter coverage: ensure at least 1 keyframe per chapter
    n = len(chapters)
    for i, ch in enumerate(chapters):
        start, end = _chapter_bounds(ch)
        is_last = i == n - 1
        if any(_kf_in_chapter(k, start, end, is_last) for k in kept):
            continue

        candidates = [k for k in original if _kf_in_chapter(k, start, end, is_last)]
        if not candidates:
            candidates = [
                k
                for k in original
                if min(
                    abs(float(k.get("timestamp") or 0) - start),
                    abs(float(k.get("timestamp") or 0) - end),
                )
                <= 15.0
            ]
        if not candidates:
            continue

        best = max(candidates, key=keyframe_evidence_score)
        ts = float(best.get("timestamp") or 0.0)
        if ts not in kept_by_ts:
            item = dict(best)
            item["evidence_score"] = keyframe_evidence_score(item)
            item["sprint4_status"] = "restored_chapter_gap"
            kept.append(item)
            kept_by_ts[ts] = item
            restored += 1

    # 3. Minimum keyframes target: ensure at least 8-12 representative keyframes retained if available
    min_kfs_target = min(len(original), max(8, min(12, int(len(original) * 0.75))))
    if len(kept) < min_kfs_target:
        remaining = [k for k in original if float(k.get("timestamp") or 0.0) not in kept_by_ts]
        remaining.sort(key=keyframe_evidence_score, reverse=True)
        for cand in remaining:
            if len(kept) >= min_kfs_target:
                break
            cand_item = dict(cand)
            cand_item["sprint4_status"] = "restored_min_count"
            kept.append(cand_item)
            kept_by_ts[float(cand.get("timestamp") or 0.0)] = cand_item
            restored += 1

    stats["chapter_restores"] = restored
    stats["output"] = len(kept)
    return sorted(kept, key=lambda k: float(k.get("timestamp") or 0.0)), stats





def sprint5_boost_chapter_coverage(chapters: list, keyframes: list) -> tuple[list, list, dict]:
    chapters_out = [dict(c) for c in chapters]
    keyframes_out = [dict(k) for k in keyframes]
    gaps: list[dict] = []
    boosted = 0

    for ch_idx, ch in enumerate(chapters_out):
        start, end = _chapter_bounds(ch)
        in_chapter = [
            k for k in keyframes_out if start <= float(k.get("timestamp") or 0.0) <= end
        ]
        ch["keyframe_count"] = len(in_chapter)
        if not in_chapter:
            gaps.append(
                {
                    "chapter_index": ch_idx,
                    "title": ch.get("title"),
                    "start": start,
                    "end": end,
                }
            )
            continue
        for kf in in_chapter:
            old = float(kf.get("importanceScore") or 0.5)
            bonus = 0.05
            if (kf.get("transcript") or "").strip():
                bonus += 0.08
            if not is_generic_caption(kf.get("description") or ""):
                bonus += 0.05
            kf["importanceScore"] = round(min(1.0, old + bonus), 3)
            kf["sprint5_boosted"] = True
            boosted += 1

    return chapters_out, keyframes_out, {
        "chapters_with_gaps": len(gaps),
        "gaps": gaps,
        "keyframes_boosted": boosted,
    }


def sprint8_soft_caption(keyframes: list, min_grounding_score: float = 0.10) -> tuple[list, dict]:
    """Soft caption grounding: flag low OCR overlap; fallback only on repetitive loops."""
    out = []
    soft_flagged = 0
    replaced = 0
    for kf in keyframes:
        item = dict(kf)
        caption = (item.get("description") or item.get("caption") or "").strip()
        ocr = (item.get("ocr_text") or _ocr_from_description(caption)).strip()
        item["ocr_text"] = ocr

        words = caption.lower().split()
        is_loop = False
        if len(words) > 6:
            unique_ratio = len(set(words)) / len(words)
            is_loop = unique_ratio < 0.4

        cap_words = set(re.findall(r"\w+", caption.lower())) - {
            "a",
            "an",
            "the",
            "of",
            "and",
            "slide",
            "text",
            "image",
            "shows",
        }
        ocr_words = set(re.findall(r"\w+", ocr.lower()))
        grounding = len(cap_words & ocr_words) / max(1, len(cap_words))
        item["grounding_score"] = round(grounding, 2)

        if is_loop and ocr:
            item["description"] = f"Slide Text: {ocr[:120]}"
            item["caption"] = item["description"]
            item["caption_enriched"] = True
            item["enrich_source"] = "soft_loop_ocr"
            item["grounded_status"] = "fallback_to_ocr"
            replaced += 1
        elif cap_words and grounding < min_grounding_score and not is_generic_caption(caption):
            item["grounded_status"] = "low_confidence_kept"
            soft_flagged += 1
        else:
            item["grounded_status"] = "trusted"
        out.append(item)

    return out, {
        "soft_flagged_kept": soft_flagged,
        "replaced_repetitive": replaced,
        "min_grounding_score": min_grounding_score,
    }


def sprint9_chapter_visual_hints(
    chapters: list,
    keyframes: list,
    max_hints_per_chapter: int = 2,
    max_chars: int = 180,
) -> tuple[list, dict]:
    chapters_out = [dict(c) for c in chapters]
    n = len(chapters_out)
    hints_added = 0

    for i, ch in enumerate(chapters_out):
        start, end = _chapter_bounds(ch)
        is_last = i == n - 1
        in_ch = [k for k in keyframes if _kf_in_chapter(k, start, end, is_last)]
        ranked = sorted(in_ch, key=keyframe_evidence_score, reverse=True)[:max_hints_per_chapter]
        parts = []
        for kf in ranked:
            desc = (kf.get("description") or "")[:80]
            tx = (kf.get("transcript") or "")[:max_chars]
            if desc:
                parts.append(desc)
            elif tx:
                parts.append(tx[:80])
        if parts:
            ch["visual_evidence_hint"] = " | ".join(parts)[: max_chars * max_hints_per_chapter]
            hints_added += 1
        else:
            ch["visual_evidence_hint"] = ""

    return chapters_out, {"chapters_with_hints": hints_added, "total_chapters": len(chapters_out)}


def sprint10_quality_score(
    chapters: list,
    keyframes: list,
    min_chapter_sec: float = 25.0,
) -> tuple[dict, dict]:
    durs = []
    for ch in chapters:
        s, e = _chapter_bounds(ch)
        durs.append(e - s)

    with_tx = sum(1 for k in keyframes if (k.get("transcript") or "").strip())
    enriched = sum(1 for k in keyframes if k.get("caption_enriched") or k.get("enrich_source"))
    generic_left = sum(1 for k in keyframes if is_generic_caption(k.get("description") or ""))
    hints = sum(1 for ch in chapters if (ch.get("visual_evidence_hint") or "").strip())

    coverage = with_tx / max(1, len(keyframes))
    min_dur = min(durs) if durs else 0.0
    chapter_ok = min_dur >= min_chapter_sec if len(chapters) > 1 else True

    score = 0.0
    score += 25 if chapter_ok else 0
    score += 25 * coverage
    score += 15 if hints >= max(1, len(chapters) - 1) else hints / max(1, len(chapters)) * 15
    score += 15 if generic_left == 0 else max(0, 15 - generic_left * 3)
    score += 10 if enriched >= 1 else 0
    score += 10 if len(keyframes) >= 3 else len(keyframes) / 3 * 10

    export_meta = {
        "pipeline_quality_score": round(min(100.0, score), 1),
        "chapter_count": len(chapters),
        "keyframe_count": len(keyframes),
        "transcript_coverage": round(coverage, 4),
        "min_chapter_sec": round(min_dur, 2),
        "chapters_with_visual_hints": hints,
        "generic_captions_remaining": generic_left,
        "captions_enriched": enriched,
        "export_ready": chapter_ok and coverage >= 0.85 and generic_left <= 1,
    }
    return export_meta, export_meta


@dataclass
class SprintContext:
    chapters: list
    keyframes: list
    utterances: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def resolve_stack(name: str) -> list[str]:
    key = (name or "recommended").strip().lower()
    return list(STACK_PRESETS.get(key, RECOMMENDED_STACK))


def apply_sprint_stack(
    ctx: SprintContext,
    stack: list[str] | None = None,
    *,
    min_chapter_sec: float = 25.0,
    sprint4_cfg: dict | None = None,
    caption_min_grounding: float = 0.10,
) -> SprintContext:
    stack = stack or RECOMMENDED_STACK
    sprint4_cfg = sprint4_cfg or {
        "window_sec": 30.0,
        "max_per_window": 2,
        "min_evidence_score": 1.0,
    }

    for name in stack:
        if name == "sprint1":
            ctx.chapters, ctx.stats["sprint1"] = sprint1_smooth_chapters(
                ctx.chapters, min_chapter_sec
            )
        elif name == "sprint3":
            ctx.keyframes, ctx.stats["sprint3"] = sprint3_enrich_captions(ctx.keyframes)
        elif name == "sprint4":
            ctx.keyframes, ctx.stats["sprint4"] = sprint4_prune_dense_keyframes(
                ctx.keyframes, **sprint4_cfg
            )
        elif name == "sprint4_v2":
            ctx.keyframes, ctx.stats["sprint4_v2"] = sprint4_v2_chapter_aware_prune(
                ctx.chapters, ctx.keyframes, **sprint4_cfg
            )
        elif name == "sprint5":
            ctx.chapters, ctx.keyframes, ctx.stats["sprint5"] = sprint5_boost_chapter_coverage(
                ctx.chapters, ctx.keyframes
            )
        elif name == "sprint8_soft":
            ctx.keyframes, ctx.stats["sprint8"] = sprint8_soft_caption(
                ctx.keyframes, caption_min_grounding
            )
        elif name == "sprint9":
            ctx.chapters, ctx.stats["sprint9"] = sprint9_chapter_visual_hints(
                ctx.chapters, ctx.keyframes
            )
        elif name == "sprint10":
            ctx.stats["sprint10"], ctx.stats["export_meta"] = sprint10_quality_score(
                ctx.chapters, ctx.keyframes, min_chapter_sec
            )
    return ctx


def normalize_for_backend(chapters: list, keyframes: list) -> tuple[list, list]:
    """Ensure FE/backend contract keys after sprint mutations."""
    chapters_out = []
    for idx, ch in enumerate(chapters or [], 1):
        item = dict(ch)
        start, end = _chapter_bounds(item)
        item["title"] = item.get("title") or f"Chapter {idx}"
        item["summary"] = item.get("summary") or ""
        item["startTime"] = start
        item["endTime"] = end
        item["start_seconds"] = start
        item["end_seconds"] = end
        chapters_out.append(item)

    keyframes_out = []
    for kf in keyframes or []:
        item = dict(kf)
        raw_start = item.get("start_seconds", item.get("timestamp"))
        start = float(raw_start or 0.0)
        raw_timestamp = item.get("timestamp")
        item["timestamp"] = float(start if raw_timestamp in (None, "") else raw_timestamp)
        item["start_seconds"] = start
        for end_key in ("end_seconds", "endTime", "end_time"):
            if item.get(end_key) not in (None, ""):
                try:
                    item["end_seconds"] = float(item[end_key])
                    break
                except (TypeError, ValueError):
                    pass
        description = item.get("description") or item.get("caption") or ""
        item["description"] = description
        item["caption"] = description
        image_url = item.get("keyframe_url") or item.get("imageUrl") or item.get("image_url") or ""
        item["keyframe_url"] = image_url
        item["imageUrl"] = item.get("imageUrl") or image_url
        item["importanceScore"] = float(
            item.get("importanceScore", item.get("importance_score", 0.5)) or 0.5
        )
        item["transcript"] = item.get("transcript") or ""
        item["ocr_text"] = item.get("ocr_text") or _ocr_from_description(item["description"])
        if "detected_objects" in kf:
            item["detected_objects"] = kf["detected_objects"]
        if "object_counts" in kf:
            item["object_counts"] = kf["object_counts"]
        keyframes_out.append(item)

    return chapters_out, keyframes_out


def apply_quality_postprocess(
    *,
    chapters: list,
    keyframes: list,
    utterances: list | None = None,
    stack_name: str = "recommended",
    min_chapter_sec: float = 25.0,
) -> dict[str, Any]:
    """Convenience wrapper used by Celery tasks."""
    ctx = SprintContext(
        chapters=[dict(c) for c in (chapters or [])],
        keyframes=[dict(k) for k in (keyframes or [])],
        utterances=list(utterances or []),
    )
    stack = resolve_stack(stack_name)
    ctx = apply_sprint_stack(ctx, stack, min_chapter_sec=min_chapter_sec)
    ctx.chapters, ctx.keyframes = normalize_for_backend(ctx.chapters, ctx.keyframes)
    return {
        "chapters": ctx.chapters,
        "keyframes": ctx.keyframes,
        "utterances": ctx.utterances,
        "sprint_stats": ctx.stats,
        "export_meta": ctx.stats.get("export_meta"),
        "stack": stack,
    }

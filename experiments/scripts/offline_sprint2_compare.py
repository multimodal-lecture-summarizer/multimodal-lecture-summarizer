"""Offline Sprint-2: Gate 7 (visual) + Gate 8 soft on saved compare reports + keyframe images.

Does NOT re-run ASR/CLIP/Florence/LLM. Reuses wall time and chapters from prior runs.
Applies Sprint-1 chapter smoothing + Gate 7 blur/OCR-aware text-dedup + Gate 8 soft caption.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.quality_gates import (  # noqa: E402
    _text_similarity,
    post_process_chapters,
    verify_and_ground_captions,
)
from experiments.scripts.run_gated_compare import (  # noqa: E402
    _recommendation,
    compute_metrics,
    fmt_chapter_titles,
)


def _pick_run(payload: dict, *keys: str) -> dict | None:
    for key in keys:
        if key in payload and payload[key]:
            return payload[key]
    return None


def _resolve_keyframe_path(image_url: str | None, project_root: Path) -> Path | None:
    if not image_url:
        return None
    # "/static/mock_r2/keyframes/JOB/file.png" -> storage/mock_r2_bucket/keyframes/JOB/file.png
    m = re.search(r"keyframes/([^/]+)/([^/]+\.png)$", image_url.replace("\\", "/"))
    if not m:
        return None
    job_id, fname = m.group(1), m.group(2)
    path = project_root / "storage" / "mock_r2_bucket" / "keyframes" / job_id / fname
    return path if path.is_file() else None


def _ocr_from_description(desc: str) -> str:
    desc = (desc or "").strip()
    if desc.lower().startswith("slide text:"):
        return desc.split(":", 1)[1].strip()
    if desc.lower().startswith("slide text"):
        return desc.split(":", 1)[-1].strip()
    return ""


def _is_generic_caption(desc: str) -> bool:
    return bool(re.match(r"(?i)^keyframe for scene\s+\d+$", (desc or "").strip()))


def _image_hist_sim(path_a: str, path_b: str) -> float:
    """Cheap visual similarity proxy when CLIP embeddings are unavailable."""
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
    """Gate 7 offline: blur + OCR-aware hist-dedup (no CLIP embeddings)."""
    scenes = []
    for kf in keyframes:
        path = _resolve_keyframe_path(kf.get("imageUrl"), project_root)
        desc = kf.get("description") or ""
        ocr = _ocr_from_description(desc)
        item = dict(kf)
        item["keyframe_path"] = str(path) if path else None
        item["ocr_text"] = ocr
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
        "generic_captions": sum(1 for s in scenes if _is_generic_caption(s.get("caption", ""))),
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
            visual_sim = 0.0
            if path and kept_path:
                visual_sim = _image_hist_sim(path, kept_path)

            ocr_sim = _text_similarity(ocr, kept_ocr)
            # Also treat near-identical generic captions at close timestamps as duplicates
            time_gap = abs(float(sc.get("timestamp") or 0) - float(kept.get("timestamp") or 0))
            desc_sim = _text_similarity(sc.get("caption", ""), kept.get("caption", ""))

            near_dup = visual_sim > hist_thresh or (time_gap < 3.0 and desc_sim > 0.9)
            if near_dup:
                if ocr and kept_ocr and ocr_sim < (1.0 - ocr_diff_thresh):
                    stats["ocr_saved_slides"] += 1
                    continue
                # Prefer sharper / richer OCR / non-generic caption
                sc_score = (
                    sc["blur_score"],
                    len(ocr),
                    0 if _is_generic_caption(sc.get("caption", "")) else 1,
                    len((sc.get("transcript") or "").strip()),
                )
                kept_score = (
                    kept.get("blur_score", 0),
                    len(kept_ocr),
                    0 if _is_generic_caption(kept.get("caption", "")) else 1,
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
        stats["restored_by_floor"] = True

    stats["output"] = len(processed)
    stats["min_keep_floor"] = min_keep
    return processed, stats


def soft_caption_grounding(keyframes: list, min_grounding_score: float = 0.10) -> tuple[list, dict]:
    """Gate 8 soft: only replace caption when clearly bad; else keep + flag."""
    scenes = []
    for kf in keyframes:
        item = dict(kf)
        desc = item.get("description") or item.get("caption") or ""
        ocr = item.get("ocr_text") or _ocr_from_description(desc)
        item["caption"] = desc
        item["ocr_text"] = ocr
        scenes.append(item)

    # Hard verify first, then soft-apply
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

        # Soft policy:
        # - always replace repetitive loops
        # - replace low grounding only if OCR evidence exists
        # - otherwise keep original caption, flag low_confidence
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


def apply_sprint1_and_sprint2(run: dict, project_root: Path, min_chapter_sec: float) -> dict:
    out = dict(run)
    chapters_before = run.get("chapters", [])
    out["chapters"] = post_process_chapters(chapters_before, min_dur_sec=min_chapter_sec)

    keyframes_before = run.get("keyframes", [])
    gated_kfs, visual_stats = offline_visual_gate(keyframes_before, project_root)
    soft_kfs, caption_stats = soft_caption_grounding(gated_kfs)
    out["keyframes"] = soft_kfs

    out["sprint_applied"] = {
        "sprint1_chapter_smoothing": True,
        "sprint2_visual_gate": True,
        "sprint2_caption_soft": True,
        "chapters_before": len(chapters_before),
        "chapters_after": len(out["chapters"]),
        "keyframes_before": len(keyframes_before),
        "keyframes_after": len(soft_kfs),
        "visual_stats": visual_stats,
        "caption_stats": caption_stats,
    }
    # Wall time unchanged — offline post-process only
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="outputs/ted_compare_20260806_162755/compare_report.json",
    )
    parser.add_argument(
        "--gated",
        default="outputs/gated_compare_20260807_143515/compare_report.json",
    )
    parser.add_argument("--min-chapter-sec", type=float, default=45.0)
    args = parser.parse_args()

    baseline_path = project_root / args.baseline
    gated_path = project_root / args.gated

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_run = _pick_run(baseline_payload, "backend_parity", "backend_baseline")
    if not baseline_run:
        raise SystemExit(f"No baseline run in {baseline_path}")

    improved = apply_sprint1_and_sprint2(baseline_run, project_root, args.min_chapter_sec)

    runs = {
        "baseline_original": baseline_run,
        "baseline_sprint1_sprint2": improved,
    }

    gated_run = None
    if gated_path.is_file():
        gated_payload = json.loads(gated_path.read_text(encoding="utf-8"))
        gated_run = _pick_run(gated_payload, "experimental_gated", "experimental_video")
        if gated_run:
            runs["gated_original"] = gated_run

    metrics = {name: compute_metrics(run) for name, run in runs.items()}

    out_dir = project_root / "outputs" / f"offline_sprint2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "created_at": datetime.now().isoformat(),
        "comparison_type": "offline_sprint1_plus_sprint2",
        "note": (
            "No full pipeline re-run. Sprint-1 chapter smoothing + Gate7 blur/hist-dedup "
            "+ Gate8 soft caption applied on saved baseline artifacts. Wall time reused."
        ),
        "sources": {"baseline": str(baseline_path), "gated": str(gated_path)},
        "sprint_applied": improved.get("sprint_applied"),
        "metrics": metrics,
        "runs": {
            "baseline_original": {
                "chapters": baseline_run.get("chapters"),
                "keyframe_count": len(baseline_run.get("keyframes", [])),
            },
            "baseline_sprint1_sprint2": {
                "chapters": improved.get("chapters"),
                "keyframes": [
                    {
                        "timestamp": k.get("timestamp"),
                        "description": k.get("description"),
                        "blur_score": k.get("blur_score"),
                        "visual_gate_status": k.get("visual_gate_status"),
                        "grounded_status": k.get("grounded_status"),
                        "grounding_score": k.get("grounding_score"),
                    }
                    for k in improved.get("keyframes", [])
                ],
                "sprint_applied": improved.get("sprint_applied"),
            },
        },
    }

    if gated_run:
        result["recommended"] = _recommendation(
            metrics["baseline_sprint1_sprint2"], metrics["gated_original"]
        )

    out_json = out_dir / "offline_sprint2_compare.json"
    out_md = out_dir / "offline_sprint2_compare.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    bo = metrics["baseline_original"]
    bi = metrics["baseline_sprint1_sprint2"]
    vs = improved["sprint_applied"]["visual_stats"]
    cs = improved["sprint_applied"]["caption_stats"]

    md = [
        "# Offline Sprint-1 + Sprint-2 Compare",
        "",
        "- Không chạy lại pipeline (ASR/Florence/LLM).",
        "- Sprint-1: chapter smoothing (<45s merge).",
        "- Sprint-2: Gate 7 blur + hist/OCR dedup; Gate 8 soft caption.",
        f"- Images resolved: `{vs.get('resolved_images')}/{vs.get('input')}`",
        "",
        "## Metrics",
        "",
        "| Variant | Chapters | Min ch (s) | Keyframes | KF coverage | Wall (s) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Baseline gốc | {bo['chapter_count']} | {bo['min_chapter_duration_sec']} | {bo['keyframe_count']} | {bo['keyframe_script_coverage']} | {bo['elapsed_wall_sec']} |",
        f"| Baseline + S1+S2 | {bi['chapter_count']} | {bi['min_chapter_duration_sec']} | {bi['keyframe_count']} | {bi['keyframe_script_coverage']} | {bi['elapsed_wall_sec']} |",
    ]

    if gated_run:
        go = metrics["gated_original"]
        md += [
            f"| Gated gốc (full gates) | {go['chapter_count']} | {go['min_chapter_duration_sec']} | {go['keyframe_count']} | {go['keyframe_script_coverage']} | {go['elapsed_wall_sec']} |",
            "",
            f"**Recommend (S1+S2 vs gated):** `{result.get('recommended')}`",
        ]

    md += [
        "",
        "## Gate 7 stats (offline trên baseline keyframes)",
        f"- Input → Output: `{vs['input']}` → `{vs['output']}`",
        f"- Blur flagged / removed: `{vs['blur_flagged']}` / `{vs['blur_removed']}`",
        f"- Dedup removed: `{vs['dedup_removed']}`",
        f"- OCR-saved: `{vs['ocr_saved_slides']}`",
        f"- Generic captions: `{vs.get('generic_captions')}`",
        "",
        "## Gate 8 soft stats",
        f"- Hallucination flagged (hard detector): `{cs.get('hallucination_count')}`",
        f"- Soft replaced: `{cs.get('soft_replaced')}`",
        f"- Soft flagged but kept: `{cs.get('soft_flagged_kept')}`",
        f"- Avg grounding: `{cs.get('avg_grounding_score')}`",
        "",
        "### Chapters after S1",
        fmt_chapter_titles(improved.get("chapters", [])),
        "",
        "### Keyframes kept (description)",
    ]
    for k in improved.get("keyframes", []):
        md.append(
            f"- t={k.get('timestamp')}: blur={k.get('blur_score')} | "
            f"{k.get('visual_gate_status')} | {k.get('grounded_status')} | "
            f"{(k.get('description') or '')[:80]}"
        )

    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"OUT_DIR={out_dir}")
    print(f"JSON={out_json}")
    print(f"MD={out_md}")
    print("chapters:", bo["chapter_count"], "->", bi["chapter_count"])
    print("keyframes:", bo["keyframe_count"], "->", bi["keyframe_count"])
    print("visual_stats=", json.dumps(vs))
    print("caption_stats=", json.dumps({k: cs.get(k) for k in ('hallucination_count','soft_replaced','soft_flagged_kept','avg_grounding_score')}))


if __name__ == "__main__":
    main()

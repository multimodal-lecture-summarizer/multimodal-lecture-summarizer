"""Merge existing baseline compare JSON with a new gated pipeline run."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.scripts.run_gated_compare import (  # noqa: E402
    _recommendation,
    compute_metrics,
    fmt_chapter_titles,
)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: merge_gated_with_baseline.py <baseline_compare.json> <gated_compare.json>")
        sys.exit(1)

    baseline_path = Path(sys.argv[1])
    gated_path = Path(sys.argv[2])
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    gated_payload = json.loads(gated_path.read_text(encoding="utf-8"))

    backend_run = baseline_payload.get("backend_parity") or baseline_payload.get("backend_baseline")
    gated_run = gated_payload.get("experimental_gated")
    if not backend_run or not gated_run:
        raise SystemExit("Missing backend or gated sections in input JSON files.")

    backend_run = dict(backend_run)
    backend_run["config_stack"] = backend_run.get("config_stack", "backend_baseline")
    backend_run["elapsed_wall_sec"] = backend_run.get("elapsed_wall_sec") or backend_run.get("processing_time")

    merged = {
        "video_path": baseline_payload.get("video_path") or gated_payload.get("video_path"),
        "created_at": datetime.now().isoformat(),
        "comparison_type": "merged_baseline_plus_gated_experimental",
        "baseline_source": str(baseline_path),
        "gated_source": str(gated_path),
        "backend_baseline": backend_run,
        "experimental_gated": gated_run,
    }

    backend_metrics = compute_metrics(backend_run)
    exp_metrics = compute_metrics(gated_run)
    merged["metrics"] = {
        "backend_baseline": backend_metrics,
        "experimental_gated": exp_metrics,
        "recommended_mode": _recommendation(backend_metrics, exp_metrics),
        "delta": {
            "chapters": exp_metrics["chapter_count"] - backend_metrics["chapter_count"],
            "keyframes": exp_metrics["keyframe_count"] - backend_metrics["keyframe_count"],
            "wall_time_sec": round(exp_metrics["elapsed_wall_sec"] - backend_metrics["elapsed_wall_sec"], 2),
        },
    }

    out_dir = gated_path.parent / "merged_with_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "compare_report.json"
    out_md = out_dir / "compare_report.md"

    out_json.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    bm = merged["metrics"]["backend_baseline"]
    em = merged["metrics"]["experimental_gated"]
    md = [
        "# Merged TED Compare: Backend Baseline + Gated Experimental",
        "",
        f"- Video: `{merged['video_path']}`",
        f"- Baseline source: `{baseline_path}`",
        f"- Gated source: `{gated_path}`",
        "",
        "## Backend Baseline",
        f"- Chapters: {bm['chapter_count']} | Keyframes: {bm['keyframe_count']} | Wall: {bm['elapsed_wall_sec']}s",
        f"- Titles: {fmt_chapter_titles(backend_run.get('chapters', []))}",
        "",
        "## Experimental Gated (5/6/7/8)",
        f"- Chapters: {em['chapter_count']} | Keyframes: {em['keyframe_count']} | Wall: {em['elapsed_wall_sec']}s",
        f"- ASR: {em.get('asr_quality_status')} | Speaker: {em.get('speaker_reliability')}",
        f"- Caption hallucinations flagged: {em.get('caption_hallucinations')}",
        f"- Avg grounding score: {em.get('avg_grounding_score')}",
        f"- Titles: {fmt_chapter_titles(gated_run.get('chapters', []))}",
        "",
        "## Delta",
        f"- Chapters: {merged['metrics']['delta']['chapters']}",
        f"- Keyframes: {merged['metrics']['delta']['keyframes']}",
        f"- Wall time: {merged['metrics']['delta']['wall_time_sec']}s",
        "",
        f"## Recommendation: `{merged['metrics']['recommended_mode']}`",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"MERGED_JSON={out_json}")
    print(f"MERGED_MD={out_md}")
    print("RECOMMENDED_MODE=", merged["metrics"]["recommended_mode"])


if __name__ == "__main__":
    main()

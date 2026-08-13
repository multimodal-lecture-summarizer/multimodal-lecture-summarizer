"""Offline Sprint-1 replay: apply chapter/utterance post-process on saved compare reports.

Does NOT re-run the pipeline. Reuses wall time, keyframes, summary from original runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.quality_gates import post_process_chapters  # noqa: E402
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


def _apply_sprint1_chapters(run: dict, min_dur_sec: float) -> dict:
    out = dict(run)
    original = run.get("chapters", [])
    smoothed = post_process_chapters(original, min_dur_sec=min_dur_sec)
    out["chapters"] = smoothed
    out["sprint1_applied"] = {
        "chapter_smoothing": True,
        "min_dur_sec": min_dur_sec,
        "chapters_before": len(original),
        "chapters_after": len(smoothed),
    }
    return out


def _chapter_table_rows(label: str, metrics: dict) -> list[str]:
    return [
        f"| {label} | {metrics['chapter_count']} | {metrics['min_chapter_duration_sec']} | "
        f"{metrics['avg_chapter_duration_sec']} | {metrics['max_chapter_duration_sec']} | "
        f"{metrics['elapsed_wall_sec']} | {metrics['keyframe_count']} | "
        f"{metrics['keyframe_script_coverage']} |",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Sprint-1 compare from saved JSON")
    parser.add_argument(
        "--baseline",
        default="outputs/ted_compare_20260806_162755/compare_report.json",
        help="Baseline compare_report.json",
    )
    parser.add_argument(
        "--gated",
        default="outputs/gated_compare_20260807_143515/compare_report.json",
        help="Gated compare_report.json (optional)",
    )
    parser.add_argument("--min-chapter-sec", type=float, default=45.0)
    args = parser.parse_args()

    baseline_path = project_root / args.baseline
    gated_path = project_root / args.gated

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_run = _pick_run(baseline_payload, "backend_parity", "backend_baseline")
    if not baseline_run:
        raise SystemExit(f"No baseline run found in {baseline_path}")

    baseline_sprint1 = _apply_sprint1_chapters(baseline_run, args.min_chapter_sec)

    runs: dict[str, dict] = {
        "baseline_original": baseline_run,
        "baseline_sprint1": baseline_sprint1,
    }

    gated_run = None
    gated_sprint1 = None
    if gated_path.is_file():
        gated_payload = json.loads(gated_path.read_text(encoding="utf-8"))
        gated_run = _pick_run(gated_payload, "experimental_gated", "experimental_video")
        if gated_run:
            gated_sprint1 = _apply_sprint1_chapters(gated_run, args.min_chapter_sec)
            runs["gated_original"] = gated_run
            runs["gated_sprint1"] = gated_sprint1

    metrics = {name: compute_metrics(run) for name, run in runs.items()}

    out_dir = project_root / "outputs" / f"offline_sprint1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "created_at": datetime.now().isoformat(),
        "comparison_type": "offline_sprint1_chapter_smoothing",
        "note": "Wall time, keyframes, summary reused from original pipeline runs. Only chapters reprocessed.",
        "min_chapter_sec": args.min_chapter_sec,
        "sources": {
            "baseline": str(baseline_path),
            "gated": str(gated_path) if gated_path.is_file() else None,
        },
        "runs": runs,
        "metrics": metrics,
        "delta_baseline_sprint1": {
            "chapters": metrics["baseline_sprint1"]["chapter_count"]
            - metrics["baseline_original"]["chapter_count"],
            "min_chapter_sec": round(
                metrics["baseline_sprint1"]["min_chapter_duration_sec"]
                - metrics["baseline_original"]["min_chapter_duration_sec"],
                2,
            ),
        },
    }

    if gated_run:
        result["delta_baseline_sprint1_vs_gated_original"] = {
            "chapters": metrics["gated_original"]["chapter_count"]
            - metrics["baseline_sprint1"]["chapter_count"],
            "min_chapter_sec": round(
                metrics["gated_original"]["min_chapter_duration_sec"]
                - metrics["baseline_sprint1"]["min_chapter_duration_sec"],
                2,
            ),
        }
        result["recommended_after_sprint1"] = _recommendation(
            metrics["baseline_sprint1"], metrics["gated_original"]
        )

    out_json = out_dir / "offline_sprint1_compare.json"
    out_md = out_dir / "offline_sprint1_compare.md"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    bo = metrics["baseline_original"]
    bs = metrics["baseline_sprint1"]
    md = [
        "# Offline Sprint-1 Compare (không chạy lại pipeline)",
        "",
        f"- Baseline source: `{baseline_path}`",
        f"- Gated source: `{gated_path if gated_path.is_file() else 'N/A'}`",
        f"- Chapter smoothing: merge chapter < **{args.min_chapter_sec}s**",
        "- Wall time / keyframes / summary: **giữ nguyên từ run cũ**",
        "",
        "## Baseline: trước vs sau Sprint-1",
        "",
        "| Variant | Chapters | Min ch (s) | Avg ch (s) | Max ch (s) | Wall (s) | Keyframes | KF coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        *_chapter_table_rows("Baseline gốc", bo),
        *_chapter_table_rows("Baseline + Sprint-1", bs),
        "",
        f"**Delta chapters:** {result['delta_baseline_sprint1']['chapters']}",
        f"**Delta min chapter:** {result['delta_baseline_sprint1']['min_chapter_sec']}s",
        "",
        "### Chapter titles sau Sprint-1 (baseline)",
        fmt_chapter_titles(baseline_sprint1.get("chapters", [])),
        "",
    ]

    if gated_run:
        go = metrics["gated_original"]
        md += [
            "## So với Gated gốc (chỉ số cũ, không re-run)",
            "",
            "| Variant | Chapters | Min ch (s) | Avg ch (s) | Wall (s) | Keyframes |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Baseline + Sprint-1 | {bs['chapter_count']} | {bs['min_chapter_duration_sec']} | {bs['avg_chapter_duration_sec']} | {bs['elapsed_wall_sec']} | {bs['keyframe_count']} |",
            f"| Gated gốc | {go['chapter_count']} | {go['min_chapter_duration_sec']} | {go['avg_chapter_duration_sec']} | {go['elapsed_wall_sec']} | {go['keyframe_count']} |",
            "",
            f"**Recommend (baseline+sprint1 vs gated gốc):** `{result.get('recommended_after_sprint1')}`",
        ]

    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"OUT_DIR={out_dir}")
    print(f"JSON={out_json}")
    print(f"MD={out_md}")
    print("BASELINE chapters:", bo["chapter_count"], "->", bs["chapter_count"])
    print("BASELINE min_chapter_sec:", bo["min_chapter_duration_sec"], "->", bs["min_chapter_duration_sec"])


if __name__ == "__main__":
    main()

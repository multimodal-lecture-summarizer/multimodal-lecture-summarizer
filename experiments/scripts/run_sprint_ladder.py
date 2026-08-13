"""Run cumulative Sprint ladder 1→10 on saved baseline (no pipeline re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.sprints import (
    RECOMMENDED_STACK_CONFIG,
    SPRINT_LADDER,
    SprintContext,
    apply_sprint_stack,
)
from experiments.scripts.offline_replay import pick_run
from experiments.scripts.run_gated_compare import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/ted_compare_20260806_162755/compare_report.json")
    parser.add_argument("--max-sprint", type=int, default=10)
    args = parser.parse_args()

    baseline_run = pick_run(
        json.loads((project_root / args.baseline).read_text(encoding="utf-8")),
        "backend_parity", "backend_baseline",
    )
    if not baseline_run:
        raise SystemExit("No baseline run.")

    ladder_results = []
    for sprint_num in range(1, min(args.max_sprint, 10) + 1):
        stack = SPRINT_LADDER.get(sprint_num, SPRINT_LADDER[10])
        ctx = SprintContext(
            chapters=[dict(c) for c in baseline_run.get("chapters", [])],
            keyframes=[dict(k) for k in baseline_run.get("keyframes", [])],
        )
        ctx = apply_sprint_stack(
            ctx,
            stack,
            sprint4_cfg=RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
        )
        run = dict(baseline_run)
        run["chapters"] = ctx.chapters
        run["keyframes"] = ctx.keyframes
        metrics = compute_metrics(run)
        export_meta = ctx.stats.get("sprint10") or ctx.stats.get("export_meta") or {}

        ladder_results.append({
            "sprint_level": sprint_num,
            "stack": stack,
            "metrics": metrics,
            "sprint_stats": ctx.stats,
            "quality_score": export_meta.get("pipeline_quality_score"),
            "export_ready": export_meta.get("export_ready"),
        })

    out_dir = project_root / "outputs" / f"sprint_ladder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sprint_ladder.json"
    out_md = out_dir / "sprint_ladder.md"

    payload = {
        "created_at": datetime.now().isoformat(),
        "video": baseline_run.get("video_title", "TED"),
        "ladder_results": ladder_results,
        "final_sprint10": ladder_results[-1] if ladder_results else None,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Sprint Ladder 1 → 10 (TED offline)",
        "",
        "| Sprint | Ch | Min ch | KF | Coverage | Quality | Export ready |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in ladder_results:
        m = r["metrics"]
        qs = r.get("quality_score")
        er = r.get("export_ready")
        lines.append(
            f"| **S{r['sprint_level']}** | {m['chapter_count']} | {m['min_chapter_duration_sec']} | "
            f"{m['keyframe_count']} | {m['keyframe_script_coverage']} | "
            f"{qs if qs is not None else '-'} | {er if er is not None else '-'} |"
        )

    final = ladder_results[-1] if ladder_results else {}
    lines += ["", "## Sprint 10 export meta", ""]
    if final.get("sprint_stats", {}).get("sprint10"):
        for k, v in final["sprint_stats"]["sprint10"].items():
            lines.append(f"- **{k}**: `{v}`")

    lines += ["", "## Stack at Sprint 10", "", f"```python\n{SPRINT_LADDER[10]}\n```"]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"OUT_DIR={out_dir}")
    print(f"JSON={out_json}")
    for r in ladder_results:
        m = r["metrics"]
        print(
            f"S{r['sprint_level']}: ch={m['chapter_count']} kf={m['keyframe_count']} "
            f"cov={m['keyframe_script_coverage']} q={r.get('quality_score')}"
        )


if __name__ == "__main__":
    main()

"""Offline ablation for Sprint 1-6 stack variants (no pipeline re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.pipeline.sprints import RECOMMENDED_STACK, RECOMMENDED_STACK_CONFIG, RECOMMENDED_STACK_V1, FULL_STACK_S10, SprintContext, apply_sprint_stack
from experiments.scripts.offline_replay import offline_visual_gate, pick_run
from experiments.scripts.run_gated_compare import _recommendation, compute_metrics

PRESETS: dict[str, dict] = {
    "baseline": {"stack": []},
    "s1": {"stack": ["sprint1"]},
    "s1_s3": {"stack": ["sprint1", "sprint3"]},
    "s1_s3_s4": {
        "stack": ["sprint1", "sprint3", "sprint4"],
        "sprint4_cfg": {"window_sec": 45.0, "max_per_window": 2, "min_evidence_score": 1.0},
    },
    "s1_s3_s4_s5": {
        "stack": ["sprint1", "sprint3", "sprint4", "sprint5"],
        "sprint4_cfg": {"window_sec": 45.0, "max_per_window": 2, "min_evidence_score": 1.0},
    },
    "s1_s2c_s3": {
        "stack": ["sprint1", "sprint2", "sprint3"],
        "visual_cfg": {"hist_thresh": 0.95, "min_keep_ratio": 0.40},
    },
    "s1_s3_s4v2_s7": {
        "stack": ["sprint1", "sprint3", "sprint4_v2", "sprint7"],
        "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
    },
    "recommended_s6_v1": {
        "stack": RECOMMENDED_STACK_V1,
        "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
    },
    "recommended_s6": {
        "stack": RECOMMENDED_STACK,
        "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
    },
    "sprint10": {
        "stack": FULL_STACK_S10,
        "sprint4_cfg": RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
    },
    "recommended_s6_plus_s2c": {
        "stack": ["sprint1", "sprint2", "sprint3", "sprint4", "sprint5", "sprint8_soft"],
        "visual_cfg": {"hist_thresh": 0.95, "min_keep_ratio": 0.40},
        "sprint4_cfg": {"window_sec": 60.0, "max_per_window": 2, "min_evidence_score": 0.8},
    },
}


def _score(m: dict, gated_m: dict, wall: float) -> float:
    s = 0.0
    if m["min_chapter_duration_sec"] >= 45:
        s += 2
    if m["keyframe_script_coverage"] >= gated_m["keyframe_script_coverage"]:
        s += 2
    if m["keyframe_count"] >= gated_m["keyframe_count"] * 0.85:
        s += 1.5
    if m["chapter_count"] >= gated_m["chapter_count"]:
        s += 1
    if m["elapsed_wall_sec"] <= wall * 1.05:
        s += 1
    return round(s, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/ted_compare_20260806_162755/compare_report.json")
    parser.add_argument("--gated", default="outputs/gated_compare_20260807_143515/compare_report.json")
    args = parser.parse_args()

    baseline_run = pick_run(
        json.loads((project_root / args.baseline).read_text(encoding="utf-8")),
        "backend_parity", "backend_baseline",
    )
    if not baseline_run:
        raise SystemExit("No baseline run.")

    gated_run = None
    gated_path = project_root / args.gated
    if gated_path.is_file():
        gated_run = pick_run(
            json.loads(gated_path.read_text(encoding="utf-8")),
            "experimental_gated", "experimental_video",
        )

    wall = float(baseline_run.get("elapsed_wall_sec") or 147.0)
    results = []

    for name, preset in PRESETS.items():
        ctx = SprintContext(
            chapters=[dict(c) for c in baseline_run.get("chapters", [])],
            keyframes=[dict(k) for k in baseline_run.get("keyframes", [])],
        )

        stack = preset.get("stack", [])
        ctx = apply_sprint_stack(
            ctx,
            stack,
            visual_gate_fn=offline_visual_gate,
            project_root=project_root,
            visual_cfg=preset.get("visual_cfg"),
            sprint4_cfg=preset.get("sprint4_cfg"),
            caption_min_grounding=preset.get("caption_min_grounding", 0.10),
        )

        run = dict(baseline_run)
        run["chapters"] = ctx.chapters
        run["keyframes"] = ctx.keyframes
        run["sprint_stack"] = stack
        run["sprint_stats"] = ctx.stats

        metrics = compute_metrics(run)
        entry = {"preset": name, "stack": stack, "metrics": metrics, "stats": ctx.stats}
        if gated_run:
            gated_m = compute_metrics(gated_run)
            entry["score_vs_gated"] = _score(metrics, gated_m, wall)
            entry["recommendation_vs_gated"] = _recommendation(metrics, gated_m)
        results.append(entry)

    results.sort(key=lambda x: x.get("score_vs_gated", 0), reverse=True)
    best = results[0]

    out_dir = project_root / "outputs" / f"offline_sprint_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(),
        "best_preset": best["preset"],
        "best_stack": best["stack"],
        "recommended_stack_constant": RECOMMENDED_STACK,
        "results": results,
    }
    out_json = out_dir / "sprint_ablation.json"
    out_md = out_dir / "sprint_ablation.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Sprint Ablation (S1–S6)",
        "",
        f"**Best preset:** `{best['preset']}`",
        f"**Stack:** `{best['stack']}`",
        "",
        "| Rank | Preset | Ch | Min ch | KF | Coverage | Score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        lines.append(
            f"| {i} | `{r['preset']}` | {m['chapter_count']} | {m['min_chapter_duration_sec']} | "
            f"{m['keyframe_count']} | {m['keyframe_script_coverage']} | {r.get('score_vs_gated', '-')} |"
        )

    bs = best.get("stats", {})
    lines += ["", "## Best preset sprint stats", ""]
    for k, v in bs.items():
        lines.append(f"- **{k}**: `{v}`")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")
    print(f"BEST={best['preset']} stack={best['stack']} score={best.get('score_vs_gated')}")


if __name__ == "__main__":
    main()

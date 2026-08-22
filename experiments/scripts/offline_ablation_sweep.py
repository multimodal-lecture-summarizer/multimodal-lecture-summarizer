"""Offline ablation sweep: baseline + Sprint variants vs gated (no pipeline re-run)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.scripts.offline_replay import apply_variant, pick_run  # noqa: E402
from experiments.scripts.run_gated_compare import compute_metrics, _recommendation  # noqa: E402

VARIANTS = [
    ("baseline", {}),
    ("s1_chapters", {"sprint1": True}),
    ("s1_s2_visual_default", {"sprint1": True, "sprint2": True}),
    ("s1_s2_conservative", {
        "sprint1": True,
        "sprint2": True,
        "visual_cfg": {"hist_thresh": 0.95, "min_keep_ratio": 0.40},
    }),
    ("s1_s2_aggressive", {
        "sprint1": True,
        "sprint2": True,
        "visual_cfg": {"hist_thresh": 0.88, "min_keep_ratio": 0.20},
    }),
    ("s1_s2_s3_ocr_enrich", {"sprint1": True, "sprint2": True, "sprint3": True}),
    ("s1_s2_s3_s8_soft", {
        "sprint1": True,
        "sprint2": True,
        "sprint3": True,
        "sprint8_soft": True,
        "caption_min_grounding": 0.10,
    }),
    ("s1_s2_s3_s8_strict", {
        "sprint1": True,
        "sprint2": True,
        "sprint3": True,
        "sprint8_soft": True,
        "caption_min_grounding": 0.22,
    }),
]


def _score_vs_gated(variant_m: dict, gated_m: dict, baseline_wall: float) -> float:
    """Higher is better. Favor chapter stability, coverage, keyframes, speed."""
    s = 0.0
    if variant_m["min_chapter_duration_sec"] >= 45:
        s += 2
    if variant_m["chapter_count"] >= gated_m["chapter_count"]:
        s += 1
    if variant_m["keyframe_script_coverage"] >= gated_m["keyframe_script_coverage"]:
        s += 2
    if variant_m["keyframe_count"] >= gated_m["keyframe_count"]:
        s += 1.5
    if variant_m["elapsed_wall_sec"] <= baseline_wall * 1.05:
        s += 1
    if variant_m["summary_lexical_diversity"] >= gated_m["summary_lexical_diversity"] * 0.95:
        s += 0.5
    return round(s, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="outputs/ted_compare_20260806_162755/compare_report.json")
    parser.add_argument("--gated", default="outputs/gated_compare_20260807_143515/compare_report.json")
    args = parser.parse_args()

    baseline_path = project_root / args.baseline
    gated_path = project_root / args.gated

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_run = pick_run(baseline_payload, "backend_parity", "backend_baseline")
    if not baseline_run:
        raise SystemExit("No baseline run found.")

    gated_run = None
    if gated_path.is_file():
        gated_payload = json.loads(gated_path.read_text(encoding="utf-8"))
        gated_run = pick_run(gated_payload, "experimental_gated", "experimental_video")

    results = []
    for name, cfg in VARIANTS:
        variant_run = apply_variant(baseline_run, project_root, **cfg)
        metrics = compute_metrics(variant_run)
        entry = {
            "variant": name,
            "config": cfg,
            "metrics": metrics,
            "stats": variant_run.get("variant_stats"),
        }
        if gated_run:
            gated_m = compute_metrics(gated_run)
            entry["score_vs_gated"] = _score_vs_gated(metrics, gated_m, float(baseline_run.get("elapsed_wall_sec") or 147))
            entry["recommendation_vs_gated"] = _recommendation(metrics, gated_m)
        results.append(entry)

    results.sort(key=lambda x: x.get("score_vs_gated", 0), reverse=True)
    best = results[0]

    out_dir = project_root / "outputs" / f"offline_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": datetime.now().isoformat(),
        "comparison_type": "offline_ablation_sweep",
        "sources": {"baseline": str(baseline_path), "gated": str(gated_path)},
        "variants_tested": len(results),
        "best_variant": best["variant"],
        "best_score_vs_gated": best.get("score_vs_gated"),
        "results": results,
    }

    out_json = out_dir / "ablation_report.json"
    out_md = out_dir / "ablation_report.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Offline Ablation Sweep (TED)",
        "",
        "Không chạy lại pipeline. Sweep Sprint-1/2/3/8 trên baseline artifacts.",
        "",
        f"**Best variant:** `{best['variant']}` (score vs gated: {best.get('score_vs_gated')})",
        "",
        "## Leaderboard",
        "",
        "| Rank | Variant | Ch | Min ch(s) | KF | KF cov | Wall | Score |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(results, 1):
        m = r["metrics"]
        md.append(
            f"| {i} | `{r['variant']}` | {m['chapter_count']} | {m['min_chapter_duration_sec']} | "
            f"{m['keyframe_count']} | {m['keyframe_script_coverage']} | {m['elapsed_wall_sec']} | "
            f"{r.get('score_vs_gated', '-')} |"
        )

    md += ["", "## Best variant details", ""]
    bs = best.get("stats") or {}
    if bs.get("visual") and not bs["visual"].get("skipped"):
        v = bs["visual"]
        md.append(f"- Visual: `{v.get('input')}` → `{v.get('output')}`, OCR-saved `{v.get('ocr_saved_slides')}`")
    if bs.get("enrich") and not bs["enrich"].get("skipped"):
        e = bs["enrich"]
        md.append(f"- OCR enrich generic captions: `{e.get('enriched_from_ocr')}`, kept generic `{e.get('kept_generic')}`")
    if bs.get("caption") and not bs["caption"].get("skipped"):
        c = bs["caption"]
        md.append(
            f"- Caption soft: replaced `{c.get('soft_replaced')}`, flagged kept `{c.get('soft_flagged_kept')}`, "
            f"avg grounding `{c.get('avg_grounding_score')}`"
        )

    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"OUT_DIR={out_dir}")
    print(f"BEST={best['variant']} score={best.get('score_vs_gated')}")
    for r in results[:3]:
        m = r["metrics"]
        print(
            f"  {r['variant']}: ch={m['chapter_count']} kf={m['keyframe_count']} "
            f"cov={m['keyframe_script_coverage']} score={r.get('score_vs_gated')}"
        )


if __name__ == "__main__":
    main()

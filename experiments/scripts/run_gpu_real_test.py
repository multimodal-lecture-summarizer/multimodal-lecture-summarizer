"""Real GPU pipeline test: baseline + Sprint-10 stack on a real video."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def pick_video(video_arg: str | None) -> Path:
    if video_arg:
        p = Path(video_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Video not found: {video_arg}")
    preferred = Path(r"D:\datasets\TEDLIUM\videos\Blaise_Agueray_Arcas.mp4")
    if preferred.is_file():
        return preferred
    fallback = project_root / "experiments" / "notebooks" / "demo_data" / "sample.mp4"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("No video found. Pass --video PATH.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None)
    parser.add_argument("--stack", default="sprint10", choices=("recommended", "sprint10"))
    parser.add_argument("--skip-sprint", action="store_true", help="Only run baseline GPU pipeline")
    args = parser.parse_args()

    os.chdir(project_root)
    # Ensure GPU is visible before any torch/CUDA imports.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["FLORENCE_DEVICE"] = "cuda"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["USE_GPU"] = "1"

    import torch
    from experiments.pipeline.orchestrator import run_backend_baseline
    from experiments.pipeline.sprints import (
        FULL_STACK_S10,
        RECOMMENDED_STACK,
        RECOMMENDED_STACK_CONFIG,
        SprintContext,
        apply_sprint_stack,
    )
    from experiments.scripts.run_gated_compare import compute_metrics, pipeline_to_run_dict

    print(f"[CUDA] available={torch.cuda.is_available()} count={torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA not available — aborting GPU real test.")

    video_path = pick_video(args.video)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / "outputs" / f"gpu_real_test_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    stack = FULL_STACK_S10 if args.stack == "sprint10" else RECOMMENDED_STACK
    device_name = torch.cuda.get_device_name(0)

    print("=" * 70)
    print(f"GPU REAL TEST | device={device_name}")
    print("mode=hybrid (Florence/CLIP CUDA, Faster-Whisper CPU - cuDNN8/9 mismatch)")
    print(f"video={video_path.name}")
    print(f"stack={stack}")
    print("=" * 70)

    print("\n[1/2] Running backend baseline on GPU...")
    baseline_pr = run_backend_baseline(
        str(video_path),
        str(out_dir),
        job_id=f"gpu_baseline_{run_id}",
        use_gpu=True,
    )
    baseline_run = pipeline_to_run_dict(baseline_pr)
    baseline_run["video_title"] = video_path.stem
    baseline_run["config_stack"] = "backend_baseline_gpu"
    baseline_metrics = compute_metrics(baseline_run)
    print(
        f"  baseline: ch={len(baseline_pr.chapters)} kf={len(baseline_pr.keyframes)} "
        f"wall={baseline_pr.wall_time_sec}s cov={baseline_metrics['keyframe_script_coverage']}"
    )

    sprint_run = None
    sprint_metrics = None
    sprint_stats = None
    if not args.skip_sprint:
        print(f"\n[2/2] Applying {args.stack} stack on live GPU output...")
        ctx = SprintContext(
            chapters=[dict(c) for c in baseline_pr.chapters],
            keyframes=[dict(k) for k in baseline_pr.keyframes],
            utterances=[dict(u) for u in (baseline_pr.utterances or [])],
        )
        ctx = apply_sprint_stack(
            ctx,
            stack,
            sprint4_cfg=RECOMMENDED_STACK_CONFIG["sprint4_cfg"],
        )
        sprint_run = dict(baseline_run)
        sprint_run["chapters"] = ctx.chapters
        sprint_run["keyframes"] = ctx.keyframes
        sprint_run["config_stack"] = f"{args.stack}_on_gpu_baseline"
        sprint_run["sprint_stack"] = stack
        sprint_run["sprint_stats"] = ctx.stats
        sprint_stats = ctx.stats
        sprint_metrics = compute_metrics(sprint_run)
        export_meta = ctx.stats.get("sprint10") or ctx.stats.get("export_meta") or {}
        print(
            f"  sprint: ch={sprint_metrics['chapter_count']} kf={sprint_metrics['keyframe_count']} "
            f"cov={sprint_metrics['keyframe_script_coverage']} "
            f"q={export_meta.get('pipeline_quality_score')} "
            f"ready={export_meta.get('export_ready')}"
        )
        if ctx.stats.get("sprint3"):
            print(f"  sprint3 OCR enrich: {ctx.stats['sprint3']}")
        if ctx.stats.get("sprint7"):
            print(f"  sprint7 transcript enrich: {ctx.stats['sprint7']}")

    payload = {
        "created_at": datetime.now().isoformat(),
        "video_path": str(video_path),
        "device": device_name,
        "cuda_available": True,
        "gpu_mode": "hybrid_florence_cuda_whisper_cpu",
        "stack": stack,
        "baseline": {
            "metrics": baseline_metrics,
            "run": baseline_run,
            "gate_reports": baseline_pr.gate_reports,
        },
        "sprint": None
        if sprint_run is None
        else {
            "metrics": sprint_metrics,
            "stats": sprint_stats,
            "export_meta": (sprint_stats or {}).get("sprint10")
            or (sprint_stats or {}).get("export_meta"),
            "run": {
                **{k: v for k, v in sprint_run.items() if k != "sprint_stats"},
                "sprint_stats": sprint_stats,
            },
        },
    }

    out_json = out_dir / "gpu_real_test.json"
    out_md = out_dir / "gpu_real_test.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    bm = baseline_metrics
    lines = [
        "# GPU Real Test",
        "",
        f"- **Device**: `{device_name}`",
        f"- **Video**: `{video_path.name}`",
        f"- **Stack**: `{args.stack}` → `{stack}`",
        "",
        "## Metrics",
        "",
        "| Variant | Ch | Min ch | KF | Coverage | Wall(s) | Quality | Export |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| Baseline GPU | {bm['chapter_count']} | {bm['min_chapter_duration_sec']} | "
            f"{bm['keyframe_count']} | {bm['keyframe_script_coverage']} | {bm['elapsed_wall_sec']} | - | - |"
        ),
    ]
    if sprint_metrics is not None:
        sm = sprint_metrics
        em = (sprint_stats or {}).get("sprint10") or {}
        lines.append(
            f"| {args.stack} on GPU | {sm['chapter_count']} | {sm['min_chapter_duration_sec']} | "
            f"{sm['keyframe_count']} | {sm['keyframe_script_coverage']} | {bm['elapsed_wall_sec']} | "
            f"{em.get('pipeline_quality_score', '-')} | {em.get('export_ready', '-')} |"
        )
        lines += ["", "## Sprint stats", ""]
        for name, stats in (sprint_stats or {}).items():
            lines.append(f"- **{name}**: `{stats}`")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nOUT_DIR={out_dir}")
    print(f"JSON={out_json}")
    print(f"MD={out_md}")


if __name__ == "__main__":
    main()

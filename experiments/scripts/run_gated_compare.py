"""Compare backend baseline vs experimental gated pipeline (issues 5/6/7/8)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path


def pick_video(project_root: Path, video_arg: str | None) -> Path:
    if video_arg:
        p = Path(video_arg)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Video not found: {video_arg}")

    ted_dir = Path(r"D:\datasets\TEDLIUM\videos")
    preferred = ted_dir / "Blaise_Agueray_Arcas.mp4"
    fallback = project_root / "experiments" / "notebooks" / "demo_data" / "sample.mp4"
    if preferred.is_file():
        return preferred
    if ted_dir.is_dir():
        mp4s = sorted(ted_dir.glob("*.mp4"), key=lambda p: p.stat().st_size)
        if mp4s:
            return mp4s[0]
    return fallback


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _chapter_duration(ch: dict) -> float:
    start = float(ch.get("startTime", ch.get("start_time", ch.get("start_seconds", 0.0))) or 0.0)
    end = float(ch.get("endTime", ch.get("end_time", ch.get("end_seconds", start))) or start)
    return max(0.0, end - start)


def pipeline_to_run_dict(pr) -> dict:
    return {
        "job_id": pr.job_id,
        "config_stack": "gated_experimental" if pr.enable_gates else "backend_baseline",
        "elapsed_wall_sec": pr.wall_time_sec,
        "video_title": "",
        "model_used": pr.model_used,
        "duration": pr.duration,
        "processing_time": pr.wall_time_sec,
        "chapters": pr.chapters,
        "keyframes": pr.keyframes,
        "summary": pr.summary,
        "gate_reports": pr.gate_reports,
    }


def compute_metrics(run: dict) -> dict:
    chapters = run.get("chapters", [])
    keyframes = run.get("keyframes", [])
    summary = run.get("summary", "")
    duration = float(run.get("duration") or 0.0)
    wall = float(run.get("elapsed_wall_sec") or 0.0)

    summary_tokens = _tokenize(summary)
    summary_unique = len(set(summary_tokens))
    summary_len = len(summary_tokens)

    chapter_durations = [_chapter_duration(ch) for ch in chapters]
    avg_chapter_dur = _safe_div(sum(chapter_durations), len(chapter_durations))
    min_chapter_dur = min(chapter_durations) if chapter_durations else 0.0
    max_chapter_dur = max(chapter_durations) if chapter_durations else 0.0

    non_empty_scripts = sum(1 for kf in keyframes if (kf.get("transcript") or "").strip())
    keyframe_script_coverage = _safe_div(non_empty_scripts, len(keyframes))
    summary_wpm = _safe_div(summary_len * 60.0, duration)
    speed_factor = _safe_div(duration, wall)

    gates = run.get("gate_reports") or {}
    return {
        "chapter_count": len(chapters),
        "keyframe_count": len(keyframes),
        "summary_word_count": summary_len,
        "summary_lexical_diversity": round(_safe_div(summary_unique, summary_len), 4),
        "summary_words_per_video_min": round(summary_wpm, 2),
        "avg_chapter_duration_sec": round(avg_chapter_dur, 2),
        "min_chapter_duration_sec": round(min_chapter_dur, 2),
        "max_chapter_duration_sec": round(max_chapter_dur, 2),
        "keyframe_script_coverage": round(keyframe_script_coverage, 4),
        "pipeline_speed_factor": round(speed_factor, 4),
        "elapsed_wall_sec": round(wall, 2),
        "duration_sec": round(duration, 2),
        "asr_quality_status": gates.get("asr", {}).get("quality_status"),
        "speaker_reliability": gates.get("speaker", {}).get("reliability"),
        "visual_gate_output": gates.get("visual", {}).get("output"),
        "caption_hallucinations": gates.get("caption", {}).get("hallucination_count"),
        "avg_grounding_score": gates.get("caption", {}).get("avg_grounding_score"),
    }


def _recommendation(backend_m: dict, exp_m: dict) -> str:
    score = 0
    if exp_m["chapter_count"] >= backend_m["chapter_count"]:
        score += 1
    if exp_m["summary_word_count"] >= backend_m["summary_word_count"]:
        score += 1
    if exp_m["keyframe_script_coverage"] >= backend_m["keyframe_script_coverage"]:
        score += 1
    if exp_m["elapsed_wall_sec"] <= backend_m["elapsed_wall_sec"]:
        score += 1
    return "experimental_gated" if score >= 3 else "backend_baseline"


def fmt_chapter_titles(chapters: list[dict]) -> str:
    if not chapters:
        return "(none)"
    return "; ".join(c.get("title", "Untitled") for c in chapters[:5])


def main() -> None:
    # Force CPU before any torch/CUDA imports in downstream modules.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLORENCE_DEVICE"] = "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to input video")
    parser.add_argument("--expected-lang", default="en")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--gated-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from experiments.pipeline.orchestrator import run_backend_baseline, run_gated_pipeline

    video_path = pick_video(project_root, args.video)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / "outputs" / f"gated_compare_{run_id}"
    output_root.mkdir(parents=True, exist_ok=True)

    backend_run = None
    gated_run = None

    if not args.gated_only:
        print("[1/2] Running backend baseline (process_video)...")
        backend_pr = run_backend_baseline(str(video_path), str(output_root), f"baseline_{run_id}")
        backend_run = pipeline_to_run_dict(backend_pr)
        print(f"  chapters={len(backend_pr.chapters)} keyframes={len(backend_pr.keyframes)} wall={backend_pr.wall_time_sec}s")

    if not args.baseline_only:
        print("[2/2] Running experimental gated pipeline (issues 5/6/7/8)...")
        gated_pr = run_gated_pipeline(
            str(video_path),
            str(output_root),
            enable_gates=True,
            expected_lang=args.expected_lang,
            job_id=f"gated_{run_id}",
        )
        gated_run = pipeline_to_run_dict(gated_pr)
        print(f"  chapters={len(gated_pr.chapters)} keyframes={len(gated_pr.keyframes)} wall={gated_pr.wall_time_sec}s")
        print(f"  gate reports: {json.dumps(gated_pr.gate_reports, ensure_ascii=False)}")

    report_json = output_root / "compare_report.json"
    report_md = output_root / "compare_report.md"

    payload: dict = {
        "video_path": str(video_path),
        "created_at": datetime.now().isoformat(),
        "comparison_type": "backend_vs_gated_experimental",
    }
    if backend_run:
        payload["backend_baseline"] = backend_run
    if gated_run:
        payload["experimental_gated"] = gated_run

    if backend_run and gated_run:
        backend_metrics = compute_metrics(backend_run)
        exp_metrics = compute_metrics(gated_run)
        payload["metrics"] = {
            "backend_baseline": backend_metrics,
            "experimental_gated": exp_metrics,
            "recommended_mode": _recommendation(backend_metrics, exp_metrics),
            "delta": {
                "chapters": exp_metrics["chapter_count"] - backend_metrics["chapter_count"],
                "keyframes": exp_metrics["keyframe_count"] - backend_metrics["keyframe_count"],
                "wall_time_sec": round(exp_metrics["elapsed_wall_sec"] - backend_metrics["elapsed_wall_sec"], 2),
            },
        }

    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [
        "# Gated Pipeline Compare Report",
        "",
        f"- Video: `{video_path}`",
        f"- Created: `{payload['created_at']}`",
        "",
    ]

    if backend_run:
        md += [
            "## Backend Baseline",
            f"- Job ID: `{backend_run['job_id']}`",
            f"- Chapters: `{len(backend_run['chapters'])}`",
            f"- Keyframes: `{len(backend_run['keyframes'])}`",
            f"- Wall time: `{backend_run['elapsed_wall_sec']}s`",
            f"- Chapter titles: {fmt_chapter_titles(backend_run['chapters'])}",
            "",
        ]

    if gated_run:
        md += [
            "## Experimental Gated (5/6/7/8)",
            f"- Job ID: `{gated_run['job_id']}`",
            f"- Chapters: `{len(gated_run['chapters'])}`",
            f"- Keyframes: `{len(gated_run['keyframes'])}`",
            f"- Wall time: `{gated_run['elapsed_wall_sec']}s`",
            f"- ASR status: `{gated_run.get('gate_reports', {}).get('asr', {}).get('quality_status')}`",
            f"- Speaker reliability: `{gated_run.get('gate_reports', {}).get('speaker', {}).get('reliability')}`",
            f"- Visual gate: `{gated_run.get('gate_reports', {}).get('visual')}`",
            f"- Caption gate: `{gated_run.get('gate_reports', {}).get('caption')}`",
            f"- Chapter titles: {fmt_chapter_titles(gated_run['chapters'])}",
            "",
        ]

    if payload.get("metrics"):
        bm = payload["metrics"]["backend_baseline"]
        em = payload["metrics"]["experimental_gated"]
        md += [
            "## Delta",
            f"- Delta chapters: `{payload['metrics']['delta']['chapters']}`",
            f"- Delta keyframes: `{payload['metrics']['delta']['keyframes']}`",
            f"- Delta wall time: `{payload['metrics']['delta']['wall_time_sec']}s`",
            "",
            "## Academic Metrics",
            "",
            "| Metric | Backend | Gated |",
            "|---|---:|---:|",
            f"| Chapter count | {bm['chapter_count']} | {em['chapter_count']} |",
            f"| Keyframe count | {bm['keyframe_count']} | {em['keyframe_count']} |",
            f"| Summary words | {bm['summary_word_count']} | {em['summary_word_count']} |",
            f"| Lexical diversity | {bm['summary_lexical_diversity']} | {em['summary_lexical_diversity']} |",
            f"| Keyframe-script coverage | {bm['keyframe_script_coverage']} | {em['keyframe_script_coverage']} |",
            f"| Wall time (s) | {bm['elapsed_wall_sec']} | {em['elapsed_wall_sec']} |",
            f"| Caption hallucinations flagged | - | {em.get('caption_hallucinations')} |",
            f"| Avg grounding score | - | {em.get('avg_grounding_score')} |",
            "",
            f"## Recommendation: `{payload['metrics']['recommended_mode']}`",
        ]

    report_md.write_text("\n".join(md), encoding="utf-8")

    print(f"REPORT_DIR={output_root}")
    print(f"REPORT_JSON={report_json}")
    print(f"REPORT_MD={report_md}")
    if payload.get("metrics"):
        print("RECOMMENDED_MODE=", payload["metrics"]["recommended_mode"])


if __name__ == "__main__":
    main()

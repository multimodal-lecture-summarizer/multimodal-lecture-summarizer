from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


def pick_ted_video(project_root: Path) -> Path:
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


def run_mode(process_video_task, job_id: str, video_path: Path, config_stack: str) -> dict:
    started = time.time()
    result = process_video_task.run(job_id, str(video_path), config_stack)
    elapsed = time.time() - started

    return {
        "job_id": job_id,
        "config_stack": config_stack,
        "elapsed_wall_sec": round(elapsed, 2),
        "video_title": result.get("video_title"),
        "model_used": result.get("model_used"),
        "duration": result.get("duration"),
        "processing_time": result.get("processing_time"),
        "chapters": result.get("chapters", []),
        "keyframes": result.get("keyframes", []),
        "summary": result.get("summary", ""),
    }


def fmt_chapter_titles(chapters: list[dict]) -> str:
    if not chapters:
        return "(none)"
    return "; ".join(c.get("title", "Untitled") for c in chapters[:5])


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _chapter_duration(ch: dict) -> float:
    start = float(ch.get("startTime", ch.get("start_time", ch.get("start_seconds", 0.0))) or 0.0)
    end = float(ch.get("endTime", ch.get("end_time", ch.get("end_seconds", start))) or start)
    return max(0.0, end - start)


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

    # Proxy: keyframes that contain aligned transcript text
    non_empty_scripts = sum(1 for kf in keyframes if (kf.get("transcript") or "").strip())
    keyframe_script_coverage = _safe_div(non_empty_scripts, len(keyframes))

    summary_wpm = _safe_div(summary_len * 60.0, duration)
    speed_factor = _safe_div(duration, wall)  # higher is faster processing throughput

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
        "processing_time_sec": round(float(run.get("processing_time") or 0.0), 2),
        "duration_sec": round(duration, 2),
    }


def _recommendation(backend_m: dict, exp_m: dict) -> str:
    score = 0
    # Prefer stable granularity (not too few chapters)
    if exp_m["chapter_count"] >= backend_m["chapter_count"]:
        score += 1
    # Prefer richer summary if not overly compressed
    if exp_m["summary_word_count"] >= backend_m["summary_word_count"]:
        score += 1
    # Prefer better coverage of keyframes by transcript
    if exp_m["keyframe_script_coverage"] >= backend_m["keyframe_script_coverage"]:
        score += 1
    # Prefer faster runtime
    if exp_m["elapsed_wall_sec"] <= backend_m["elapsed_wall_sec"]:
        score += 1
    return "experimental_video" if score >= 3 else "backend_parity"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Stability toggles for this environment:
    # - disable denoise to avoid cudnn_ops_infer64_8.dll failure
    # - keep storage/rag out of this comparison run
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["CF_R2_ACCESS_KEY_ID"] = ""
    os.environ["CF_R2_SECRET_ACCESS_KEY"] = ""

    from ai_workers.tasks import process_video
    # Monkey patch denoise step to avoid CUDA/cuDNN instability in this environment.
    from ai_workers.modules.common import denoise as denoise_mod
    import librosa
    import soundfile as sf

    def _read_raw_audio_safe(audio_path: str):
        data, sr = sf.read(audio_path)
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        return data.astype("float32")

    denoise_mod.get_denoised_audio_array = _read_raw_audio_safe

    # Running task logic directly (without Celery worker) has no task_id;
    # disable update_state side effects for offline experiment runs.
    process_video.update_state = lambda *args, **kwargs: None

    video_path = pick_ted_video(project_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    backend_job = f"ted_backend_{run_id}"
    exp_job = f"ted_experimental_{run_id}"

    backend = run_mode(
        process_video,
        job_id=backend_job,
        video_path=video_path,
        config_stack="hybrid_no_rag_no_storage",
    )
    experimental = run_mode(
        process_video,
        job_id=exp_job,
        video_path=video_path,
        config_stack="experimental_video_no_rag_no_storage",
    )

    report_dir = project_root / "outputs" / f"ted_compare_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_json = report_dir / "compare_report.json"
    report_md = report_dir / "compare_report.md"

    payload = {
        "video_path": str(video_path),
        "created_at": datetime.now().isoformat(),
        "backend_parity": backend,
        "experimental_video": experimental,
    }
    backend_metrics = compute_metrics(backend)
    exp_metrics = compute_metrics(experimental)
    payload["metrics"] = {
        "backend_parity": backend_metrics,
        "experimental_video": exp_metrics,
        "recommended_mode": _recommendation(backend_metrics, exp_metrics),
    }
    report_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    md = []
    md.append("# TED Pipeline Compare Report")
    md.append("")
    md.append(f"- Video: `{video_path}`")
    md.append(f"- Created: `{payload['created_at']}`")
    md.append("")
    md.append("## Backend Parity")
    md.append(f"- Job ID: `{backend['job_id']}`")
    md.append(f"- Model: `{backend['model_used']}`")
    md.append(f"- Duration: `{backend['duration']}`")
    md.append(f"- Processing Time (pipeline): `{backend['processing_time']}`")
    md.append(f"- Wall Time (script): `{backend['elapsed_wall_sec']}`")
    md.append(f"- Chapters: `{len(backend['chapters'])}`")
    md.append(f"- Keyframes: `{len(backend['keyframes'])}`")
    md.append(f"- Chapter titles: {fmt_chapter_titles(backend['chapters'])}")
    md.append("")
    md.append("## Experimental Video")
    md.append(f"- Job ID: `{experimental['job_id']}`")
    md.append(f"- Model: `{experimental['model_used']}`")
    md.append(f"- Duration: `{experimental['duration']}`")
    md.append(f"- Processing Time (pipeline): `{experimental['processing_time']}`")
    md.append(f"- Wall Time (script): `{experimental['elapsed_wall_sec']}`")
    md.append(f"- Chapters: `{len(experimental['chapters'])}`")
    md.append(f"- Keyframes: `{len(experimental['keyframes'])}`")
    md.append(f"- Chapter titles: {fmt_chapter_titles(experimental['chapters'])}")
    md.append("")
    md.append("## Delta")
    md.append(f"- Delta chapters: `{len(experimental['chapters']) - len(backend['chapters'])}`")
    md.append(f"- Delta keyframes: `{len(experimental['keyframes']) - len(backend['keyframes'])}`")
    md.append("")
    md.append("## Academic Evaluation Metrics")
    md.append("")
    md.append("| Metric | Backend | Experimental |")
    md.append("|---|---:|---:|")
    md.append(f"| Chapter count | {backend_metrics['chapter_count']} | {exp_metrics['chapter_count']} |")
    md.append(f"| Keyframe count | {backend_metrics['keyframe_count']} | {exp_metrics['keyframe_count']} |")
    md.append(f"| Summary words | {backend_metrics['summary_word_count']} | {exp_metrics['summary_word_count']} |")
    md.append(f"| Lexical diversity | {backend_metrics['summary_lexical_diversity']} | {exp_metrics['summary_lexical_diversity']} |")
    md.append(f"| Avg chapter duration (s) | {backend_metrics['avg_chapter_duration_sec']} | {exp_metrics['avg_chapter_duration_sec']} |")
    md.append(f"| Min chapter duration (s) | {backend_metrics['min_chapter_duration_sec']} | {exp_metrics['min_chapter_duration_sec']} |")
    md.append(f"| Max chapter duration (s) | {backend_metrics['max_chapter_duration_sec']} | {exp_metrics['max_chapter_duration_sec']} |")
    md.append(f"| Keyframe-script coverage | {backend_metrics['keyframe_script_coverage']} | {exp_metrics['keyframe_script_coverage']} |")
    md.append(f"| Summary words / video min | {backend_metrics['summary_words_per_video_min']} | {exp_metrics['summary_words_per_video_min']} |")
    md.append(f"| Wall time (s) | {backend_metrics['elapsed_wall_sec']} | {exp_metrics['elapsed_wall_sec']} |")
    md.append("")
    md.append("## Methodological Notes")
    md.append("- Same input video and same LLM provider/model for both runs.")
    md.append("- RAG indexing and storage upload are disabled for fair video-pipeline comparison.")
    md.append("- Metrics are automatic proxies; human rubric scoring is still required before production promotion.")
    md.append("")
    md.append("## Recommendation")
    md.append(f"- Recommended mode (current run): `{payload['metrics']['recommended_mode']}`")
    md.append("")
    md.append("## Summary Preview")
    md.append("")
    md.append("### Backend")
    md.append((backend["summary"] or "")[:700] + "...")
    md.append("")
    md.append("### Experimental")
    md.append((experimental["summary"] or "")[:700] + "...")
    md.append("")

    report_md.write_text("\n".join(md), encoding="utf-8")

    print(f"REPORT_DIR={report_dir}")
    print(f"REPORT_MD={report_md}")
    print(f"REPORT_JSON={report_json}")
    print("BACKEND_CHAPTERS=", len(backend["chapters"]))
    print("EXPERIMENTAL_CHAPTERS=", len(experimental["chapters"]))
    print("BACKEND_KEYFRAMES=", len(backend["keyframes"]))
    print("EXPERIMENTAL_KEYFRAMES=", len(experimental["keyframes"]))
    print("RECOMMENDED_MODE=", payload["metrics"]["recommended_mode"])


if __name__ == "__main__":
    main()

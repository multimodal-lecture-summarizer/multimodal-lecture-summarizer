#!/usr/bin/env python3
"""Run thesis evaluation tables (Bang 1–12) and write EVAL_TABLES.md.

Usage examples:
  python experiments/scripts/run_eval_tables.py --dry-report
  python experiments/scripts/run_eval_tables.py --manifest benchmarks/manifest_eval.csv
  python experiments/scripts/run_eval_tables.py --asr-only --asr-models base.en small.en
  python experiments/scripts/run_eval_tables.py --stages asr,scene,chapter --limit 2

Manifest columns (optional refs relative to benchmarks/ or absolute):
  video_id,video_path,dataset,language,
  reference_transcript,reference_speech,reference_scenes,reference_keyframes,
  reference_ocr,reference_captions,reference_alignments,reference_chapters,
  reference_summary,reference_qa
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluation.datasets import (
    load_tedlium_rows,
    pick_ted_lecture_videos,
    pick_ted_vad_items,
    pick_tedlium_asr_items,
    pick_tvsum_videos,
    tvsum_scene_boundaries,
)
from experiments.evaluation.metrics import aggregate_caption_scores, mean_ignore_nan
from experiments.evaluation.report import write_report
from experiments.evaluation.runners import (
    eval_asr_file,
    eval_captions,
    eval_chapters,
    eval_keyframe_filter,
    eval_ocr_items,
    eval_scene_boundaries,
    eval_scene_video,
    eval_summary_pair,
    eval_ted_pending_stages,
    eval_ted_timeline_chapter,
    eval_timeline_alignment,
    eval_tvsum_keyframe_video,
    eval_vad_from_segments,
    load_ref_if_exists,
    predict_vad_energy,
)
from experiments.evaluation.schemas import normalize_intervals


def _resolve(path_str: str | None, base: Path) -> Path | None:
    if not path_str or str(path_str).strip() in {"", "-", "None", "nan"}:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p if p.exists() else p  # may not exist yet


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def conclude(metric_name: str, values: list[float], *, higher_better: bool = True, good: float = 0.8) -> str:
    nums = [v for v in values if v is not None and v == v]
    if not nums:
        return f"_Kết luận: chưa có số liệu {metric_name} — cần annotate / chạy stage._"
    avg = sum(nums) / len(nums)
    if higher_better:
        status = "ổn" if avg >= good else "cần cải thiện"
    else:
        status = "ổn" if avg <= good else "cần cải thiện"
    return f"_Kết luận: {metric_name} trung bình = {avg:.3f} → stage **{status}**._"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    bench_root = Path(args.benchmarks_root).resolve()
    results: dict[str, Any] = {
        "asr": [],
        "vad": [],
        "scene": [],
        "keyframe": [],
        "ocr": [],
        "caption": [],
        "timeline": [],
        "chapter": [],
        "summary": [],
        "rag": [],
        "ablation": [],
        "stage_status": {},
    }
    stages = {s.strip().lower() for s in args.stages.split(",") if s.strip()}
    if "all" in stages:
        stages = {
            "asr",
            "vad",
            "scene",
            "keyframe",
            "ocr",
            "caption",
            "timeline",
            "chapter",
            "summary",
            "ablation",
        }

    rows = []
    if args.manifest and not args.auto_datasets:
        rows = read_manifest(Path(args.manifest))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

    # --- ASR ---
    if "asr" in stages:
        asr_models = [m.strip() for m in args.asr_models.split(",") if m.strip()]
        if args.auto_datasets:
            items = pick_tedlium_asr_items(limit=args.asr_limit or 6)
            print(f"[ASR] TED-LIUM clips: {len(items)}")
            for it in items:
                for model in asr_models:
                    try:
                        scored = eval_asr_file(
                            it["wav_path"],
                            it["text"],
                            model_size=model,
                            language="en",
                        )
                        scored["dataset"] = f"TED-LIUM:{it['speaker_id']}"
                        scored["model"] = f"faster-whisper-{model}"
                        results["asr"].append(scored)
                        print(
                            f"[ASR] {it['id']} {model} WER={scored['wer']:.3f} RTF={scored['rtf']:.3f}"
                        )
                    except Exception as e:
                        print(f"[ASR][WARN] {it['id']} {model}: {e}")
        else:
            for row in rows:
                vid = row.get("video_id") or Path(row.get("video_path", "video")).stem
                video_path = _resolve(row.get("video_path"), bench_root)
                ref_path = _resolve(row.get("reference_transcript"), bench_root)
                if not video_path or not video_path.exists() or not ref_path or not ref_path.exists():
                    continue
                ref_text = ref_path.read_text(encoding="utf-8")
                dataset = row.get("dataset") or "lecture"
                for model in asr_models:
                    try:
                        scored = eval_asr_file(
                            video_path,
                            ref_text,
                            model_size=model,
                            language=row.get("language") or "en",
                        )
                        scored["dataset"] = f"{dataset}:{vid}"
                        results["asr"].append(scored)
                        print(f"[ASR] {vid} {model} WER={scored['wer']:.3f} RTF={scored['rtf']:.3f}")
                    except Exception as e:
                        print(f"[ASR][WARN] {vid} {model}: {e}")
        results["stage_status"]["asr"] = "done" if results["asr"] else "pending"
        results["asr_conclusion"] = conclude(
            "WER",
            [r["wer"] for r in results["asr"]],
            higher_better=False,
            good=0.20,
        )

    # --- VAD ---
    if "vad" in stages:
        if args.auto_datasets:
            for item in pick_ted_vad_items(limit=2):
                print(f"[VAD] TED-LIUM talk={item['video_id']} media={item['media_path'].name}")
                try:
                    pred = predict_vad_energy(item["media_path"])
                    results["vad"].append(
                        eval_vad_from_segments(
                            pred,
                            item["intervals"],
                            video=f"TED-LIUM:{item['video_id']}",
                        )
                    )
                except Exception as e:
                    print(f"[VAD][WARN] {item['video_id']}: {e}")
        else:
            for row in rows:
                vid = row.get("video_id") or "video"
                ref_path = _resolve(row.get("reference_speech"), bench_root)
                pred_path = _resolve(row.get("pred_speech"), bench_root)
                if not ref_path or not ref_path.exists():
                    continue
                ref = normalize_intervals(load_ref_if_exists(ref_path))
                pred = normalize_intervals(load_ref_if_exists(pred_path)) if pred_path and pred_path.exists() else []
                if not pred:
                    continue
                dur = float(row["duration_sec"]) if row.get("duration_sec") else None
                results["vad"].append(eval_vad_from_segments(pred, ref, duration_sec=dur, video=vid))
        results["stage_status"]["vad"] = "done" if results["vad"] else "pending"
        results["vad_conclusion"] = conclude("F1", [r.get("f1", float("nan")) for r in results["vad"]])

    # --- Scene ---
    if "scene" in stages:
        if args.auto_datasets:
            tv_items = pick_tvsum_videos(limit=args.tvsum_limit or 2)
            print(f"[Scene] TVSum videos: {[x['video_id'] for x in tv_items]}")
            for it in tv_items:
                try:
                    import cv2

                    cap = cv2.VideoCapture(str(it["video_path"]))
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
                    cap.release()
                    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

                    det = SceneDetector({"threshold": 27.0})
                    scenes = det.detect_scenes(str(it["video_path"]))
                    pred = [s["start_seconds"] for s in scenes if s.get("start_seconds", 0) > 0]
                    ref = tvsum_scene_boundaries(it["scores"], fps=fps)
                    scored = eval_scene_boundaries(
                        pred, ref, tolerance_sec=max(args.scene_tol, 2.0), video=f"TVSum:{it['video_id']}"
                    )
                    results["scene"].append(scored)
                    print(f"[Scene] {it['video_id']} F1={scored['f1']:.3f} pred={len(pred)} ref={len(ref)}")
                except Exception as e:
                    print(f"[Scene][WARN] {it['video_id']}: {e}")
        else:
            for row in rows:
                vid = row.get("video_id") or Path(row.get("video_path", "video")).stem
                video_path = _resolve(row.get("video_path"), bench_root)
                ref_path = _resolve(row.get("reference_scenes"), bench_root)
                if not video_path or not video_path.exists() or not ref_path or not ref_path.exists():
                    continue
                ref_payload = load_ref_if_exists(ref_path)
                try:
                    scored = eval_scene_video(video_path, ref_payload, tolerance_sec=args.scene_tol)
                    scored["video"] = vid
                    results["scene"].append(scored)
                    print(f"[Scene] {vid} F1={scored['f1']:.3f}")
                except Exception as e:
                    print(f"[Scene][WARN] {vid}: {e}")
        results["stage_status"]["scene"] = "done" if results["scene"] else "pending"
        results["scene_conclusion"] = conclude("F1", [r.get("f1", float("nan")) for r in results["scene"]])

    # --- Keyframe ---
    if "keyframe" in stages:
        if args.auto_datasets:
            tv_items = pick_tvsum_videos(limit=args.tvsum_limit or 2)
            print(f"[Keyframe] TVSum videos: {[x['video_id'] for x in tv_items]}")
            for it in tv_items:
                try:
                    scored = eval_tvsum_keyframe_video(
                        it["video_path"],
                        it["scores"],
                        video=f"TVSum:{it['video_id']}",
                    )
                    results["keyframe"].append(scored)
                    print(f"[Keyframe] {it['video_id']} F1={scored['f1']:.3f}")
                except Exception as e:
                    print(f"[Keyframe][WARN] {it['video_id']}: {e}")
        else:
            for row in rows:
                vid = row.get("video_id") or "video"
                ref_path = _resolve(row.get("reference_keyframes"), bench_root)
                pred_path = _resolve(row.get("pred_keyframes"), bench_root)
                if not ref_path or not ref_path.exists() or not pred_path or not pred_path.exists():
                    continue
                ref = load_ref_if_exists(ref_path) or {}
                pred = load_ref_if_exists(pred_path) or {}
                must = ref.get("must_keep") if isinstance(ref, dict) else []
                before = pred.get("before") if isinstance(pred, dict) else []
                after = pred.get("after") if isinstance(pred, dict) else pred.get("ids", [])
                results["keyframe"].append(
                    eval_keyframe_filter(list(before or []), list(after or []), list(must or []), video=vid)
                )
        results["stage_status"]["keyframe"] = "done" if results["keyframe"] else "pending"
        results["keyframe_conclusion"] = conclude("F1", [r.get("f1", float("nan")) for r in results["keyframe"]])

    # --- OCR / Caption / Summary / Ablation on TED lecture videos ---
    pending_needed = {"ocr", "caption", "summary", "ablation"} & stages
    if args.auto_datasets and pending_needed:
        ted_pending = pick_ted_lecture_videos(limit=2)
        print(f"[Pending] TED videos: {[p.name for p in ted_pending]}")
        for vp in ted_pending:
            try:
                pending = eval_ted_pending_stages(
                    vp,
                    max_ocr_frames=8,
                    max_caption_frames=4,
                    run_ocr=not args.skip_ocr,
                )
            except Exception as e:
                print(f"[Pending][WARN] {vp.name}: {e}")
                continue
            if "ocr" in stages:
                results["ocr"].extend(pending.get("ocr") or [])
            if "caption" in stages:
                results["caption"].extend(pending.get("caption") or [])
            if "summary" in stages:
                results["summary"].extend(pending.get("summary") or [])
            if "ablation" in stages:
                results["ablation"].extend(pending.get("ablation") or [])
        if "ocr" in stages:
            results["stage_status"]["ocr"] = "done" if results["ocr"] else "pending"
            results["ocr_conclusion"] = conclude(
                "CER",
                [r.get("cer", float("nan")) for r in results["ocr"]],
                higher_better=False,
                good=0.35,
            ) + " OCR vs lời nói TED-LIUM cùng timestamp (weak GT, không phải GT chữ slide)."
        if "caption" in stages:
            agg = aggregate_caption_scores(results["caption"])
            results["stage_status"]["caption"] = "done" if results["caption"] else "pending"
            acc = agg.get("accuracy")
            hall = agg.get("hallucination_rate")
            results["caption_conclusion"] = (
                f"_Kết luận: accuracy={acc:.3f} hallucination_rate={hall:.3f} "
                "(heuristic grounded trên OCR; Florence-2 tắt để tránh tranh GPU)._"
                if results["caption"] and acc == acc and hall == hall
                else "_Kết luận: TBD._"
            )
        if "summary" in stages:
            results["stage_status"]["summary"] = "done" if results["summary"] else "pending"
            results["summary_conclusion"] = conclude(
                "ROUGE-L",
                [r.get("rouge_l", float("nan")) for r in results["summary"]],
                good=0.25,
            ) + " Reference = câu đầu mỗi cửa sổ 40s từ transcript TED-LIUM; hyp = extractive TF-IDF."
        if "ablation" in stages:
            results["stage_status"]["ablation"] = "done" if results["ablation"] else "pending"
            results["ablation_conclusion"] = conclude(
                "Summary ROUGE-L",
                [r.get("summary_score", float("nan")) for r in results["ablation"]],
                good=0.25,
            ) + " So sánh Audio / Visual / Audio+Visual trên cùng TED talk."

    # --- OCR ---
    if "ocr" in stages and not args.auto_datasets:
        for row in rows:
            ref_path = _resolve(row.get("reference_ocr"), bench_root)
            if not ref_path or not ref_path.exists():
                continue
            payload = load_ref_if_exists(ref_path)
            items = payload if isinstance(payload, list) else payload.get("items", [])
            # only score items that already include hypothesis; else leave TBD
            scored_items = [it for it in items if it.get("hypothesis") or it.get("ocr")]
            if scored_items:
                results["ocr"].extend(eval_ocr_items(scored_items))
        results["stage_status"]["ocr"] = "done" if results["ocr"] else "pending"
        results["ocr_conclusion"] = conclude(
            "CER",
            [r.get("cer", float("nan")) for r in results["ocr"]],
            higher_better=False,
            good=0.12,
        )

    # --- Caption ---
    if "caption" in stages and not args.auto_datasets:
        for row in rows:
            ref_path = _resolve(row.get("reference_captions"), bench_root)
            if not ref_path or not ref_path.exists():
                continue
            payload = load_ref_if_exists(ref_path)
            items = payload if isinstance(payload, list) else payload.get("items", [])
            results["caption"].extend(eval_captions(items))
        agg = aggregate_caption_scores(results["caption"])
        results["stage_status"]["caption"] = "done" if results["caption"] else "pending"
        results["caption_conclusion"] = (
            f"_Kết luận: accuracy={agg.get('accuracy')} hallucination_rate={agg.get('hallucination_rate')}_"
            if results["caption"]
            else "_Kết luận: TBD._"
        )

    # --- Timeline + Chapter on TED lecture videos ---
    if args.auto_datasets and ("timeline" in stages or "chapter" in stages):
        ted_videos = pick_ted_lecture_videos(limit=2)
        ted_rows = load_tedlium_rows()
        for vp in ted_videos:
            speaker = vp.stem
            utts = [
                {"start": r["start"], "end": r["end"], "text": r["text"]}
                for r in ted_rows
                if r["speaker_id"] == speaker and r["start"] is not None and r["end"] is not None
            ]
            if len(utts) < 3:
                # fallback: any speaker whose name appears in filename
                utts = [
                    {"start": r["start"], "end": r["end"], "text": r["text"]}
                    for r in ted_rows
                    if r["speaker_id"].lower() in speaker.lower()
                    and r["start"] is not None
                ]
            if len(utts) < 3:
                print(f"[TED] skip timeline/chapter {vp.name}: not enough TED-LIUM utterances")
                continue
            print(f"[TED] timeline/chapter on {vp.name} utt={len(utts)}")
            try:
                tl, ch = eval_ted_timeline_chapter(
                    vp, utts, video=f"TED:{speaker}", chapter_tol=args.chapter_tol
                )
                if "timeline" in stages:
                    results["timeline"].append(tl)
                if "chapter" in stages:
                    results["chapter"].append(ch)
            except Exception as e:
                print(f"[TED][WARN] {vp.name}: {e}")
        if "timeline" in stages:
            results["stage_status"]["timeline"] = "done" if results["timeline"] else "pending"
            results["timeline_conclusion"] = conclude(
                "Accuracy", [r.get("accuracy", float("nan")) for r in results["timeline"]]
            )
        if "chapter" in stages:
            results["stage_status"]["chapter"] = "done" if results["chapter"] else "pending"
            results["chapter_conclusion"] = conclude(
                "Boundary F1", [r.get("f1", float("nan")) for r in results["chapter"]]
            )
    else:
        if "timeline" in stages:
            for row in rows:
                vid = row.get("video_id") or "video"
                ref_path = _resolve(row.get("reference_alignments"), bench_root)
                pred_path = _resolve(row.get("pred_alignments"), bench_root)
                if not ref_path or not ref_path.exists() or not pred_path or not pred_path.exists():
                    continue
                ref = load_ref_if_exists(ref_path)
                pred = load_ref_if_exists(pred_path)
                ref_list = ref if isinstance(ref, list) else ref.get("items", [])
                pred_list = pred if isinstance(pred, list) else pred.get("items", [])
                results["timeline"].append(eval_timeline_alignment(pred_list, ref_list, video=vid))
            results["stage_status"]["timeline"] = "done" if results["timeline"] else "pending"
            results["timeline_conclusion"] = conclude(
                "Accuracy", [r.get("accuracy", float("nan")) for r in results["timeline"]]
            )

        if "chapter" in stages:
            for row in rows:
                vid = row.get("video_id") or "video"
                ref_path = _resolve(row.get("reference_chapters"), bench_root)
                pred_path = _resolve(row.get("pred_chapters"), bench_root)
                if not ref_path or not ref_path.exists():
                    continue
                ref = load_ref_if_exists(ref_path)
                pred = load_ref_if_exists(pred_path) if pred_path and pred_path.exists() else None
                if pred is None:
                    continue
                results["chapter"].append(
                    eval_chapters(pred, ref, tolerance_sec=args.chapter_tol, video=vid)
                )
            results["stage_status"]["chapter"] = "done" if results["chapter"] else "pending"
            results["chapter_conclusion"] = conclude("Boundary F1", [r.get("f1", float("nan")) for r in results["chapter"]])

    # --- Summary ---
    if "summary" in stages and not args.auto_datasets:
        for row in rows:
            vid = row.get("video_id") or "video"
            ref_path = _resolve(row.get("reference_summary"), bench_root)
            pred_path = _resolve(row.get("pred_summary"), bench_root)
            if not ref_path or not ref_path.exists() or not pred_path or not pred_path.exists():
                continue
            ref = ref_path.read_text(encoding="utf-8")
            hyp = pred_path.read_text(encoding="utf-8")
            results["summary"].append(
                eval_summary_pair(
                    ref,
                    hyp,
                    video=vid,
                    input_used=row.get("summary_input") or "Transcript + OCR + caption",
                    compute_bertscore=args.bertscore,
                )
            )
        results["stage_status"]["summary"] = "done" if results["summary"] else "pending"
        results["summary_conclusion"] = conclude(
            "ROUGE-L", [r.get("rouge_l", float("nan")) for r in results["summary"]], good=0.25
        )

    # --- Ablation (expects pred summaries under modality folders or JSON) ---
    if "ablation" in stages and not args.auto_datasets:
        abl_path = bench_root / "references" / "ablation_results.json"
        if abl_path.exists():
            payload = json.loads(abl_path.read_text(encoding="utf-8"))
            results["ablation"] = payload if isinstance(payload, list) else payload.get("rows", [])
            results["stage_status"]["ablation"] = "done"
            results["ablation_conclusion"] = (
                "_Kết luận: so sánh Audio / Visual / Audio+Visual từ ablation_results.json._"
            )
        else:
            results["ablation"] = [
                {"config": "Audio only", "input": "Transcript", "summary_score": None, "hit_at_3": None, "hit_at_5": None},
                {"config": "Visual only", "input": "OCR + caption", "summary_score": None, "hit_at_3": None, "hit_at_5": None},
                {"config": "Audio + Visual", "input": "Transcript + OCR + caption", "summary_score": None, "hit_at_3": None, "hit_at_5": None},
            ]
            results["stage_status"]["ablation"] = "pending"
            results["ablation_conclusion"] = (
                "_Kết luận: TBD — tạo benchmarks/references/ablation_results.json sau khi chạy 3 cấu hình._"
            )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill thesis evaluation tables Bang 1–12")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "benchmarks" / "manifest_eval.csv"))
    parser.add_argument("--benchmarks-root", type=str, default=str(ROOT / "benchmarks"))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--stages", type=str, default="all", help="Comma list or 'all'")
    parser.add_argument("--asr-models", type=str, default="base.en,small.en")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--dry-report", action="store_true", help="Write empty TBD tables only")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scene-tol", type=float, default=2.0)
    parser.add_argument("--chapter-tol", type=float, default=15.0)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument(
        "--auto-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TED-LIUM/TVSum instead of sample.mp4 (default: on)",
    )
    parser.add_argument("--asr-limit", type=int, default=12, help="TED-LIUM ASR clips")
    parser.add_argument("--tvsum-limit", type=int, default=4, help="TVSum videos for scene/keyframe")
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip PaddleOCR download/init (still run caption/summary/RAG/ablation)",
    )
    args = parser.parse_args()

    if args.asr_only:
        args.stages = "asr"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / f"eval_tables_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_report:
        results: dict[str, Any] = {}
    else:
        results = run_eval(args)

    report_path = write_report(results, out_dir / "EVAL_TABLES.md")
    (out_dir / "EVAL_TABLES.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"[OK] Wrote {report_path}")
    print(f"[OK] Wrote {out_dir / 'EVAL_TABLES.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

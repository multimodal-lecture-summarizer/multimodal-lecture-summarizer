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

from experiments.evaluation.aggregate import build_asr_model_comparison, build_dataset_aggregates, group_mean
from experiments.evaluation.datasets import (
    TED_DATASET,
    load_tedlium_rows,
    pick_ted_unified_talks,
    pick_ted_vad_items,
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


def _production_asr_name(model_size: str) -> str:
    return f"faster-whisper-{model_size}"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    bench_root = Path(args.benchmarks_root).resolve()
    prod_asr = _production_asr_name(args.production_asr.strip())
    results: dict[str, Any] = {
        "production_model": prod_asr,
        "dataset_primary": TED_DATASET,
        "asr": [],
        "vad": [],
        "scene": [],
        "keyframe": [],
        "ocr": [],
        "caption": [],
        "timeline": [],
        "chapter": [],
        "summary": [],
        "ablation": [],
        "model_comparison": {"asr": [], "keyframe": []},
        "stage_status": {},
    }
    stages = {s.strip().lower() for s in args.stages.split(",") if s.strip()}
    if "all" in stages:
        stages = {
            "asr",
            "scene",
            "keyframe",
            "ocr",
            "caption",
            "timeline",
            "chapter",
            "summary",
        }
        if args.include_vad:
            stages.add("vad")
        if args.include_ablation:
            stages.add("ablation")

    rows = []
    if args.manifest and not args.auto_datasets:
        rows = read_manifest(Path(args.manifest))
        if args.limit and args.limit > 0:
            rows = rows[: args.limit]

    ted_talks: list[dict[str, Any]] = []
    if args.auto_datasets:
        clips_per = max(1, args.asr_limit // max(1, args.ted_limit))
        ted_talks = pick_ted_unified_talks(limit=args.ted_limit, asr_clips_per_talk=clips_per)
        print(f"[TED] Unified talks ({TED_DATASET}): {[t['talk_id'] for t in ted_talks]}")

    # --- ASR (TED, multi-model vs production) ---
    if "asr" in stages:
        asr_models = [m.strip() for m in args.asr_models.split(",") if m.strip()]
        if args.model_compare:
            for m in [x.strip() for x in args.compare_asr_sizes.split(",") if x.strip()]:
                if m not in asr_models:
                    asr_models.append(m)
        if args.auto_datasets:
            for talk in ted_talks:
                for clip in talk.get("asr_clips") or []:
                    for model in asr_models:
                        try:
                            scored = eval_asr_file(
                                clip["wav_path"],
                                clip["text"],
                                model_size=model,
                                language="en",
                            )
                            scored["dataset"] = TED_DATASET
                            scored["talk_id"] = talk["talk_id"]
                            scored["clip_id"] = clip.get("id") or clip["wav_path"].stem
                            scored["model"] = f"faster-whisper-{model}"
                            results["asr"].append(scored)
                            print(
                                f"[ASR] {talk['talk_id']} {model} WER={scored['wer']:.3f} RTF={scored['rtf']:.3f}"
                            )
                        except Exception as e:
                            print(f"[ASR][WARN] {talk['talk_id']} {model}: {e}")
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
        ) + f" Dataset **{TED_DATASET}**; production `{prod_asr}`."
        if len(asr_models) > 1:
            results["model_comparison"]["asr"] = build_asr_model_comparison(
                results["asr"], production_model=prod_asr
            )
            by_model = group_mean(results["asr"], "model", "wer")
            parts = [f"{m}: WER={v:.3f}" for m, v in sorted(by_model.items())]
            results["asr_conclusion"] += " " + "; ".join(parts) + "."

    # --- VAD (optional) ---
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
                    scored["dataset"] = "TVSum"
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
                    scored["dataset"] = "TVSum"
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

    # --- OCR / Caption / Summary on unified TED talks ---
    pending_needed = {"ocr", "caption", "summary", "ablation"} & stages
    if args.auto_datasets and pending_needed and ted_talks:
        for talk in ted_talks:
            vp = talk["video_path"]
            print(f"[TED] OCR/Caption/Summary on {vp.name} (dataset={TED_DATASET})")
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
            ) + f" Dataset **{TED_DATASET}**; weak GT = transcript cùng timestamp."
        if "caption" in stages:
            agg = aggregate_caption_scores(results["caption"])
            results["stage_status"]["caption"] = "done" if results["caption"] else "pending"
            acc = agg.get("accuracy")
            hall = agg.get("hallucination_rate")
            hs = mean_ignore_nan(r.get("human_score") for r in results["caption"])
            results["caption_conclusion"] = (
                f"_Kết luận: human_score≈{hs:.1f}/5, accuracy={acc:.3f}, hallucination={hall:.3f} "
                f"(dataset **{TED_DATASET}**; human score = proxy heuristic)._"
                if results["caption"] and acc == acc and hall == hall
                else "_Kết luận: TBD._"
            )
        if "summary" in stages:
            results["stage_status"]["summary"] = "done" if results["summary"] else "pending"
            results["summary_conclusion"] = (
                conclude("ROUGE-L", [r.get("rouge_l", float("nan")) for r in results["summary"]], good=0.25)
                + f" Dataset **{TED_DATASET}**; "
                + f"factuality≈{mean_ignore_nan(r.get('factuality') for r in results['summary']):.3f}, "
                + f"coverage≈{mean_ignore_nan(r.get('coverage') for r in results['summary']):.3f}."
            )
        if "ablation" in stages:
            results["stage_status"]["ablation"] = "done" if results["ablation"] else "pending"
            results["ablation_conclusion"] = (
                conclude("Summary ROUGE-L", [r.get("summary_score", float("nan")) for r in results["ablation"]], good=0.25)
                + f" Dataset **{TED_DATASET}** (Audio / Visual / Audio+Visual)."
            )

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

    # --- Timeline + Chapter on unified TED talks ---
    if args.auto_datasets and ("timeline" in stages or "chapter" in stages) and ted_talks:
        for talk in ted_talks:
            vp = talk["video_path"]
            utts = talk.get("utterances") or []
            if len(utts) < 3:
                print(f"[TED] skip timeline/chapter {vp.name}: not enough utterances")
                continue
            print(f"[TED] timeline/chapter on {vp.name} utt={len(utts)} dataset={TED_DATASET}")
            try:
                tl, ch = eval_ted_timeline_chapter(
                    vp,
                    utts,
                    video=talk["talk_id"],
                    chapter_tol=args.chapter_tol,
                )
                if "timeline" in stages:
                    results["timeline"].append(tl)
                if "chapter" in stages:
                    results["chapter"].append(ch)
            except Exception as e:
                print(f"[TED][WARN] {vp.name}: {e}")
        if "timeline" in stages:
            results["stage_status"]["timeline"] = "done" if results["timeline"] else "pending"
            results["timeline_conclusion"] = (
                conclude("MAE (s)", [r.get("mae_sec", float("nan")) for r in results["timeline"]], higher_better=False, good=2.0)
                + f" Dataset **{TED_DATASET}**."
            )
        if "chapter" in stages:
            results["stage_status"]["chapter"] = "done" if results["chapter"] else "pending"
            results["chapter_conclusion"] = (
                conclude("Boundary F1", [r.get("f1", float("nan")) for r in results["chapter"]])
                + f" Dataset **{TED_DATASET}**."
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

    if args.model_compare and args.auto_datasets:
        _run_model_comparisons(args, results, ted_talks)

    return results


def _run_model_comparisons(args: argparse.Namespace, results: dict[str, Any], ted_talks: list[dict[str, Any]]) -> None:
    from experiments.evaluation.model_compare import (
        build_justification_summary,
        compare_caption_models,
        compare_keyframe_strategies,
        compare_ocr_engines,
        compare_scene_thresholds,
    )

    mc = results.setdefault("model_comparison", {"asr": results.get("model_comparison", {}).get("asr", [])})

    tv_items = pick_tvsum_videos(limit=max(1, min(2, args.tvsum_limit or 1)))
    if tv_items:
        it = tv_items[0]
        try:
            import cv2

            cap = cv2.VideoCapture(str(it["video_path"]))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
            cap.release()
            thresholds = tuple(float(x.strip()) for x in args.scene_thresholds.split(",") if x.strip())
            print(f"[Compare] Scene thresholds {thresholds} on {it['video_id']}")
            mc["scene_threshold"] = compare_scene_thresholds(
                it["video_path"],
                it["scores"],
                fps=fps,
                thresholds=thresholds,
                tolerance_sec=args.scene_tol,
            )
            print(f"[Compare] Keyframe strategies on {it['video_id']}")
            mc["keyframe"] = compare_keyframe_strategies(
                it["video_path"],
                it["scores"],
                video_label=f"TVSum:{it['video_id']}",
            )
        except Exception as e:
            print(f"[Compare][WARN] scene/keyframe: {e}")

    if ted_talks:
        vp = ted_talks[0]["video_path"]
        try:
            print(f"[Compare] Caption models on {vp.name}")
            mc["caption"] = compare_caption_models(vp, max_frames=min(3, args.caption_compare_frames))
        except Exception as e:
            print(f"[Compare][WARN] caption: {e}")

        if not args.skip_ocr and ted_talks[0].get("utterances"):
            try:
                from ai_workers.modules.visual_v2.scene_detector import SceneDetector

                det = SceneDetector({"threshold": 27.0})
                scenes = det.detect_scenes(str(vp))
                work = vp.parent / "_ocr_cmp" / vp.stem
                det.extract_keyframes(str(vp), scenes, str(work), strategy="middle")
                keyed = [s for s in scenes if s.get("keyframe_path")]
                if keyed:
                    sc = keyed[len(keyed) // 2]
                    path = sc.get("keyframe_path") or ""
                    mid = 0.5 * (float(sc.get("start_seconds", 0)) + float(sc.get("end_seconds", 0)))
                    ref_utts = [
                        u["text"]
                        for u in ted_talks[0]["utterances"]
                        if float(u["start"]) <= mid <= float(u["end"]) + 2.0
                    ]
                    ref = " ".join(ref_utts) or " "
                    print(f"[Compare] OCR engines on {Path(path).name}")
                    mc["ocr"] = compare_ocr_engines(path, ref)
            except Exception as e:
                print(f"[Compare][WARN] ocr: {e}")

    results["model_justification"] = build_justification_summary(mc)
    print(f"[Compare] {results['model_justification'][:160]}...")


def finalize_results(results: dict[str, Any]) -> dict[str, Any]:
    """Attach dataset-level aggregates (one row per TED / TVSum)."""
    results["aggregated"] = build_dataset_aggregates(results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill thesis evaluation tables Bang 1–12")
    parser.add_argument("--manifest", type=str, default=str(ROOT / "benchmarks" / "manifest_eval.csv"))
    parser.add_argument("--benchmarks-root", type=str, default=str(ROOT / "benchmarks"))
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--stages", type=str, default="all", help="Comma list or 'all'")
    parser.add_argument("--asr-models", type=str, default="tiny.en,base.en,small.en")
    parser.add_argument("--production-asr", type=str, default="base.en", help="Production Faster-Whisper size")
    parser.add_argument("--ted-limit", type=int, default=2, help="Number of TED talks (unified dataset)")
    parser.add_argument("--include-vad", action="store_true", help="Include optional VAD table")
    parser.add_argument("--include-ablation", action="store_true", help="Include optional ablation table")
    parser.add_argument(
        "--model-compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare candidate models vs production (scene threshold, CLIP, caption, OCR, extra ASR)",
    )
    parser.add_argument(
        "--scene-thresholds",
        type=str,
        default="20,27,35",
        help="PySceneDetect thresholds to compare (production=27)",
    )
    parser.add_argument("--caption-compare-frames", type=int, default=2, help="Keyframes per TED talk for caption bakeoff")
    parser.add_argument(
        "--compare-asr-sizes",
        type=str,
        default="medium.en",
        help="Extra ASR sizes when --model-compare (appended if missing)",
    )
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
        results = finalize_results(run_eval(args))

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

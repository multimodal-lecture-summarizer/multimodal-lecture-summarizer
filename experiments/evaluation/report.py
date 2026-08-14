"""Markdown report generator for thesis pipeline evaluation (5 priority stages)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.evaluation.aggregate import build_dataset_aggregates
from experiments.evaluation.metrics import mean_ignore_nan


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "TBD"
    if isinstance(v, float):
        if v != v:
            return "N/A"
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v: Any, digits: int = 2) -> str:
    if v is None:
        return "TBD"
    if isinstance(v, float) and v != v:
        return "N/A"
    try:
        fv = float(v)
        return f"{fv * 100:.{digits}f}" if abs(fv) <= 1.5 else f"{fv:.{digits}f}"
    except Exception:
        return str(v)


def _delta(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "N/A"
    fv = float(v)
    sign = "+" if fv > 0 else ""
    return f"{sign}{fv:.3f}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |")
    return "\n".join(lines)


def _mean_rows(rows: list[dict[str, Any]], key: str) -> float:
    return mean_ignore_nan(r.get(key) for r in rows)


def render_eval_report(results: dict[str, Any], *, title: str | None = None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = title or "Bảng đánh giá pipeline AI (không gồm RAG/Q&A)"
    prod = results.get("production_model", "faster-whisper-base.en")
    parts: list[str] = [
        f"# {title}",
        "",
        f"*Generated: {now}*",
        "",
        "> Phạm vi: 5 stage ưu tiên — ASR, Scene/Keyframe, OCR/Caption, Timeline/Chapter, Summary.",
        "> Dataset **TED** = một dòng gộp toàn bộ talk (clip ASR + video + timeline trên cùng corpus).",
        f"> Mô hình production (ASR): `{prod}`.",
        "",
    ]

    agg = results.get("aggregated") or build_dataset_aggregates(results)

    # --- Bảng 1: ASR ---
    parts.append("## Bảng 1. ASR — WER, CER, RTF (dataset TED)")
    parts.append("")
    asr_agg = agg.get("asr") or []
    parts.append(
        _md_table(
            ["Dataset", "Model", "Clips", "WER (%)", "CER (%)", "RTF"],
            [
                [
                    r.get("dataset", "TED"),
                    r.get("model", "TBD"),
                    _fmt(r.get("n_clips"), 0),
                    _pct(r.get("wer"), 2),
                    _pct(r.get("cer"), 2),
                    _fmt(r.get("rtf")),
                ]
                for r in (asr_agg or [{"dataset": "TED", "model": prod}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("asr_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    cmp_asr = results.get("model_comparison", {}).get("asr", [])
    if cmp_asr:
        parts.append("### So sánh ASR: candidate vs production")
        parts.append("")
        parts.append(
            _md_table(
                ["Candidate model", "Metric", "Production", "Candidate", "Δ", "Better?"],
                [
                    [
                        r.get("model", ""),
                        r.get("metric", "").upper(),
                        _pct(r.get("production"), 2) if r.get("metric") in {"wer", "cer"} else _fmt(r.get("production")),
                        _pct(r.get("candidate"), 2) if r.get("metric") in {"wer", "cer"} else _fmt(r.get("candidate")),
                        _delta(r.get("delta")),
                        "Yes" if r.get("better_than_production") else "No",
                    ]
                    for r in cmp_asr
                ],
            )
        )
        parts.append("")

    # --- Bảng 2: Scene ---
    parts.append("## Bảng 2. Scene Detection — Precision / Recall / F1")
    parts.append("")
    scene_agg = agg.get("scene") or []
    parts.append(
        _md_table(
            ["Dataset", "Videos", "Scene GT", "Scene pred", "P", "R", "F1"],
            [
                [
                    r.get("dataset", "TVSum"),
                    _fmt(r.get("n_videos"), 0),
                    _fmt(r.get("n_ref"), 0),
                    _fmt(r.get("n_pred"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                ]
                for r in (scene_agg or [{"dataset": "TVSum"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("scene_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 3: Keyframe ---
    parts.append("## Bảng 3. Keyframe & CLIP Filtering — P / R / F1")
    parts.append("")
    kf_agg = agg.get("keyframe") or []
    parts.append(
        _md_table(
            ["Dataset", "Videos", "Trước lọc", "Sau lọc", "P", "R", "F1", "Nén"],
            [
                [
                    r.get("dataset", "TVSum"),
                    _fmt(r.get("n_videos"), 0),
                    _fmt(r.get("n_before"), 0),
                    _fmt(r.get("n_after"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                    _fmt(r.get("compression_ratio")),
                ]
                for r in (kf_agg or [{"dataset": "TVSum"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("keyframe_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    kf_cmp = results.get("model_comparison", {}).get("keyframe", [])
    if kf_cmp:
        parts.append("### So sánh chiến lược keyframe vs production (CLIP agglomerative)")
        parts.append("")
        parts.append(
            _md_table(
                ["Strategy", "P", "R", "F1", "Δ F1 vs prod"],
                [
                    [
                        r.get("strategy", ""),
                        _fmt(r.get("precision")),
                        _fmt(r.get("recall")),
                        _fmt(r.get("f1")),
                        _delta(r.get("delta_f1")),
                    ]
                    for r in kf_cmp
                ],
            )
        )
        parts.append("")

    # --- Bảng 4: OCR ---
    parts.append("## Bảng 4. OCR — CER, Word Accuracy (dataset TED)")
    parts.append("")
    ocr_agg = agg.get("ocr") or []
    parts.append(
        _md_table(
            ["Dataset", "Keyframes", "CER (%)", "Word Acc"],
            [
                [
                    r.get("dataset", "TED"),
                    _fmt(r.get("n_keyframes"), 0),
                    _pct(r.get("cer"), 2),
                    _fmt(r.get("word_accuracy")),
                ]
                for r in (ocr_agg or [{"dataset": "TED"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("ocr_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 5: Caption ---
    parts.append("## Bảng 5. Caption — Human score (proxy), Hallucination (dataset TED)")
    parts.append("")
    cap_agg = agg.get("caption") or []
    parts.append(
        _md_table(
            ["Dataset", "Keyframes", "Human score (1–5)", "Content OK", "Hallucination rate"],
            [
                [
                    r.get("dataset", "TED"),
                    _fmt(r.get("n_keyframes"), 0),
                    _fmt(r.get("human_score"), 1),
                    _fmt(r.get("content_ok_rate")),
                    _fmt(r.get("hallucination_rate")),
                ]
                for r in (cap_agg or [{"dataset": "TED"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("caption_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 6: Timeline ---
    parts.append("## Bảng 6. Timeline Alignment — Accuracy, MAE (dataset TED)")
    parts.append("")
    tl_agg = agg.get("timeline") or []
    parts.append(
        _md_table(
            ["Dataset", "Talks", "Segments", "Accuracy", "MAE (s)"],
            [
                [
                    r.get("dataset", "TED"),
                    _fmt(r.get("n_talks"), 0),
                    _fmt(r.get("n_segments"), 0),
                    _fmt(r.get("accuracy")),
                    _fmt(r.get("mae_sec")),
                ]
                for r in (tl_agg or [{"dataset": "TED"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("timeline_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 7: Chapter ---
    parts.append("## Bảng 7. Chapter Segmentation — Boundary P/R/F1, MAE (dataset TED)")
    parts.append("")
    ch_agg = agg.get("chapter") or []
    parts.append(
        _md_table(
            ["Dataset", "Talks", "Ch GT", "Ch pred", "Boundary P", "Boundary R", "Boundary F1", "MAE (s)"],
            [
                [
                    r.get("dataset", "TED"),
                    _fmt(r.get("n_talks"), 0),
                    _fmt(r.get("n_ref"), 0),
                    _fmt(r.get("n_pred"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                    _fmt(r.get("mae")),
                ]
                for r in (ch_agg or [{"dataset": "TED"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("chapter_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 8: Summary ---
    parts.append("## Bảng 8. Summary — ROUGE-L, BERTScore, Factuality, Coverage (dataset TED)")
    parts.append("")
    sum_agg = agg.get("summary") or []
    parts.append(
        _md_table(
            ["Dataset", "Talks", "Input", "ROUGE-L", "BERTScore", "Factuality", "Coverage"],
            [
                [
                    r.get("dataset", "TED"),
                    _fmt(r.get("n_talks"), 0),
                    r.get("input", "Transcript + OCR + caption"),
                    _fmt(r.get("rouge_l")),
                    _fmt(r.get("bertscore_f1")),
                    _fmt(r.get("factuality")),
                    _fmt(r.get("coverage")),
                ]
                for r in (sum_agg or [{"dataset": "TED"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("summary_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # --- Bảng 9: Tổng hợp ---
    parts.append("## Bảng 9. Tổng hợp metric theo stage (dataset TED + TVSum scene/keyframe)")
    parts.append("")
    stage_rows = results.get("stages") or _default_stage_summary(results)
    parts.append(
        _md_table(
            ["Stage", "Production model", "Dataset", "Metrics chính", "Giá trị TB", "Status"],
            [
                [
                    r.get("stage", "TBD"),
                    r.get("model", "TBD"),
                    r.get("dataset", "TBD"),
                    r.get("metrics", "TBD"),
                    r.get("value", "TBD"),
                    r.get("status", "pending"),
                ]
                for r in stage_rows
            ],
        )
    )
    parts.append("")
    parts.append("### Thứ tự ưu tiên")
    parts.append("")
    parts.append("1. ASR (WER, CER, RTF)")
    parts.append("2. Scene/keyframe (P/R/F1)")
    parts.append("3. OCR/Caption (CER, human score, hallucination)")
    parts.append("4. Timeline/chapter (MAE, boundary F1)")
    parts.append("5. Summary (factuality, coverage, ROUGE-L/BERTScore)")
    parts.append("")

    # --- Bảng 10: Model justification ---
    mc = results.get("model_comparison") or {}
    parts.append("## Bảng 10. Chứng minh lựa chọn mô hình production vs candidate")
    parts.append("")
    parts.append(results.get("model_justification", "_Chạy với `--model-compare` để sinh bảng này._"))
    parts.append("")

    scene_cmp = mc.get("scene_threshold") or []
    if scene_cmp:
        parts.append("### Scene — PySceneDetect threshold")
        parts.append("")
        parts.append(
            _md_table(
                ["Model", "Cuts", "P", "R", "F1", "Production?"],
                [
                    [
                        r.get("model", ""),
                        _fmt(r.get("n_pred"), 0),
                        _fmt(r.get("precision")),
                        _fmt(r.get("recall")),
                        _fmt(r.get("f1")),
                        "Yes" if r.get("is_production") else "No",
                    ]
                    for r in scene_cmp
                ],
            )
        )
        parts.append("")

    kf_cmp = mc.get("keyframe") or []
    if kf_cmp:
        parts.append("### Keyframe — keep-all vs temporal vs CLIP (production)")
        parts.append("")
        parts.append(
            _md_table(
                ["Strategy", "P", "R", "F1", "Δ F1", "Production?"],
                [
                    [
                        r.get("strategy", r.get("model", "")),
                        _fmt(r.get("precision")),
                        _fmt(r.get("recall")),
                        _fmt(r.get("f1")),
                        _delta(r.get("delta_f1_vs_prod", r.get("delta_f1"))),
                        "Yes" if r.get("is_production") else "No",
                    ]
                    for r in kf_cmp
                ],
            )
        )
        parts.append("")

    cap_cmp = mc.get("caption") or []
    if cap_cmp:
        parts.append("### Caption — placeholder vs OCR-grounded vs Florence-2")
        parts.append("")
        parts.append(
            _md_table(
                ["Model", "Content OK", "Hallucination", "Human score", "Production?"],
                [
                    [
                        r.get("model", ""),
                        _fmt(r.get("content_ok_rate")),
                        _fmt(r.get("hallucination_rate")),
                        _fmt(r.get("human_score"), 1),
                        "Yes" if r.get("is_production") else "No",
                    ]
                    for r in cap_cmp
                    if not r.get("error")
                ],
            )
        )
        parts.append("")

    ocr_cmp = mc.get("ocr") or []
    if ocr_cmp:
        parts.append("### OCR — PaddleOCR vs EasyOCR vs Tesseract")
        parts.append("")
        parts.append(
            _md_table(
                ["Engine", "CER (%)", "Word Acc", "Time (s)", "Production?"],
                [
                    [
                        r.get("engine", ""),
                        _pct(r.get("cer"), 2),
                        _fmt(r.get("word_accuracy")),
                        _fmt(r.get("wall_sec")),
                        "Yes" if r.get("is_production") else "No",
                    ]
                    for r in ocr_cmp
                    if not r.get("error")
                ],
            )
        )
        parts.append("")

    return "\n".join(parts)


def _default_stage_summary(results: dict[str, Any]) -> list[dict[str, Any]]:
    status = results.get("stage_status", {})
    prod = results.get("production_model", "faster-whisper-base.en")
    agg = results.get("aggregated") or build_dataset_aggregates(results)
    asr_ted = next((r for r in agg.get("asr") or [] if r.get("model") == prod), {})
    tl_ted = (agg.get("timeline") or [{}])[0] if agg.get("timeline") else {}
    ch_ted = (agg.get("chapter") or [{}])[0] if agg.get("chapter") else {}
    scene_tv = (agg.get("scene") or [{}])[0] if agg.get("scene") else {}
    kf_tv = (agg.get("keyframe") or [{}])[0] if agg.get("keyframe") else {}
    ocr_ted = (agg.get("ocr") or [{}])[0] if agg.get("ocr") else {}
    cap_ted = (agg.get("caption") or [{}])[0] if agg.get("caption") else {}
    sum_ted = (agg.get("summary") or [{}])[0] if agg.get("summary") else {}
    return [
        {
            "stage": "ASR",
            "model": prod,
            "dataset": "TED",
            "metrics": "WER, CER, RTF",
            "value": f"WER={_pct(asr_ted.get('wer'), 1)}" if asr_ted else "TBD",
            "status": status.get("asr", "pending"),
        },
        {
            "stage": "Scene",
            "model": "PySceneDetect",
            "dataset": "TVSum",
            "metrics": "P, R, F1",
            "value": f"F1={_fmt(scene_tv.get('f1'))}",
            "status": status.get("scene", "pending"),
        },
        {
            "stage": "Keyframe",
            "model": "CLIP ViT-B/32 agglomerative",
            "dataset": "TVSum",
            "metrics": "P, R, F1",
            "value": f"F1={_fmt(kf_tv.get('f1'))}",
            "status": status.get("keyframe", "pending"),
        },
        {
            "stage": "OCR",
            "model": "PaddleOCR",
            "dataset": "TED",
            "metrics": "CER, Word Acc",
            "value": f"CER={_pct(ocr_ted.get('cer'), 1)}",
            "status": status.get("ocr", "pending"),
        },
        {
            "stage": "Caption",
            "model": "Florence-2-base / OCR-grounded",
            "dataset": "TED",
            "metrics": "Human score, Hallucination",
            "value": f"score={_fmt(cap_ted.get('human_score'), 1)}",
            "status": status.get("caption", "pending"),
        },
        {
            "stage": "Timeline",
            "model": "Temporal proximity + scene",
            "dataset": "TED",
            "metrics": "Accuracy, MAE",
            "value": f"Acc={_fmt(tl_ted.get('accuracy'))}, MAE={_fmt(tl_ted.get('mae_sec'))}s",
            "status": status.get("timeline", "pending"),
        },
        {
            "stage": "Chapter",
            "model": "TF-IDF semantic shift",
            "dataset": "TED",
            "metrics": "Boundary F1, MAE",
            "value": f"F1={_fmt(ch_ted.get('f1'))}",
            "status": status.get("chapter", "pending"),
        },
        {
            "stage": "Summary",
            "model": "Extractive TF-IDF (proxy LLM)",
            "dataset": "TED",
            "metrics": "ROUGE-L, BERTScore, Factuality, Coverage",
            "value": f"ROUGE-L={_fmt(sum_ted.get('rouge_l'))}",
            "status": status.get("summary", "pending"),
        },
    ]


def write_report(results: dict[str, Any], out_path: Path, **kwargs: Any) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_eval_report(results, **kwargs)
    out_path.write_text(text, encoding="utf-8")
    return out_path

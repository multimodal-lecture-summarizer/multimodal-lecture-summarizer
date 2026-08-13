"""Markdown report generator for thesis evaluation tables (Bang 1–12)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "TBD"
    if isinstance(v, float):
        if v != v:  # NaN
            return "N/A"
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v: Any, digits: int = 2) -> str:
    if v is None:
        return "TBD"
    if isinstance(v, float) and v != v:
        return "N/A"
    try:
        return f"{float(v) * 100:.{digits}f}" if abs(float(v)) <= 1.5 else f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(c) if not isinstance(c, str) else c for c in row) + " |")
    return "\n".join(lines)


def render_eval_report(results: dict[str, Any], *, title: str | None = None) -> str:
    """Build full markdown matching the TTTN/DATN evaluation doc structure."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = title or "Bảng đánh giá pipeline AI (không gồm RAG/Q&A)"
    parts: list[str] = [
        f"# {title}",
        "",
        f"*Generated: {now}*",
        "",
        "> Phạm vi: model/algorithm trong pipeline video. Không đo hardware/UI/auth/deploy.",
        "",
    ]

    # Bang 1 ASR
    parts.append("## Bảng 1. Đánh giá ASR Faster-Whisper")
    parts.append("")
    parts.append("**Mục đích:** chứng minh chọn Faster-Whisper (`base.en` / `small.en`) cho bài giảng tiếng Anh.")
    parts.append("")
    asr_rows = results.get("asr", [])
    parts.append(
        _md_table(
            ["Dataset", "Model", "WER (%)", "CER (%)", "RTF"],
            [
                [
                    r.get("dataset", "TBD"),
                    r.get("model", "TBD"),
                    _pct(r.get("wer"), 2) if isinstance(r.get("wer"), float) else _fmt(r.get("wer_pct", r.get("wer"))),
                    _pct(r.get("cer"), 2) if isinstance(r.get("cer"), float) else _fmt(r.get("cer_pct", r.get("cer"))),
                    _fmt(r.get("rtf")),
                ]
                for r in (asr_rows or [{"dataset": "TED-LIUM", "model": "base.en"}, {"dataset": "TED-LIUM", "model": "small.en"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("asr_conclusion", "_Kết luận: TBD — chạy eval để điền._"))
    parts.append("")

    # Bang 2 VAD
    parts.append("## Bảng 2. Đánh giá VAD / Silence Filtering")
    parts.append("")
    parts.append("**Mục đích:** lọc silence làm transcript sạch hơn, không cắt nhầm speech.")
    parts.append("")
    vad_rows = results.get("vad", [])
    parts.append(
        _md_table(
            ["Video", "Tổng thời lượng (s)", "Speech thực tế (s)", "Speech detected (s)", "Precision", "Recall", "False Cut Rate"],
            [
                [
                    r.get("video", "TBD"),
                    _fmt(r.get("duration_sec")),
                    _fmt(r.get("speech_ref_sec")),
                    _fmt(r.get("speech_pred_sec")),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("false_cut_rate")),
                ]
                for r in (vad_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("vad_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 3 Scene
    parts.append("## Bảng 3. Đánh giá Scene Detection (PySceneDetect)")
    parts.append("")
    scene_rows = results.get("scene", [])
    parts.append(
        _md_table(
            ["Video", "Scene thật", "Scene phát hiện", "Precision", "Recall", "F1"],
            [
                [
                    r.get("video", "TBD"),
                    _fmt(r.get("n_ref"), 0),
                    _fmt(r.get("n_pred"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                ]
                for r in (scene_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("scene_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 4 Keyframe
    parts.append("## Bảng 4. Đánh giá Keyframe & CLIP Filtering")
    parts.append("")
    kf_rows = results.get("keyframe", [])
    parts.append(
        _md_table(
            ["Video", "Keyframe ban đầu", "Keyframe sau lọc", "Precision", "Recall", "F1", "Tỷ lệ nén"],
            [
                [
                    r.get("video", "TBD"),
                    _fmt(r.get("n_before"), 0),
                    _fmt(r.get("n_after"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                    _fmt(r.get("compression_ratio")),
                ]
                for r in (kf_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("keyframe_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 5 OCR
    parts.append("## Bảng 5. Đánh giá OCR PaddleOCR")
    parts.append("")
    ocr_rows = results.get("ocr", [])
    parts.append(
        _md_table(
            ["Ảnh slide", "Số dòng chữ thật", "Số dòng OCR đúng", "CER (%)", "Word Acc"],
            [
                [
                    r.get("image", "TBD"),
                    _fmt(r.get("n_lines_ref"), 0),
                    _fmt(r.get("n_lines_ok"), 0),
                    _pct(r.get("cer"), 2) if isinstance(r.get("cer"), float) else _fmt(r.get("cer")),
                    _fmt(r.get("word_accuracy")),
                ]
                for r in (ocr_rows or [{"image": "Slide 1"}, {"image": "Slide 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("ocr_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 6 Caption
    parts.append("## Bảng 6. Đánh giá Caption Florence-2")
    parts.append("")
    cap_rows = results.get("caption", [])
    parts.append(
        _md_table(
            ["Keyframe", "Caption sinh ra", "Đúng nội dung?", "Hallucination"],
            [
                [
                    r.get("keyframe", "TBD"),
                    (r.get("caption") or "TBD")[:80],
                    "Yes" if r.get("content_ok") else ("No" if r.get("content_ok") is False else "TBD"),
                    "Yes" if r.get("hallucinated") else ("No" if r.get("hallucinated") is False else "TBD"),
                ]
                for r in (cap_rows or [{"keyframe": "Keyframe 1"}, {"keyframe": "Keyframe 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("caption_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 7 Timeline
    parts.append("## Bảng 7. Đánh giá Timeline Alignment")
    parts.append("")
    tl_rows = results.get("timeline", [])
    parts.append(
        _md_table(
            ["Video", "Segment transcript", "Slide/keyframe đúng", "Slide/keyframe predicted", "Accuracy", "MAE (s)"],
            [
                [
                    r.get("video", "TBD"),
                    _fmt(r.get("n_segments"), 0),
                    _fmt(r.get("n_correct"), 0),
                    _fmt(r.get("n_predicted"), 0),
                    _fmt(r.get("accuracy")),
                    _fmt(r.get("mae_sec")),
                ]
                for r in (tl_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("timeline_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 8 Chapter
    parts.append("## Bảng 8. Đánh giá Chapter Segmentation")
    parts.append("")
    ch_rows = results.get("chapter", [])
    parts.append(
        _md_table(
            ["Video", "Số chapter chuẩn", "Số chapter sinh ra", "Boundary P", "Boundary R", "Boundary F1", "MAE (s)"],
            [
                [
                    r.get("video", "TBD"),
                    _fmt(r.get("n_ref"), 0),
                    _fmt(r.get("n_pred"), 0),
                    _fmt(r.get("precision")),
                    _fmt(r.get("recall")),
                    _fmt(r.get("f1")),
                    _fmt(r.get("mae")),
                ]
                for r in (ch_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("chapter_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 9 Summary
    parts.append("## Bảng 9. Đánh giá Summary LLM")
    parts.append("")
    sum_rows = results.get("summary", [])
    parts.append(
        _md_table(
            ["Video", "Input sử dụng", "ROUGE-L", "BERTScore", "Factuality", "Coverage"],
            [
                [
                    r.get("video", "TBD"),
                    r.get("input", "Transcript + OCR + caption"),
                    _fmt(r.get("rouge_l")),
                    _fmt(r.get("bertscore_f1")),
                    _fmt(r.get("factuality")),
                    _fmt(r.get("coverage")),
                ]
                for r in (sum_rows or [{"video": "Video 1"}, {"video": "Video 2"}])
            ],
        )
    )
    parts.append("")
    parts.append(results.get("summary_conclusion", "_Kết luận: TBD._"))
    parts.append("")

    # Bang 10 Ablation (RAG/Q&A table removed)
    parts.append("## Bảng 10. Ablation: đóng góp của từng modality")
    parts.append("")
    ab_rows = results.get("ablation", [])
    parts.append(
        _md_table(
            ["Talk", "Cấu hình", "Input", "Summary score"],
            [
                [
                    r.get("video", "TED"),
                    r.get("config", "TBD"),
                    r.get("input", "TBD"),
                    _fmt(r.get("summary_score")),
                ]
                for r in (
                    ab_rows
                    or [
                        {"config": "Audio only", "input": "Transcript"},
                        {"config": "Visual only", "input": "OCR + caption"},
                        {"config": "Audio + Visual", "input": "Transcript + OCR + caption"},
                    ]
                )
            ],
        )
    )
    parts.append("")
    parts.append(results.get("ablation_conclusion", "_Kết luận: TBD — bảng quan trọng để bảo vệ tính multimodal._"))
    parts.append("")

    # Bang 11 stage summary
    parts.append("## Bảng 11. Tổng hợp metric theo stage")
    parts.append("")
    stage_rows = results.get("stages", [])
    default_stages = [
        ("ASR", "Faster-Whisper", "TED-LIUM / video lecture", "WER, CER, RTF"),
        ("VAD", "Voice Activity Detection", "Video lecture annotate", "Precision, Recall, F1"),
        ("Scene", "PySceneDetect", "TVSum / lecture annotate", "Precision, Recall, F1"),
        ("Keyframe", "CLIP filtering", "TVSum/SumMe / lecture", "Precision, Recall, F1"),
        ("OCR", "PaddleOCR", "TED keyframe + aligned transcript (weak GT)", "CER, Word accuracy"),
        ("Caption", "OCR-grounded heuristic / Florence-2", "TED keyframe + hallucination heuristic", "Accuracy, Hallucination rate"),
        ("Timeline", "TF-IDF + temporal proximity", "TED-LIUM utterances ↔ scenes", "Accuracy, MAE"),
        ("Chapter", "TF-IDF semantic shift / 60s fallback", "TED-LIUM 60s slices", "Boundary F1"),
        ("Summary", "Extractive TF-IDF", "TED-LIUM window-lead reference", "ROUGE-L, BERTScore, factuality"),
    ]
    if not stage_rows:
        stage_rows = [
            {
                "stage": s,
                "model": m,
                "dataset": d,
                "metrics": met,
                "status": results.get("stage_status", {}).get(s.lower().split("/")[0], "pending"),
            }
            for s, m, d, met in default_stages
        ]
    parts.append(
        _md_table(
            ["Stage", "Model/Algorithm", "Dataset/Method", "Metrics", "Status"],
            [
                [
                    r.get("stage", "TBD"),
                    r.get("model", "TBD"),
                    r.get("dataset", "TBD"),
                    r.get("metrics", "TBD"),
                    r.get("status", "pending"),
                ]
                for r in stage_rows
            ],
        )
    )
    parts.append("")
    parts.append("### Thứ tự ưu tiên nếu thiếu thời gian")
    parts.append("")
    parts.append("1. ASR (WER, CER, RTF)")
    parts.append("2. Scene/keyframe (P/R/F1)")
    parts.append("3. OCR/Caption (CER, human score, hallucination)")
    parts.append("4. Timeline/chapter (MAE, boundary F1)")
    parts.append("5. Summary (factuality, coverage, ROUGE-L/BERTScore)")
    parts.append("")
    return "\n".join(parts)


def write_report(results: dict[str, Any], out_path: Path, **kwargs: Any) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = render_eval_report(results, **kwargs)
    out_path.write_text(text, encoding="utf-8")
    return out_path

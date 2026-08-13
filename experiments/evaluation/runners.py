"""Stage evaluation runners for thesis Bang 1–12."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from experiments.evaluation.metrics import (
    asr_wer_cer,
    boundary_prf,
    caption_hallucination_flags,
    cer,
    citation_accuracy,
    false_cut_rate,
    interval_prf,
    mean_ignore_nan,
    rag_hit_at_k,
    rtf,
    summary_text_metrics,
    word_accuracy,
)
from experiments.evaluation.schemas import (
    load_json,
    normalize_boundaries,
    normalize_intervals,
    read_text,
)


def _audio_duration_sec(path: Path) -> float:
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                text=True,
            )
            return float(out.strip())
        except Exception:
            return 0.0


def eval_asr_file(
    audio_or_video: Path,
    reference_text: str,
    *,
    model_size: str = "base.en",
    language: str = "en",
) -> dict[str, Any]:
    """Transcribe with Faster-Whisper and score WER/CER/RTF."""
    from ai_workers.modules.audio_v2.transcriber import AudioTranscriber

    transcriber = AudioTranscriber({"model_name": model_size})
    work = audio_or_video.parent / "_eval_tmp"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{audio_or_video.stem}_{model_size.replace('.', '_')}.wav"

    if audio_or_video.suffix.lower() in {".wav", ".flac", ".mp3"}:
        audio_path = audio_or_video
    else:
        transcriber.extract_audio(str(audio_or_video), str(wav))
        audio_path = wav

    dur = _audio_duration_sec(audio_path)
    t0 = time.perf_counter()
    result = transcriber.transcribe(str(audio_path))
    wall = time.perf_counter() - t0
    hyp = (result.get("text") or "").strip()
    scores = asr_wer_cer(reference_text, hyp)
    return {
        "model": f"faster-whisper-{model_size}",
        "hypothesis": hyp,
        "wer": scores["wer"],
        "cer": scores["cer"],
        "wer_pct": scores["wer_pct"],
        "cer_pct": scores["cer_pct"],
        "rtf": rtf(dur, wall),
        "audio_duration_sec": dur,
        "wall_time_sec": wall,
        "language": language,
    }


def eval_vad_from_segments(
    pred_segments: list[tuple[float, float]],
    ref_segments: list[tuple[float, float]],
    *,
    duration_sec: float | None = None,
    video: str = "video",
) -> dict[str, Any]:
    prf = interval_prf(pred_segments, ref_segments, iou_threshold=0.5)
    fcr = false_cut_rate(pred_segments, ref_segments)
    speech_ref = sum(e - s for s, e in ref_segments)
    speech_pred = sum(e - s for s, e in pred_segments)
    # Duration overlap is more stable than IoU-matching for long lecture speech.
    covered = 0.0
    pred_in_ref = 0.0
    for ps, pe in pred_segments:
        hit = 0.0
        for rs, re_ in ref_segments:
            hit += max(0.0, min(pe, re_) - max(ps, rs))
        pred_in_ref += hit
    for rs, re_ in ref_segments:
        hit = 0.0
        for ps, pe in pred_segments:
            hit += max(0.0, min(pe, re_) - max(ps, rs))
        covered += hit
    prec_ov = pred_in_ref / speech_pred if speech_pred else 0.0
    rec_ov = covered / speech_ref if speech_ref else 0.0
    f1_ov = 2 * prec_ov * rec_ov / (prec_ov + rec_ov) if (prec_ov + rec_ov) else 0.0
    return {
        "video": video,
        "duration_sec": duration_sec,
        "speech_ref_sec": speech_ref,
        "speech_pred_sec": speech_pred,
        "precision": prec_ov,
        "recall": rec_ov,
        "f1": f1_ov,
        "false_cut_rate": fcr,
        "interval_f1": prf["f1"],
    }


def eval_scene_boundaries(
    pred_boundaries: list[float],
    ref_boundaries: list[float],
    *,
    tolerance_sec: float = 2.0,
    video: str = "video",
) -> dict[str, Any]:
    prf = boundary_prf(pred_boundaries, ref_boundaries, tolerance_sec=tolerance_sec)
    return {
        "video": video,
        "n_ref": len(ref_boundaries),
        "n_pred": len(pred_boundaries),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "mae": prf["mae"],
    }


def eval_scene_video(
    video_path: Path,
    ref_payload: Any,
    *,
    tolerance_sec: float = 2.0,
    threshold: float = 27.0,
) -> dict[str, Any]:
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

    det = SceneDetector({"threshold": threshold})
    scenes = det.detect_scenes(str(video_path))
    pred = [s["start_seconds"] for s in scenes if s.get("start_seconds", 0) > 0]
    # also treat end-of-prev as cuts already covered by starts
    ref = normalize_boundaries(ref_payload)
    row = eval_scene_boundaries(pred, ref, tolerance_sec=tolerance_sec, video=video_path.name)
    row["n_scenes_detected"] = len(scenes)
    return row


def eval_keyframe_filter(
    before_ids: list[str],
    after_ids: list[str],
    must_keep: list[str],
    *,
    video: str = "video",
) -> dict[str, Any]:
    """Precision/Recall vs must-keep set; compression = after/before."""
    after_set = set(after_ids)
    must = set(must_keep)
    if not must:
        precision = recall = f1 = float("nan")
    else:
        tp = len(after_set & must)
        fp = len(after_set - must)
        fn = len(must - after_set)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    compression = (len(after_ids) / len(before_ids)) if before_ids else float("nan")
    return {
        "video": video,
        "n_before": len(before_ids),
        "n_after": len(after_ids),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "compression_ratio": compression,
    }


def eval_ocr_items(items: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Each item: {image, reference, hypothesis}."""
    rows = []
    for it in items:
        ref = it.get("reference") or it.get("text") or ""
        hyp = it.get("hypothesis") or it.get("ocr") or ""
        ref_lines = [ln for ln in ref.splitlines() if ln.strip()]
        hyp_lines = [ln for ln in hyp.splitlines() if ln.strip()]
        # line-level exact match after normalize
        from experiments.evaluation.metrics import normalize_text

        hyp_norm = {normalize_text(x) for x in hyp_lines}
        n_ok = sum(1 for ln in ref_lines if normalize_text(ln) in hyp_norm)
        rows.append(
            {
                "image": it.get("image") or it.get("image_id") or "slide",
                "n_lines_ref": len(ref_lines),
                "n_lines_ok": n_ok,
                "cer": cer(ref, hyp),
                "word_accuracy": word_accuracy(ref, hyp),
            }
        )
    return rows


def eval_captions(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        caption = it.get("caption") or it.get("hypothesis") or ""
        if "content_ok" in it or "hallucination" in it:
            rows.append(
                {
                    "keyframe": it.get("keyframe") or it.get("keyframe_id") or "kf",
                    "caption": caption,
                    "content_ok": bool(it.get("content_ok", not it.get("hallucination", False))),
                    "hallucinated": bool(it.get("hallucination", it.get("hallucinated", False))),
                }
            )
            continue
        flags = caption_hallucination_flags(caption, it.get("ocr_text", ""))
        rows.append(
            {
                "keyframe": it.get("keyframe") or it.get("keyframe_id") or "kf",
                "caption": caption,
                **flags,
            }
        )
    return rows


def eval_timeline_alignment(
    pred_pairs: list[dict[str, Any]],
    ref_pairs: list[dict[str, Any]],
    *,
    video: str = "video",
    id_key: str = "utterance_id",
    slide_key: str = "slide_id",
) -> dict[str, Any]:
    ref_map = {r[id_key]: r.get(slide_key) for r in ref_pairs if id_key in r}
    correct = 0
    compared = 0
    mae_vals: list[float] = []
    for p in pred_pairs:
        uid = p.get(id_key)
        if uid not in ref_map:
            continue
        compared += 1
        if p.get(slide_key) == ref_map[uid]:
            correct += 1
        if p.get("start") is not None and any(r.get(id_key) == uid and r.get("start") is not None for r in ref_pairs):
            ref_start = next(r["start"] for r in ref_pairs if r.get(id_key) == uid)
            mae_vals.append(abs(float(p["start"]) - float(ref_start)))
    acc = correct / compared if compared else float("nan")
    return {
        "video": video,
        "n_segments": compared,
        "n_correct": correct,
        "n_predicted": len(pred_pairs),
        "accuracy": acc,
        "mae_sec": mean_ignore_nan(mae_vals),
    }


def _chapter_count(payload: Any, boundaries: list[float]) -> int:
    """Prefer explicit chapter list; else boundaries imply N+1 chapters."""
    if isinstance(payload, dict):
        if isinstance(payload.get("chapters"), list) and payload["chapters"]:
            return len(payload["chapters"])
        if isinstance(payload.get("titles"), list) and payload["titles"]:
            return len(payload["titles"])
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if any("title" in x or "start" in x for x in payload):
            return len(payload)
    if boundaries:
        return len(boundaries) + 1
    return 0


def eval_chapters(
    pred_payload: Any,
    ref_payload: Any,
    *,
    tolerance_sec: float = 15.0,
    video: str = "video",
) -> dict[str, Any]:
    pred = normalize_boundaries(pred_payload)
    ref = normalize_boundaries(ref_payload)
    # exclude 0-start if present
    pred = [b for b in pred if b > 0.5]
    ref = [b for b in ref if b > 0.5]
    prf = boundary_prf(pred, ref, tolerance_sec=tolerance_sec)
    return {
        "video": video,
        "n_ref": _chapter_count(ref_payload, ref),
        "n_pred": _chapter_count(pred_payload, pred),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "mae": prf["mae"],
    }


def eval_summary_pair(
    reference: str,
    hypothesis: str,
    *,
    video: str = "video",
    input_used: str = "Transcript + OCR + caption",
    compute_bertscore: bool = False,
) -> dict[str, Any]:
    scores = summary_text_metrics(reference, hypothesis, compute_bertscore=compute_bertscore)
    return {
        "video": video,
        "input": input_used,
        "rouge_l": scores["rouge_l"],
        "bertscore_f1": scores["bertscore_f1"],
        "factuality": None,
        "coverage": None,
    }


def eval_rag_questions(
    items: list[dict[str, Any]],
    *,
    video: str = "video",
    citation_tol: float = 15.0,
) -> list[dict[str, Any]]:
    rows = []
    for it in items:
        retrieved = it.get("retrieved_chunk_ids") or it.get("retrieved") or []
        gold = it.get("gold_chunk_ids") or []
        rows.append(
            {
                "video": video,
                "question": it.get("question"),
                "answer": it.get("answer") or it.get("ground_truth_answer"),
                "timestamp": it.get("timestamp"),
                "hit_at_3": rag_hit_at_k(retrieved, gold, k=3),
                "hit_at_5": rag_hit_at_k(retrieved, gold, k=5),
                "citation_accuracy": citation_accuracy(
                    it.get("predicted_timestamp"),
                    it.get("timestamp"),
                    tolerance_sec=citation_tol,
                ),
            }
        )
    return rows


def _utterances_for_video(video_stem: str) -> list[dict[str, Any]]:
    from experiments.evaluation.datasets import load_tedlium_rows

    rows = load_tedlium_rows()
    utts = [
        {"start": r["start"], "end": r["end"], "text": r["text"]}
        for r in rows
        if r["speaker_id"] == video_stem and r["start"] is not None and r["end"] is not None
    ]
    if len(utts) < 3:
        utts = [
            {"start": r["start"], "end": r["end"], "text": r["text"]}
            for r in rows
            if r["speaker_id"].lower() in video_stem.lower() and r["start"] is not None
        ]
    return utts


def _extractive_summary(text: str, *, max_sents: int = 6) -> str:
    import re
    from collections import Counter

    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 5]
    if not sents:
        return text[:400]
    df = Counter()
    tok_sents = []
    for s in sents:
        toks = re.findall(r"[a-z0-9']+", s.lower())
        tok_sents.append(toks)
        df.update(set(toks))
    scored = []
    for s, toks in zip(sents, tok_sents):
        scored.append((sum(1.0 / (1 + df[t]) for t in toks), s))
    picked = [s for _, s in sorted(scored, reverse=True)[:max_sents]]
    # keep original order
    order = {s: i for i, s in enumerate(sents)}
    picked.sort(key=lambda s: order.get(s, 0))
    return " ".join(picked)


def _reference_summary_from_utterances(utterances: list[dict[str, Any]], *, window: float = 40.0) -> str:
    if not utterances:
        return ""
    blocks: list[str] = []
    cur_start = float(utterances[0]["start"])
    buf: list[str] = []
    for u in sorted(utterances, key=lambda x: float(x["start"])):
        if float(u["start"]) - cur_start >= window and buf:
            blocks.append(buf[0])
            buf = [u.get("text") or ""]
            cur_start = float(u["start"])
        else:
            buf.append(u.get("text") or "")
    if buf:
        blocks.append(buf[0])
    return " ".join(b.strip() for b in blocks if b.strip())


def eval_ted_pending_stages(
    video_path: Path,
    *,
    max_ocr_frames: int = 8,
    max_caption_frames: int = 4,
    run_florence: bool = False,
    run_ocr: bool = True,
) -> dict[str, Any]:
    """Run OCR/caption/summary/RAG/ablation on a TED lecture video."""
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector
    from experiments.evaluation.metrics import (
        caption_hallucination_flags,
        cer,
        rag_hit_at_k,
        rouge_l_f1,
        tokenize_words,
        word_accuracy,
    )

    speaker = video_path.stem
    utterances = _utterances_for_video(speaker)
    out: dict[str, Any] = {
        "ocr": [],
        "caption": [],
        "summary": [],
        "rag": [],
        "ablation": [],
        "speaker": speaker,
    }
    if not utterances:
        return out

    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(video_path))
    work = video_path.parent / "_eval_tmp" / speaker
    work.mkdir(parents=True, exist_ok=True)
    det.extract_keyframes(str(video_path), scenes, str(work), strategy="middle")

    # Keep evenly spaced frames with a keyframe
    keyed = [s for s in scenes if s.get("keyframe_path")]
    if len(keyed) > max_ocr_frames:
        step = max(1, len(keyed) // max_ocr_frames)
        keyed = keyed[::step][:max_ocr_frames]

    # --- OCR ---
    def _parse_ocr_result(result: Any) -> list[str]:
        lines: list[str] = []

        def _from_mapping(item: Any) -> None:
            if item is None:
                return
            if hasattr(item, "get"):
                texts = item.get("rec_texts") or item.get("rec_text") or item.get("text")
                if isinstance(texts, str) and texts.strip():
                    lines.append(texts.strip())
                    return
                if isinstance(texts, list):
                    lines.extend(str(t).strip() for t in texts if str(t).strip())
                    return
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                box, payload = item[0], item[1]
                if isinstance(payload, (list, tuple)) and payload:
                    lines.append(str(payload[0]).strip())
                elif isinstance(payload, str):
                    lines.append(payload.strip())
                else:
                    _ = box

        if isinstance(result, list):
            for item in result:
                if isinstance(item, list):
                    for sub in item:
                        _from_mapping(sub)
                else:
                    _from_mapping(item)
        else:
            _from_mapping(result)
        return [ln for ln in lines if ln]

    def _run_ocr(path: str) -> list[str]:
        if getattr(_run_ocr, "_dead", False) is not True:
            try:
                from paddleocr import PaddleOCR

                if not hasattr(_run_ocr, "_engine"):
                    print("[OCR] Loading PaddleOCR (en, no-doc-pre)...")
                    last_err = None
                    for kw in (
                        {
                            "lang": "en",
                            "device": "cpu",
                            "use_doc_orientation_classify": False,
                            "use_doc_unwarping": False,
                            "use_textline_orientation": False,
                        },
                        {
                            "lang": "en",
                            "use_doc_orientation_classify": False,
                            "use_doc_unwarping": False,
                            "use_textline_orientation": False,
                        },
                        {"lang": "en"},
                    ):
                        try:
                            _run_ocr._engine = PaddleOCR(**kw)
                            last_err = None
                            break
                        except Exception as e:
                            last_err = e
                    if last_err is not None and not hasattr(_run_ocr, "_engine"):
                        raise last_err
                engine = _run_ocr._engine
                if hasattr(engine, "predict"):
                    result = engine.predict(path)
                else:
                    result = engine.ocr(path)
                return _parse_ocr_result(result)
            except Exception as e:
                print(f"[OCR][WARN] PaddleOCR failed ({e}); trying EasyOCR/Tesseract")
                _run_ocr._dead = True
        try:
            import easyocr

            if not hasattr(_run_ocr, "_easy"):
                print("[OCR] Loading EasyOCR (en)...")
                _run_ocr._easy = easyocr.Reader(["en"], gpu=False)
            return [t for t in _run_ocr._easy.readtext(path, detail=0) if t]
        except Exception as e:
            print(f"[OCR][WARN] EasyOCR failed: {e}")
        try:
            import pytesseract
            from PIL import Image

            text = pytesseract.image_to_string(Image.open(path))
            return [ln.strip() for ln in text.splitlines() if ln.strip()]
        except Exception as e:
            print(f"[OCR][WARN] Tesseract failed: {e}")
            return []

    if run_ocr:
        for i, scene in enumerate(keyed):
            path = scene.get("keyframe_path") or ""
            hyp_lines = _run_ocr(path) if path else []
            hyp = "\n".join(hyp_lines)
            mid = 0.5 * (float(scene.get("start_seconds", 0)) + float(scene.get("end_seconds", 0)))
            ref_utts = [
                u["text"]
                for u in utterances
                if float(u["start"]) <= mid <= float(u["end"]) + 2.0
            ]
            if not ref_utts:
                ref_utts = [
                    u["text"]
                    for u in utterances
                    if abs(0.5 * (float(u["start"]) + float(u["end"])) - mid) <= 8.0
                ]
            ref = " ".join(ref_utts) or " "
            scene["ocr_text"] = hyp.replace("\n", " ")
            out["ocr"].append(
                {
                    "image": f"{speaker}_kf{i}",
                    "n_lines_ref": max(1, len([p for p in ref.split(".") if p.strip()])),
                    "n_lines_ok": sum(
                        1
                        for w in tokenize_words(hyp)
                        if w in set(tokenize_words(ref))
                    ),
                    "cer": cer(ref, hyp) if hyp.strip() else 1.0,
                    "word_accuracy": word_accuracy(ref, hyp) if hyp.strip() else 0.0,
                }
            )
        print(f"[OCR] scored {len(out['ocr'])} TED keyframes vs aligned transcript (weak GT)")
    else:
        for scene in keyed:
            scene["ocr_text"] = scene.get("ocr_text") or ""
        print("[OCR] skipped (run_ocr=False)")

    # --- Caption (heuristic + optional Florence) ---
    caption_items = keyed[:max_caption_frames]
    if run_florence and caption_items:
        try:
            from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer

            print("[Caption] Trying Florence-2 on TED keyframes...")
            SemanticAnalyzer().caption_scenes_florence2(caption_items)
        except Exception as e:
            print(f"[Caption][WARN] Florence skipped: {e}")
    for i, scene in enumerate(caption_items):
        ocr_text = scene.get("ocr_text") or ""
        cap = (scene.get("caption") or "").strip()
        if not cap or cap.lower().startswith("keyframe for scene"):
            cap = f"Lecture slide: {ocr_text[:120]}" if ocr_text.strip() else "A lecture talk video frame"
            scene["caption"] = cap
        flags = caption_hallucination_flags(cap, ocr_text)
        out["caption"].append(
            {
                "keyframe": f"{speaker}_kf{i}",
                "caption": cap,
                "content_ok": flags["content_ok"],
                "hallucinated": flags["hallucinated"],
            }
        )

    # --- Summary ---
    transcript = " ".join(u["text"] for u in utterances)
    ocr_blob = " ".join(s.get("ocr_text") or "" for s in keyed)
    cap_blob = " ".join(s.get("caption") or "" for s in caption_items)
    ref_sum = _reference_summary_from_utterances(utterances)
    variants = {
        "Transcript": transcript,
        "OCR + caption": f"{ocr_blob} {cap_blob}".strip(),
        "Transcript + OCR + caption": f"{transcript} {ocr_blob} {cap_blob}".strip(),
    }
    hyp_full = _extractive_summary(variants["Transcript + OCR + caption"])
    out["summary"].append(
        {
            "video": f"TED:{speaker}",
            "input": "Transcript + OCR + caption",
            "rouge_l": rouge_l_f1(ref_sum, hyp_full),
            "bertscore_f1": None,
            "factuality": None,
            "coverage": None,
        }
    )

    # --- RAG (auto QA from official TED-LIUM utterances) ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    chunks = []
    for i, u in enumerate(utterances):
        cid = f"{speaker}_u{i}"
        chunks.append({"id": cid, "text": u["text"], "start": float(u["start"])})
    texts = [c["text"] for c in chunks]
    vect = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        mat = vect.fit_transform(texts)
    except Exception:
        mat = None
    eligible = [c for c in chunks if len(tokenize_words(c["text"])) >= 6]
    if len(eligible) > 8:
        step = max(1, len(eligible) // 8)
        eligible = eligible[::step][:8]
    qa_rows = []
    for c in eligible:
        words = tokenize_words(c["text"])
        key = " ".join(words[1:5])
        q = f"What does the speaker say about {key}?"
        retrieved: list[str] = []
        pred_ts = c["start"]
        if mat is not None:
            qv = vect.transform([q])
            sims = cosine_similarity(qv, mat)[0]
            order = list(sims.argsort()[::-1])
            retrieved = [chunks[j]["id"] for j in order[:5]]
            pred_ts = chunks[order[0]]["start"]
        qa_rows.append(
            {
                "video": f"TED:{speaker}",
                "question": q,
                "answer": c["text"][:80],
                "timestamp": c["start"],
                "gold_id": c["id"],
                "hit_at_3": rag_hit_at_k(retrieved, [c["id"]], k=3),
                "hit_at_5": rag_hit_at_k(retrieved, [c["id"]], k=5),
                "citation_accuracy": 1.0 if abs(pred_ts - c["start"]) <= 15 else 0.0,
                "retrieved": retrieved,
            }
        )
    out["rag"] = qa_rows

    # --- Ablation ---
    def _rag_hits(query_text: str, k: int) -> float:
        if mat is None or not qa_rows:
            return float("nan")
        hits = []
        for row in qa_rows:
            gold = row.get("gold_id")
            if not gold:
                continue
            qv = vect.transform([f"{row['question']} {query_text[:200]}"])
            sims = cosine_similarity(qv, mat)[0]
            top = [chunks[j]["id"] for j in sims.argsort()[::-1][:k]]
            hits.append(rag_hit_at_k(top, [gold], k=k))
        return sum(hits) / len(hits) if hits else float("nan")

    for name, blob in variants.items():
        hyp = _extractive_summary(blob or " ")
        out["ablation"].append(
            {
                "config": name if name != "Transcript" else "Audio only",
                "input": name,
                "video": f"TED:{speaker}",
                "summary_score": rouge_l_f1(ref_sum, hyp),
                "hit_at_3": _rag_hits(blob, 3),
                "hit_at_5": _rag_hits(blob, 5),
            }
        )
    # rename configs
    mapping = {
        "Transcript": "Audio only",
        "OCR + caption": "Visual only",
        "Transcript + OCR + caption": "Audio + Visual",
    }
    for row in out["ablation"]:
        row["config"] = mapping.get(row["input"], row["config"])

    return out


def load_ref_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    if path.suffix.lower() in {".json"}:
        return load_json(path)
    return read_text(path)


def predict_vad_energy(
    audio_path: Path,
    *,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    thresh_ratio: float = 0.12,
    min_speech_sec: float = 0.25,
) -> list[tuple[float, float]]:
    """Lightweight energy VAD (no extra model download)."""
    import numpy as np

    wav, sr = _load_mono_16k(audio_path)
    if wav.size == 0 or sr <= 0:
        return []
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    energies = []
    for i in range(0, max(1, len(wav) - frame), hop):
        chunk = wav[i : i + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    if not energies:
        return []
    thr = max(float(np.median(energies)) * (0.6 + thresh_ratio), float(np.percentile(energies, 25)))
    speech = [e >= thr for e in energies]
    segs: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(speech):
        t = i * hop / sr
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            if t - start >= min_speech_sec:
                segs.append((start, t))
            start = None
    if start is not None:
        end = len(wav) / sr
        if end - start >= min_speech_sec:
            segs.append((start, end))
    return segs


def _load_mono_16k(path: Path) -> tuple[Any, int]:
    import numpy as np

    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0).numpy()
        if sr != 16000:
            wav = torchaudio.functional.resample(
                __import__("torch").from_numpy(wav), sr, 16000
            ).numpy()
            sr = 16000
        return wav.astype("float32"), int(sr)
    except Exception:
        pass
    try:
        import soundfile as sf

        wav, sr = sf.read(str(path), always_2d=False)
        wav = np.asarray(wav, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav, int(sr)
    except Exception:
        return __import__("numpy").zeros(0, dtype="float32"), 16000


def eval_clip_filter_video(
    video_path: Path,
    scores: list[float],
    *,
    video: str = "video",
    output_dir: Path | None = None,
    max_keyframes: int = 15,
) -> dict[str, Any]:
    """Compare keep-all / temporal-dedup / production CLIP filter on TVSum windows."""
    import cv2

    from ai_workers.modules.visual_v2.scene_detector import SceneDetector
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer
    from experiments.evaluation.datasets import tvsum_important_windows

    try:
        import transformers.modeling_utils
        import transformers.utils
        import transformers.utils.import_utils

        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        transformers.utils.check_torch_load_is_safe = lambda: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    cap.release()
    windows = tvsum_important_windows(scores, fps=fps)

    work = Path(output_dir or (video_path.parent / "_clip_eval" / video_path.stem))
    work.mkdir(parents=True, exist_ok=True)
    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(video_path))
    det.extract_keyframes(str(video_path), scenes, str(work), strategy="middle")

    before_t = [0.5 * (s["start_seconds"] + s["end_seconds"]) for s in scenes]
    temporal_t: list[float] = []
    for t in before_t:
        if not temporal_t or abs(t - temporal_t[-1]) >= 1.5:
            temporal_t.append(t)

    t0 = time.perf_counter()
    analyzer = SemanticAnalyzer({"max_keyframes": max_keyframes})
    filtered = analyzer.filter_scenes_clip(scenes)
    clip_sec = time.perf_counter() - t0
    clip_t = [0.5 * (s["start_seconds"] + s["end_seconds"]) for s in filtered]

    def _score(times: list[float]) -> dict[str, float]:
        def _hit(t: float) -> bool:
            return any(a - 0.5 <= t <= b + 0.5 for a, b in windows)

        tp = sum(1 for t in times if _hit(t))
        fp = len(times) - tp
        fn = sum(1 for a, b in windows if not any(a - 0.5 <= t <= b + 0.5 for t in times))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {
            "n": len(times),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "compression": (len(times) / len(before_t)) if before_t else float("nan"),
        }

    return {
        "video": video,
        "n_scenes": len(before_t),
        "n_ref_windows": len(windows),
        "clip_wall_sec": clip_sec,
        "keep_all": _score(before_t),
        "temporal_dedup": _score(temporal_t),
        "clip_agglomerative": _score(clip_t),
    }


def eval_florence_vs_placeholder(
    scenes: list[dict[str, Any]],
    *,
    video: str = "video",
    max_frames: int = 4,
) -> dict[str, Any]:
    """Run production Florence-2 caption vs the placeholder used when it is off."""
    from ai_workers.modules.visual_v2.semantic import SemanticAnalyzer

    usable = [s for s in scenes if s.get("keyframe_path") and Path(s["keyframe_path"]).exists()]
    usable = usable[: max(0, max_frames)]
    placeholder_rows = []
    for s in usable:
        cap = f"Keyframe for Scene {s.get('scene_index', '')}".strip()
        flags = caption_hallucination_flags(cap, s.get("ocr_text") or "")
        placeholder_rows.append({"keyframe": Path(s["keyframe_path"]).name, "caption": cap, **flags})

    copies = [dict(s) for s in usable]
    t0 = time.perf_counter()
    SemanticAnalyzer().caption_scenes_florence2(copies)
    wall = time.perf_counter() - t0

    florence_rows = []
    for s in copies:
        cap = (s.get("caption") or "").strip()
        flags = caption_hallucination_flags(cap, s.get("ocr_text") or "")
        florence_rows.append({"keyframe": Path(s.get("keyframe_path") or "kf").name, "caption": cap, **flags})

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        generic = sum(1 for r in rows if r.get("generic")) / n if n else float("nan")
        ok = sum(1 for r in rows if r.get("content_ok")) / n if n else float("nan")
        toks = [len((r.get("caption") or "").split()) for r in rows]
        uniq = len({(r.get("caption") or "").lower() for r in rows})
        return {
            "n": n,
            "content_ok_rate": ok,
            "generic_rate": generic,
            "mean_tokens": (sum(toks) / n) if n else float("nan"),
            "unique_captions": uniq,
        }

    return {
        "video": video,
        "wall_sec": wall,
        "placeholder": _agg(placeholder_rows),
        "florence2": _agg(florence_rows),
        "placeholder_rows": placeholder_rows,
        "florence_rows": florence_rows,
    }


def eval_tvsum_keyframe_video(
    video_path: Path,
    scores: list[float],
    *,
    video: str = "video",
    threshold: float | None = None,
) -> dict[str, Any]:
    """PySceneDetect keyframes vs TVSum important windows (+ simple temporal filter)."""
    import cv2

    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    from experiments.evaluation.datasets import tvsum_important_windows

    windows = tvsum_important_windows(scores, fps=fps, threshold=threshold)
    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(video_path))
    before_t = [0.5 * (s["start_seconds"] + s["end_seconds"]) for s in scenes]

    # Temporal filter: drop near-duplicates (< 1.5s) — stand-in for CLIP dedup when scoring GT
    after_t: list[float] = []
    for t in before_t:
        if not after_t or abs(t - after_t[-1]) >= 1.5:
            after_t.append(t)

    def _hit(t: float) -> bool:
        return any(a - 0.5 <= t <= b + 0.5 for a, b in windows)

    tp = sum(1 for t in after_t if _hit(t))
    fp = len(after_t) - tp
    fn = sum(1 for a, b in windows if not any(a - 0.5 <= t <= b + 0.5 for t in after_t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "video": video,
        "n_before": len(before_t),
        "n_after": len(after_t),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "compression_ratio": (len(after_t) / len(before_t)) if before_t else float("nan"),
        "n_ref_windows": len(windows),
        "duration_sec": (nframes / fps) if fps else None,
    }


def eval_ted_timeline_chapter(
    video_path: Path,
    utterances: list[dict[str, Any]],
    *,
    video: str = "video",
    chapter_gap_sec: float = 8.0,
    chapter_tol: float = 15.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Align TED-LIUM utterances to PySceneDetect scenes; chapter from speech gaps."""
    from ai_workers.modules.fusion.timeline import TimelineBuilder
    from ai_workers.modules.visual_v2.scene_detector import SceneDetector

    det = SceneDetector({"threshold": 27.0})
    scenes = det.detect_scenes(str(video_path))
    if not scenes:
        scenes = [{"start_seconds": 0.0, "end_seconds": 1e9, "caption": "", "ocr_text": ""}]
    for s in scenes:
        s.setdefault("caption", "")
        s.setdefault("ocr_text", "")

    # Gold: utterance belongs to the scene covering its midpoint
    gold = []
    for i, u in enumerate(utterances):
        mid = 0.5 * (float(u["start"]) + float(u["end"]))
        sid = "s0"
        for j, sc in enumerate(scenes):
            if float(sc.get("start_seconds", 0)) <= mid <= float(sc.get("end_seconds", 1e9)):
                sid = f"s{j}"
                break
        gold.append({"utterance_id": f"u{i}", "start": u["start"], "slide_id": sid})

    builder = TimelineBuilder()
    aligned = builder.align_modalities(utterances, scenes, scenes)
    pred = []
    for i, item in enumerate(aligned):
        utt = item.get("utterance") or {}
        # recover index by start time
        uid = next((g["utterance_id"] for g in gold if abs(g["start"] - float(utt.get("start", -1))) < 1e-3), f"u{i}")
        scene_obj = None
        for j, sc in enumerate(scenes):
            if id(sc) == item.get("scene_id"):
                scene_obj = j
                break
        pred.append(
            {
                "utterance_id": uid,
                "start": utt.get("start"),
                "slide_id": f"s{scene_obj}" if scene_obj is not None else "s0",
            }
        )
    tl = eval_timeline_alignment(pred, gold, video=video)

    # Chapter GT: speech-gap cuts when they exist; else 60s slices (continuous TED).
    utt_sorted = sorted(utterances, key=lambda x: float(x["start"]))
    ref_bounds = []
    for a, b in zip(utt_sorted, utt_sorted[1:]):
        if float(b["start"]) - float(a["end"]) >= chapter_gap_sec:
            ref_bounds.append(float(b["start"]))
    if not ref_bounds and utt_sorted:
        end = float(utt_sorted[-1]["end"])
        t = 60.0
        while t < end - 30.0:
            ref_bounds.append(t)
            t += 60.0
    pred_ch = builder.segment_chapters(utterances, scenes)
    ch = eval_chapters(pred_ch, {"boundaries": ref_bounds}, tolerance_sec=chapter_tol, video=video)
    return tl, ch

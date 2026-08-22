"""Quality gates for experimental multimodal pipeline (issues 5, 6, 7, 8)."""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

import cv2
import numpy as np


# --- Issue 5: ASR ---


def validate_audio_asr(
    audio_res: dict,
    expected_lang: str = "en",
    min_conf_thresh: float = 0.6,
) -> dict:
    segments = audio_res.get("segments", [])
    full_text = audio_res.get("text", "").strip()
    detected_lang = audio_res.get("language", "unknown")

    report = {
        "total_segments": len(segments),
        "total_chars": len(full_text),
        "detected_language": detected_lang,
        "language_mismatch": False,
        "low_confidence_segments": 0,
        "empty_segments": 0,
        "no_speech_ratio": 0.0,
        "average_confidence": 1.0,
        "quality_status": "HIGH",
        "visual_dominant_fallback": False,
        "recommended_mode": "audio_visual_balanced",
        "asr_weight": 0.75,
        "warnings": [],
    }

    if not segments or not full_text:
        report["quality_status"] = "FAILED"
        report["visual_dominant_fallback"] = True
        report["recommended_mode"] = "visual_only_emergency"
        report["asr_weight"] = 0.10
        report["warnings"].append("CRITICAL: Audio transcript is empty or extraction failed.")
        return report

    if expected_lang and detected_lang != expected_lang:
        report["language_mismatch"] = True
        report["warnings"].append(
            f"Language mismatch: detected '{detected_lang}', expected '{expected_lang}'."
        )

    conf_scores: list[float] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        seg["low_confidence_flag"] = False
        if not text:
            report["empty_segments"] += 1
            continue

        words = seg.get("words", [])
        if words:
            probs = [w.get("probability") for w in words if w.get("probability") is not None]
            seg_conf = sum(probs) / len(probs) if probs else 0.85
        else:
            seg_conf = 0.85

        seg["asr_confidence"] = round(float(seg_conf), 3)
        conf_scores.append(seg_conf)
        if seg_conf < min_conf_thresh:
            report["low_confidence_segments"] += 1
            seg["low_confidence_flag"] = True

    if conf_scores:
        report["average_confidence"] = float(sum(conf_scores) / len(conf_scores))

    report["no_speech_ratio"] = report["empty_segments"] / max(1, len(segments))
    low_ratio = report["low_confidence_segments"] / max(1, len(segments))

    if low_ratio > 0.40 or report["average_confidence"] < 0.65:
        report["quality_status"] = "LOW_QUALITY_WARNING"
        report["visual_dominant_fallback"] = True
        report["recommended_mode"] = "visual_dominant"
        report["asr_weight"] = 0.35
        report["warnings"].append(
            f"High low-confidence ASR ratio ({low_ratio * 100:.1f}%)."
        )
    elif low_ratio > 0.15:
        report["quality_status"] = "MEDIUM"
        report["asr_weight"] = 0.55

    if report["no_speech_ratio"] > 0.25:
        report["warnings"].append(
            f"High no-speech ratio ({report['no_speech_ratio'] * 100:.1f}%)."
        )

    return report


def build_asr_safe_segments(segments: list, report: dict, min_chars: int = 3) -> list:
    if not segments:
        return []

    safe: list = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if len(text) < min_chars:
            continue
        if report.get("visual_dominant_fallback") and seg.get("low_confidence_flag"):
            continue
        safe.append(seg)
    return safe if safe else segments


# --- Issue 6: Speaker ---


def validate_speaker_diarization(utterances: list, duration_sec: float) -> dict:
    if not utterances:
        return {
            "speaker_count": 0,
            "short_turn_ratio": 0.0,
            "switch_rate_per_min": 0.0,
            "reliability": "UNKNOWN",
            "warnings": ["No utterances for diarization."],
        }

    speakers = {u.get("speaker", "SPEAKER") for u in utterances}
    short_turns = 0
    switches = 0
    prev = None
    for u in utterances:
        dur = float(u.get("end", 0.0)) - float(u.get("start", 0.0))
        if dur < 1.2:
            short_turns += 1
        sp = u.get("speaker")
        if prev is not None and sp != prev:
            switches += 1
        prev = sp

    short_ratio = short_turns / max(1, len(utterances))
    switch_rate = switches / max(1.0, duration_sec / 60.0)

    reliability = "HIGH"
    warnings: list[str] = []
    if len(speakers) > 3 and short_ratio > 0.35:
        reliability = "LOW"
        warnings.append("Many speakers with frequent short turns; labels may be unstable.")
    elif short_ratio > 0.45 or switch_rate > 25:
        reliability = "MEDIUM"
        warnings.append("Elevated speaker switching; diarization confidence reduced.")

    return {
        "speaker_count": len(speakers),
        "short_turn_ratio": round(short_ratio, 4),
        "switch_rate_per_min": round(switch_rate, 2),
        "reliability": reliability,
        "warnings": warnings,
    }


def stabilize_unreliable_speakers(utterances: list, report: dict) -> list:
    if report.get("reliability") != "LOW":
        return utterances
    out = []
    for u in utterances:
        item = dict(u)
        item["speaker"] = "SPEAKER_01"
        item["speaker_reliability"] = "collapsed_low_confidence"
        out.append(item)
    return out


# --- Issue 7: Visual ---


def _text_similarity(t1: str, t2: str) -> float:
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return SequenceMatcher(None, t1.lower(), t2.lower()).ratio()


def smart_visual_quality_gate(
    scenes: list,
    min_blur_var: float = 30.0,
    cosine_thresh: float = 0.88,
    ocr_diff_thresh: float = 0.30,
    min_keep_ratio: float = 0.25,
) -> tuple[list, dict]:
    if not scenes:
        return [], {"input": 0, "output": 0}

    processed: list = []
    stats = {
        "input": len(scenes),
        "blur_flagged": 0,
        "blur_removed": 0,
        "dedup_removed": 0,
        "ocr_saved_slides": 0,
    }

    for sc in scenes:
        path = sc.get("keyframe_path")
        blur_var = 100.0
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                blur_var = cv2.Laplacian(img, cv2.CV_64F).var()

        ocr = sc.get("ocr_text", "").strip()
        sc["blur_score"] = round(float(blur_var), 2)
        sc["is_blurry"] = blur_var < min_blur_var
        sc["visual_gate_status"] = "kept"

        if sc["is_blurry"]:
            stats["blur_flagged"] += 1
            if len(ocr) < 8:
                sc["visual_gate_status"] = "dropped_blur_no_evidence"
                stats["blur_removed"] += 1
                continue

        is_duplicate = False
        emb = sc.get("embedding")
        for kept in processed:
            kept_emb = kept.get("embedding")
            kept_ocr = kept.get("ocr_text", "").strip()
            visual_sim = 0.0
            if emb is not None and kept_emb is not None:
                visual_sim = float(
                    np.dot(emb, kept_emb)
                    / (np.linalg.norm(emb) * np.linalg.norm(kept_emb) + 1e-8)
                )
            ocr_sim = _text_similarity(ocr, kept_ocr)
            if visual_sim > cosine_thresh:
                if ocr and kept_ocr and ocr_sim < (1.0 - ocr_diff_thresh):
                    stats["ocr_saved_slides"] += 1
                    continue
                if sc["blur_score"] > kept.get("blur_score", 0):
                    kept["visual_gate_status"] = "replaced_by_sharper_frame"
                    processed.remove(kept)
                    break
                sc["visual_gate_status"] = "dropped_duplicate"
                is_duplicate = True
                stats["dedup_removed"] += 1
                break
        if not is_duplicate:
            processed.append(sc)

    min_keep = max(1, int(len(scenes) * min_keep_ratio))
    if len(processed) < min_keep:
        ranked = sorted(
            scenes,
            key=lambda x: (x.get("blur_score", 0), len((x.get("ocr_text") or "").strip())),
            reverse=True,
        )
        processed = ranked[:min_keep]

    stats["output"] = len(processed)
    stats["min_keep_floor"] = min_keep
    return processed, stats


# --- Issue 8: Caption grounding ---


def verify_and_ground_captions(
    scenes: list,
    min_grounding_score: float = 0.22,
) -> tuple[list, dict]:
    if not scenes:
        return [], {"hallucination_count": 0, "low_grounding_count": 0}

    hallucination_count = 0
    low_grounding_count = 0
    stopwords = {
        "a", "an", "the", "is", "are", "of", "on", "in", "and", "with", "this", "image", "shows", "slide",
    }

    for sc in scenes:
        caption = sc.get("caption", "").strip()
        ocr_text = sc.get("ocr_text", "").strip()
        is_hallucinated = False
        reason = None

        words = caption.lower().split()
        if len(words) > 6 and len(set(words)) / len(words) < 0.4:
            is_hallucinated = True
            reason = "Repetitive word loop in caption."

        ocr_words = set(re.findall(r"\w+", ocr_text.lower()))
        cap_words = set(re.findall(r"\w+", caption.lower()))
        cap_keywords = cap_words - stopwords
        grounded = cap_keywords.intersection(ocr_words)
        grounding_score = len(grounded) / max(1, len(cap_keywords))

        if cap_keywords and grounding_score < min_grounding_score:
            is_hallucinated = True
            reason = reason or "Low OCR grounding score."
            low_grounding_count += 1

        sc["grounding_score"] = round(grounding_score, 2)
        sc["is_hallucinated"] = is_hallucinated
        sc["hallucination_reason"] = reason

        if is_hallucinated:
            hallucination_count += 1
            sc["verified_caption"] = (
                f"Slide Text: {ocr_text[:120]}" if ocr_text else "Lecture Slide"
            )
            sc["grounded_status"] = "fallback_to_ocr"
        else:
            sc["verified_caption"] = caption
            sc["grounded_status"] = "trusted"

        sc["llm_context_block"] = (
            f"[SLIDE @ {sc.get('start_timecode')}]\n"
            f"- OCR: {ocr_text or '[No Text]'}\n"
            f"- Visual: {sc['verified_caption']} (grounding={grounding_score:.2f})"
        )

    return scenes, {
        "hallucination_count": hallucination_count,
        "low_grounding_count": low_grounding_count,
        "avg_grounding_score": round(
            sum(s.get("grounding_score", 0) for s in scenes) / max(1, len(scenes)), 3
        ),
    }


# --- Utterance / chapter post-process ---


def clean_transcript_text(text: str, lang: str = "en") -> str:
    if not text:
        return ""
    fillers_by_lang = {
        "vi": [r"\bà\b", r"\bừm\b", r"\bờ\b"],
        "en": [r"\buhm\b", r"\bumm\b", r"\bhmm\b", r"\byou know\b"],
    }
    fillers = fillers_by_lang.get(lang, fillers_by_lang["en"])
    cleaned = text
    for f in fillers:
        cleaned = re.sub(f, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def post_process_utterances(
    utterances: list,
    min_dur: float = 1.5,
    lang: str = "en",
    min_chars: int = 2,
    visual_dominant: bool = False,
) -> list:
    if not utterances:
        return []

    processed: list = []
    for utt in utterances:
        text = clean_transcript_text(utt.get("text", ""), lang=lang)
        if len(text) < min_chars:
            continue
        if visual_dominant and len(text) < 8:
            continue

        dur = float(utt.get("end", 0.0)) - float(utt.get("start", 0.0))
        if processed and dur < min_dur and processed[-1].get("speaker") == utt.get("speaker"):
            processed[-1]["end"] = utt["end"]
            processed[-1]["text"] = (processed[-1].get("text", "") + " " + text).strip()
        else:
            item = dict(utt)
            item["text"] = text
            processed.append(item)

    return processed if processed else utterances


def prepare_slides_for_summarizer(slides: list) -> list:
    out = []
    for s in slides:
        item = dict(s)
        if item.get("verified_caption"):
            item["caption"] = item["verified_caption"]
        out.append(item)
    return out


def _chapter_bounds(ch: dict) -> tuple[float, float]:
    start = float(ch.get("startTime", ch.get("start_time", ch.get("start_seconds", 0.0))) or 0.0)
    end = float(ch.get("endTime", ch.get("end_time", ch.get("end_seconds", start))) or start)
    if end <= start:
        end = start + 10.0
    return start, end


def post_process_chapters(chapters: list, min_dur_sec: float = 45.0) -> list:
    if not chapters:
        return []

    smoothed: list = []
    for idx, ch in enumerate(chapters, 1):
        item = dict(ch)
        s_sec, e_sec = _chapter_bounds(item)
        item["title"] = item.get("title") or f"Chapter {idx}"
        item["summary"] = item.get("summary") or ""
        for k in ("startTime", "start_time", "start_seconds", "endTime", "end_time", "end_seconds"):
            pass
        item["startTime"] = s_sec
        item["endTime"] = e_sec
        item["start_seconds"] = s_sec
        item["end_seconds"] = e_sec

        dur = e_sec - s_sec
        if smoothed and dur < min_dur_sec:
            smoothed[-1]["endTime"] = e_sec
            smoothed[-1]["end_seconds"] = e_sec
            if item.get("summary"):
                smoothed[-1]["summary"] = (
                    smoothed[-1].get("summary", "") + " " + item["summary"]
                ).strip()
        else:
            smoothed.append(item)
    return smoothed

"""Benchmark metrics for lecture pipeline evaluation (Bang 1–12).

Pure functions: no model I/O. Runners call these after collecting predictions + GT.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Dataclasses (kept for compatibility with older imports)
# ---------------------------------------------------------------------------


@dataclass
class ASRMetrics:
    wer: float | None = None
    cer: float | None = None
    rtf: float | None = None
    timestamp_precision: float | None = None
    timestamp_recall: float | None = None
    hallucination_rate: float | None = None


@dataclass
class DiarizationMetrics:
    der: float | None = None
    speaker_confusion: float | None = None
    missed_speech: float | None = None
    false_alarm: float | None = None


@dataclass
class VisualMetrics:
    scene_precision: float | None = None
    scene_recall: float | None = None
    scene_f1: float | None = None
    keyframe_precision: float | None = None
    keyframe_recall: float | None = None
    keyframe_f1: float | None = None
    frame_compression_ratio: float | None = None


@dataclass
class SemanticMetrics:
    ocr_cer: float | None = None
    ocr_char_accuracy: float | None = None
    ocr_word_accuracy: float | None = None
    caption_accuracy: float | None = None
    hallucination_rate: float | None = None
    clip_alignment_score: float | None = None


@dataclass
class SummaryMetrics:
    rouge_l: float | None = None
    bertscore_f1: float | None = None
    factuality_rate: float | None = None
    coverage: float | None = None
    coherence: float | None = None
    human_rating_1_5: float | None = None


@dataclass
class TimelineMetrics:
    alignment_accuracy: float | None = None
    slide_sync_mae_sec: float | None = None
    chapter_boundary_f1: float | None = None
    rag_hit_at_3: float | None = None
    rag_hit_at_5: float | None = None


@dataclass
class SystemMetrics:
    total_latency_sec: float | None = None
    gpu_vram_peak_gb: float | None = None
    api_cost_usd: float | None = None
    cost_per_hour_video_usd: float | None = None


@dataclass
class BenchmarkResult:
    lecture_id: str
    stack: str
    duration_min: float
    asr: ASRMetrics = field(default_factory=ASRMetrics)
    diarization: DiarizationMetrics = field(default_factory=DiarizationMetrics)
    visual: VisualMetrics = field(default_factory=VisualMetrics)
    semantic: SemanticMetrics = field(default_factory=SemanticMetrics)
    summary: SummaryMetrics = field(default_factory=SummaryMetrics)
    timeline: TimelineMetrics = field(default_factory=TimelineMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


def normalize_text(text: str, *, lowercase: bool = True) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    if lowercase:
        t = t.lower()
    return t


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize_text(text))


# ---------------------------------------------------------------------------
# ASR / OCR string metrics
# ---------------------------------------------------------------------------


def _levenshtein(a: Sequence[str] | str, b: Sequence[str] | str) -> int:
    """Classic Levenshtein distance on sequences (chars or tokens)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate in [0, +inf). Prefer jiwer when installed."""
    try:
        import jiwer

        ref = normalize_text(reference)
        hyp = normalize_text(hypothesis)
        if not ref:
            return 0.0 if not hyp else 1.0
        return float(jiwer.wer(ref, hyp))
    except Exception:
        ref_toks = tokenize_words(reference)
        hyp_toks = tokenize_words(hypothesis)
        if not ref_toks:
            return 0.0 if not hyp_toks else 1.0
        return _levenshtein(ref_toks, hyp_toks) / len(ref_toks)


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate in [0, +inf). Prefer jiwer when installed."""
    try:
        import jiwer

        ref = normalize_text(reference).replace(" ", "")
        hyp = normalize_text(hypothesis).replace(" ", "")
        if not ref:
            return 0.0 if not hyp else 1.0
        return float(jiwer.cer(ref, hyp))
    except Exception:
        ref = list(normalize_text(reference).replace(" ", ""))
        hyp = list(normalize_text(hypothesis).replace(" ", ""))
        if not ref:
            return 0.0 if not hyp else 1.0
        return _levenshtein(ref, hyp) / len(ref)


def asr_wer_cer(reference: str, hypothesis: str) -> dict[str, float]:
    return {
        "wer": wer(reference, hypothesis),
        "cer": cer(reference, hypothesis),
        "wer_pct": wer(reference, hypothesis) * 100.0,
        "cer_pct": cer(reference, hypothesis) * 100.0,
    }


def rtf(audio_duration_sec: float, wall_time_sec: float) -> float:
    if audio_duration_sec <= 0:
        return float("inf")
    return wall_time_sec / audio_duration_sec


def word_accuracy(reference: str, hypothesis: str) -> float:
    """1 - WER clipped to [0, 1]."""
    return max(0.0, 1.0 - wer(reference, hypothesis))


# ---------------------------------------------------------------------------
# Interval / boundary metrics (VAD, scene, chapter)
# ---------------------------------------------------------------------------


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def interval_prf(
    pred: Sequence[tuple[float, float]],
    ref: Sequence[tuple[float, float]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Precision/Recall/F1 for time intervals via IoU matching (greedy)."""
    preds = [(float(s), float(e)) for s, e in pred if e > s]
    refs = [(float(s), float(e)) for s, e in ref if e > s]
    if not preds and not refs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(refs)}
    if not refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(preds), "fn": 0}

    pairs: list[tuple[float, int, int]] = []
    for i, (ps, pe) in enumerate(preds):
        for j, (rs, re_) in enumerate(refs):
            inter = _overlap(ps, pe, rs, re_)
            union = (pe - ps) + (re_ - rs) - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    used_p: set[int] = set()
    used_r: set[int] = set()
    tp = 0
    for _, i, j in pairs:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        tp += 1

    fp = len(preds) - tp
    fn = len(refs) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def false_cut_rate(
    pred_speech: Sequence[tuple[float, float]],
    ref_speech: Sequence[tuple[float, float]],
) -> float:
    """Fraction of reference speech duration that was NOT covered by predicted speech."""
    total = sum(max(0.0, e - s) for s, e in ref_speech)
    if total <= 0:
        return 0.0
    missed = 0.0
    for rs, re_ in ref_speech:
        covered = 0.0
        for ps, pe in pred_speech:
            covered += _overlap(rs, re_, ps, pe)
        missed += max(0.0, (re_ - rs) - covered)
    return missed / total


def boundary_prf(
    pred_boundaries: Sequence[float],
    ref_boundaries: Sequence[float],
    *,
    tolerance_sec: float = 15.0,
) -> dict[str, float]:
    """Boundary Precision/Recall/F1 with ±tolerance matching."""
    preds = sorted(float(x) for x in pred_boundaries)
    refs = sorted(float(x) for x in ref_boundaries)
    if not preds and not refs:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mae": 0.0}
    if not preds:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": float("nan")}
    if not refs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": float("nan")}

    used_r: set[int] = set()
    tp = 0
    abs_errs: list[float] = []
    for p in preds:
        best_j = None
        best_d = None
        for j, r in enumerate(refs):
            if j in used_r:
                continue
            d = abs(p - r)
            if d <= tolerance_sec and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is not None and best_d is not None:
            used_r.add(best_j)
            tp += 1
            abs_errs.append(best_d)

    fp = len(preds) - tp
    fn = len(refs) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mae = sum(abs_errs) / len(abs_errs) if abs_errs else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "mae": mae, "tp": float(tp)}


def timestamp_mae(pred: Sequence[float], ref: Sequence[float]) -> float:
    """Mean |pred_i - ref_i| after sorting and truncating to common length."""
    p = sorted(float(x) for x in pred)
    r = sorted(float(x) for x in ref)
    n = min(len(p), len(r))
    if n == 0:
        return float("nan")
    return sum(abs(p[i] - r[i]) for i in range(n)) / n


# ---------------------------------------------------------------------------
# Caption / human rubric helpers
# ---------------------------------------------------------------------------


def caption_hallucination_flags(
    caption: str,
    ocr_text: str = "",
    *,
    min_grounding: float = 0.15,
) -> dict[str, Any]:
    """Heuristic hallucination detector used when human labels are absent."""
    cap = normalize_text(caption)
    ocr = normalize_text(ocr_text)
    generic = bool(
        re.search(r"keyframe for scene|an image of|a picture of|photo of a", cap)
    )
    cap_toks = set(tokenize_words(cap))
    ocr_toks = set(tokenize_words(ocr))
    if not cap_toks:
        grounding = 0.0
    elif not ocr_toks:
        grounding = 0.5 if not generic else 0.0
    else:
        grounding = len(cap_toks & ocr_toks) / max(1, len(cap_toks))
    hallucinated = bool(generic or (bool(ocr_toks) and grounding < min_grounding))
    return {
        "generic": generic,
        "grounding_score": grounding,
        "hallucinated": hallucinated,
        "content_ok": not hallucinated,
    }


def aggregate_caption_scores(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {"accuracy": float("nan"), "hallucination_rate": float("nan"), "n": 0}
    n = len(rows)
    ok = sum(1 for r in rows if r.get("content_ok") or r.get("dung_noi_dung"))
    hall = sum(1 for r in rows if r.get("hallucinated") or r.get("hallucination"))
    return {
        "accuracy": ok / n,
        "hallucination_rate": hall / n,
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(scorer.score(reference or "", hypothesis or "")["rougeL"].fmeasure)
    except Exception:
        # LCS-based fallback
        ref = tokenize_words(reference)
        hyp = tokenize_words(hypothesis)
        if not ref or not hyp:
            return 0.0
        # DP LCS length
        dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[-1][-1]
        prec = lcs / len(hyp)
        rec = lcs / len(ref)
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def bertscore_f1(reference: str, hypothesis: str, *, lang: str = "en") -> float | None:
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score([hypothesis or ""], [reference or ""], lang=lang, verbose=False)
        return float(f1.mean().item())
    except Exception:
        return None


def factuality_coverage(
    source_text: str,
    hypothesis: str,
    reference_summary: str,
) -> dict[str, float]:
    """Proxy metrics when human factuality labels are unavailable.

    - factuality: fraction of hypothesis words grounded in source (transcript/OCR).
    - coverage: fraction of reference-summary words echoed in hypothesis.
    """
    hyp_toks = set(tokenize_words(hypothesis))
    src_toks = set(tokenize_words(source_text))
    ref_toks = set(tokenize_words(reference_summary))
    factuality = len(hyp_toks & src_toks) / len(hyp_toks) if hyp_toks else 0.0
    coverage = len(hyp_toks & ref_toks) / len(ref_toks) if ref_toks else 0.0
    return {"factuality": factuality, "coverage": coverage}


def human_score_from_caption(flags: dict[str, Any]) -> float:
    """Map heuristic caption flags to a 1–5 rubric (automated proxy)."""
    if flags.get("hallucinated"):
        return 1.0
    grounding = float(flags.get("grounding_score") or 0.0)
    if grounding >= 0.6:
        return 5.0
    if grounding >= 0.4:
        return 4.0
    if grounding >= 0.25:
        return 3.0
    if grounding >= 0.15:
        return 2.0
    return 1.5


def summary_text_metrics(
    reference: str,
    hypothesis: str,
    *,
    source_text: str = "",
    lang: str = "en",
    compute_bertscore: bool = True,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "rouge_l": rouge_l_f1(reference, hypothesis),
        "bertscore_f1": None,
        "factuality": None,
        "coverage": None,
    }
    if compute_bertscore:
        out["bertscore_f1"] = bertscore_f1(reference, hypothesis, lang=lang)
    if source_text.strip():
        fc = factuality_coverage(source_text, hypothesis, reference)
        out["factuality"] = fc["factuality"]
        out["coverage"] = fc["coverage"]
    return out


# ---------------------------------------------------------------------------
# RAG metrics
# ---------------------------------------------------------------------------


def rag_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    gold_chunk_ids: Sequence[str],
    *,
    k: int = 5,
) -> float:
    top = list(retrieved_chunk_ids)[:k]
    gold = set(gold_chunk_ids)
    if not gold:
        return float("nan")
    return 1.0 if any(c in gold for c in top) else 0.0


def citation_accuracy(
    predicted_ts: float | None,
    gold_ts: float | None,
    *,
    tolerance_sec: float = 15.0,
) -> float:
    if predicted_ts is None or gold_ts is None:
        return float("nan")
    return 1.0 if abs(float(predicted_ts) - float(gold_ts)) <= tolerance_sec else 0.0


def mean_ignore_nan(values: Iterable[float | None]) -> float:
    nums = [float(v) for v in values if v is not None and v == v]  # filter NaN
    if not nums:
        return float("nan")
    return sum(nums) / len(nums)

"""Temporal Chaptering and Semantic Segmentation Metrics.

Implements standard benchmark metrics for video chaptering:
- Collar F1 (with configurable tolerance window: ±3s, ±5s, ±10s)
- P_k metric (Beeferman et al., 1999)
- WindowDiff metric (Pevzner & Hearst, 2002)

Compatible with chunkseg, YTSeg, and standard NLP boundary evaluation protocols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CollarMetrics:
    precision: float
    recall: float
    f1: float
    tolerance_sec: float
    true_positives: int
    num_pred: int
    num_gold: int


def collar_f1(
    gold_boundaries_sec: Sequence[float],
    pred_boundaries_sec: Sequence[float],
    *,
    tolerance_sec: float = 3.0,
) -> CollarMetrics:
    """Calculate 1-to-1 greedy matched boundary Precision, Recall, and F1 within a collar window.

    Args:
        gold_boundaries_sec: List of reference boundary timestamps in seconds (sorted or unsorted).
        pred_boundaries_sec: List of predicted boundary timestamps in seconds.
        tolerance_sec: Maximum allowable absolute difference |pred - gold| in seconds (default 3.0s).

    Returns:
        CollarMetrics object with precision, recall, f1, and counts.
    """
    golds = sorted([float(t) for t in gold_boundaries_sec if t is not None])
    preds = sorted([float(t) for t in pred_boundaries_sec if t is not None])

    n_gold = len(golds)
    n_pred = len(preds)

    if n_gold == 0 and n_pred == 0:
        return CollarMetrics(
            precision=1.0,
            recall=1.0,
            f1=1.0,
            tolerance_sec=tolerance_sec,
            true_positives=0,
            num_pred=0,
            num_gold=0,
        )

    if n_gold == 0 or n_pred == 0:
        return CollarMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            tolerance_sec=tolerance_sec,
            true_positives=0,
            num_pred=n_pred,
            num_gold=n_gold,
        )

    # 1-to-1 matching: match each prediction to the nearest available gold boundary within tolerance
    matched_gold_indices = set()
    tp = 0

    for p in preds:
        best_idx = None
        best_dist = float("inf")
        for idx, g in enumerate(golds):
            if idx in matched_gold_indices:
                continue
            dist = abs(p - g)
            if dist <= tolerance_sec and dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_idx is not None:
            matched_gold_indices.add(best_idx)
            tp += 1

    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_gold if n_gold > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return CollarMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tolerance_sec=tolerance_sec,
        true_positives=tp,
        num_pred=n_pred,
        num_gold=n_gold,
    )


def compute_multi_collar_f1(
    gold_boundaries_sec: Sequence[float],
    pred_boundaries_sec: Sequence[float],
    tolerances: Sequence[float] = (3.0, 5.0, 10.0),
) -> dict[str, float]:
    """Compute Collar F1 scores across multiple tolerance windows."""
    results = {}
    for tol in tolerances:
        metric = collar_f1(gold_boundaries_sec, pred_boundaries_sec, tolerance_sec=tol)
        key_prefix = f"collar_{int(tol) if tol.is_integer() else tol}s"
        results[f"{key_prefix}_f1"] = metric.f1
        results[f"{key_prefix}_precision"] = metric.precision
        results[f"{key_prefix}_recall"] = metric.recall
    return results


def boundaries_to_binary_sequence(
    boundaries_sec: Sequence[float],
    total_duration_sec: float,
    *,
    step_sec: float = 1.0,
) -> list[int]:
    """Convert continuous timestamp boundaries into a discrete binary boundary sequence."""
    n_steps = max(1, int(math.ceil(total_duration_sec / step_sec)))
    seq = [0] * n_steps
    for b in boundaries_sec:
        idx = int(round(b / step_sec))
        if 0 <= idx < n_steps:
            seq[idx] = 1
    return seq


def pk_metric(
    ref_binary: Sequence[int],
    hyp_binary: Sequence[int],
    *,
    k: int | None = None,
) -> float:
    """Calculate Beeferman's P_k penalty metric (lower is better, 0.0 is perfect).

    Args:
        ref_binary: Discrete binary sequence of reference boundaries (1 at boundary, 0 otherwise).
        hyp_binary: Discrete binary sequence of predicted boundaries.
        k: Window size. Defaults to half the average reference segment size.
    """
    n = len(ref_binary)
    if n != len(hyp_binary) or n == 0:
        raise ValueError("ref_binary and hyp_binary must have equal non-zero length")

    num_ref_boundaries = sum(ref_binary)
    if k is None:
        avg_segment_len = n / (num_ref_boundaries + 1)
        k = max(1, int(round(avg_segment_len / 2.0)))

    if k >= n:
        return 0.0

    # Precompute cumulative sums for fast segment check
    ref_cumsum = [0] * (n + 1)
    hyp_cumsum = [0] * (n + 1)
    for i in range(n):
        ref_cumsum[i + 1] = ref_cumsum[i] + ref_binary[i]
        hyp_cumsum[i + 1] = hyp_cumsum[i] + hyp_binary[i]

    penalties = 0
    total_windows = n - k
    for i in range(total_windows):
        # same segment in ref if sum between i and i+k is 0
        ref_same = (ref_cumsum[i + k] - ref_cumsum[i]) == 0
        hyp_same = (hyp_cumsum[i + k] - hyp_cumsum[i]) == 0
        if ref_same != hyp_same:
            penalties += 1

    return penalties / total_windows if total_windows > 0 else 0.0


def window_diff(
    ref_binary: Sequence[int],
    hyp_binary: Sequence[int],
    *,
    k: int | None = None,
) -> float:
    """Calculate Pevzner & Hearst's WindowDiff metric (lower is better, 0.0 is perfect).

    Args:
        ref_binary: Discrete binary sequence of reference boundaries (1 at boundary, 0 otherwise).
        hyp_binary: Discrete binary sequence of predicted boundaries.
        k: Window size. Defaults to half the average reference segment size.
    """
    n = len(ref_binary)
    if n != len(hyp_binary) or n == 0:
        raise ValueError("ref_binary and hyp_binary must have equal non-zero length")

    num_ref_boundaries = sum(ref_binary)
    if k is None:
        avg_segment_len = n / (num_ref_boundaries + 1)
        k = max(1, int(round(avg_segment_len / 2.0)))

    if k >= n:
        return 0.0

    ref_cumsum = [0] * (n + 1)
    hyp_cumsum = [0] * (n + 1)
    for i in range(n):
        ref_cumsum[i + 1] = ref_cumsum[i] + ref_binary[i]
        hyp_cumsum[i + 1] = hyp_cumsum[i] + hyp_binary[i]

    penalties = 0
    total_windows = n - k
    for i in range(total_windows):
        ref_count = ref_cumsum[i + k] - ref_cumsum[i]
        hyp_count = hyp_cumsum[i + k] - hyp_cumsum[i]
        if ref_count != hyp_count:
            penalties += 1

    return penalties / total_windows if total_windows > 0 else 0.0


def compute_all_chapter_metrics(
    gold_boundaries_sec: Sequence[float],
    pred_boundaries_sec: Sequence[float],
    video_duration_sec: float = 1800.0,
) -> dict[str, float]:
    """Calculate all chaptering benchmark metrics in one call:
    - Collar F1 @ 3s, 5s, 10s
    - Beeferman's P_k
    - Pevzner & Hearst's WindowDiff
    """
    multi_f1 = compute_multi_collar_f1(gold_boundaries_sec, pred_boundaries_sec, tolerances=(3.0, 5.0, 10.0))
    
    ref_bin = boundaries_to_binary_sequence(gold_boundaries_sec, video_duration_sec, step_sec=1.0)
    hyp_bin = boundaries_to_binary_sequence(pred_boundaries_sec, video_duration_sec, step_sec=1.0)
    
    pk = pk_metric(ref_bin, hyp_bin)
    wd = window_diff(ref_bin, hyp_bin)
    
    return {
        "collar_f1_3s": multi_f1["collar_3s_f1"],
        "collar_f1_5s": multi_f1["collar_5s_f1"],
        "collar_f1_10s": multi_f1["collar_10s_f1"],
        "pk": pk,
        "window_diff": wd,
    }

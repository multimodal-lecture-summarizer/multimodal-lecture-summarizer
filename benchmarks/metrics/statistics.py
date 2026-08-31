"""Statistical Significance and Multiple-Comparison Correction Engine.

Implements rigorous video-level statistics per decisions-log.md D-T07 & 03-colab-runbook.md §12:
- Paired Video-level Bootstrap 95% Confidence Intervals (1,000 resamples).
- Family-wise Holm-Bonferroni multi-comparison correction:
    * RQ1 ablations (4 deltas: C2−C1, C3−C1, C4−C1, C5−C6)
    * RQ2 S-pairs (4: S1−S0, S3−S1, S4−S3, S2−S1)
    * RQ3 Q-pairs (3: Q1−Q0, Q2−Q0, Q3−Q2)
- Effect Size: Cohen's d with small-sample Hedges' g correction (when n < 20).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class BootstrapCI:
    mean: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_resamples: int = 1000


@dataclass(frozen=True)
class ComparisonResult:
    label: str
    n_samples: int
    mean_diff: float
    raw_p_value: float
    corrected_p_value: float
    reject_null: bool
    cohens_d: float
    hedges_g: float
    ci_95: Tuple[float, float]


def calculate_cohens_d(x: Sequence[float], y: Optional[Sequence[float]] = None) -> float:
    """Calculate Cohen's d for paired differences (1D) or two independent samples (2D).

    Args:
        x: If y is None, x represents paired differences (x_i = a_i - b_i).
           If y is provided, x is sample 1 and y is sample 2.
    """
    arr_x = np.asarray(x, dtype=np.float64)
    if y is None:
        # Paired difference
        mean_d = float(np.mean(arr_x))
        std_d = float(np.std(arr_x, ddof=1)) if len(arr_x) > 1 else 0.0
        return mean_d / std_d if std_d > 1e-12 else 0.0
    
    arr_y = np.asarray(y, dtype=np.float64)
    n1, n2 = len(arr_x), len(arr_y)
    mean1, mean2 = float(np.mean(arr_x)), float(np.mean(arr_y))
    var1 = float(np.var(arr_x, ddof=1)) if n1 > 1 else 0.0
    var2 = float(np.var(arr_y, ddof=1)) if n2 > 1 else 0.0
    
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1))
    return (mean1 - mean2) / pooled_sd if pooled_sd > 1e-12 else 0.0


def calculate_hedges_g(d: float, n: int) -> float:
    """Apply Hedges' g correction factor for small sample bias (n < 20)."""
    if n <= 2:
        return d
    correction = 1.0 - (3.0 / (4.0 * n - 5.0))
    return float(d * correction)


def paired_bootstrap_ci(
    differences: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Compute empirical percentile bootstrap confidence interval for paired differences.

    Args:
        differences: Per-video metric differences (e.g. F1_modelA - F1_modelB).
        n_resamples: Number of bootstrap resamples (default 1000).
        confidence_level: Desired confidence level (default 0.95).
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapCI dataclass with mean and [lower, upper] bounds.
    """
    arr = np.asarray(differences, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0, confidence_level=confidence_level, n_resamples=0)
    
    if n == 1:
        val = float(arr[0])
        return BootstrapCI(mean=val, ci_lower=val, ci_upper=val, confidence_level=confidence_level, n_resamples=1)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(arr[boot_indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(arr))

    return BootstrapCI(
        mean=mean_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )


def holm_bonferroni_family(
    deltas_dict: Dict[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, ComparisonResult]:
    """Execute family-wise Holm-Bonferroni multiple testing procedure per decisions-log.md D-T07.

    Args:
        deltas_dict: Mapping from comparison label to 1D array of per-video paired differences.
                     Example: {"C5 - C1": diff_c5_c1, "C2 - C1": diff_c2_c1, ...}
        alpha: Overall family-wise error rate threshold (default 0.05).
        n_resamples: Number of resamples for bootstrap CIs.
        seed: Random seed for bootstrapping.

    Returns:
        Dictionary mapping each comparison label to a ComparisonResult.
    """
    labels = list(deltas_dict.keys())
    k = len(labels)
    if k == 0:
        return {}

    raw_stats: List[Dict[str, Any]] = []

    for label in labels:
        diffs = np.asarray(deltas_dict[label], dtype=np.float64)
        n = len(diffs)
        mean_d = float(np.mean(diffs)) if n > 0 else 0.0

        if n > 1 and float(np.std(diffs, ddof=1)) > 1e-12:
            t_res = stats.ttest_1samp(diffs, 0.0)
            p_val = float(t_res.pvalue)
        else:
            p_val = 1.0

        d_val = calculate_cohens_d(diffs)
        g_val = calculate_hedges_g(d_val, n)
        boot_ci = paired_bootstrap_ci(diffs, n_resamples=n_resamples, seed=seed)

        raw_stats.append({
            "label": label,
            "n": n,
            "mean_diff": mean_d,
            "raw_p": p_val,
            "d": d_val,
            "g": g_val,
            "ci": (boot_ci.ci_lower, boot_ci.ci_upper),
        })

    # Sort comparisons by raw p-value ascending for Holm step-down
    sorted_indices = sorted(range(k), key=lambda idx: raw_stats[idx]["raw_p"])
    
    # Step-down adjusted p-values
    # p_adj_i = min(1.0, max(p_adj_{i-1}, (k - i) * p_i))
    corrected_p_vals = [0.0] * k
    running_max_p = 0.0

    for step, rank_idx in enumerate(sorted_indices):
        m_step = k - step
        p_raw = raw_stats[rank_idx]["raw_p"]
        p_adjusted = min(1.0, p_raw * m_step)
        running_max_p = max(running_max_p, p_adjusted)
        corrected_p_vals[rank_idx] = running_max_p

    results: Dict[str, ComparisonResult] = {}
    for i, s in enumerate(raw_stats):
        lbl = s["label"]
        p_corr = corrected_p_vals[i]
        results[lbl] = ComparisonResult(
            label=lbl,
            n_samples=s["n"],
            mean_diff=s["mean_diff"],
            raw_p_value=s["raw_p"],
            corrected_p_value=p_corr,
            reject_null=bool(p_corr < alpha and s["mean_diff"] > 0),
            cohens_d=s["d"],
            hedges_g=s["g"],
            ci_95=s["ci"],
        )

    return results

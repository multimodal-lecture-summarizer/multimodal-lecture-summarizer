"""Benchmark evaluation metrics and statistical analysis tools."""

from benchmarks.metrics.chapter_metrics import (
    CollarMetrics,
    boundaries_to_binary_sequence,
    collar_f1,
    compute_multi_collar_f1,
    pk_metric,
    window_diff,
)
from benchmarks.metrics.statistics import (
    BootstrapCI,
    ComparisonResult,
    calculate_cohens_d,
    calculate_hedges_g,
    holm_bonferroni_family,
    paired_bootstrap_ci,
)

__all__ = [
    "CollarMetrics",
    "collar_f1",
    "compute_multi_collar_f1",
    "boundaries_to_binary_sequence",
    "pk_metric",
    "window_diff",
    "BootstrapCI",
    "ComparisonResult",
    "calculate_cohens_d",
    "calculate_hedges_g",
    "paired_bootstrap_ci",
    "holm_bonferroni_family",
]

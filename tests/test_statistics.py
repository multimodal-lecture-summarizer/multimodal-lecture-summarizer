"""Unit tests for statistical significance, bootstrap CIs, and Holm-Bonferroni corrections."""

import numpy as np
import pytest
from benchmarks.metrics.statistics import (
    calculate_cohens_d,
    calculate_hedges_g,
    holm_bonferroni_family,
    paired_bootstrap_ci,
)


def test_cohens_d_zero_variance():
    diffs = [0.0, 0.0, 0.0]
    assert calculate_cohens_d(diffs) == 0.0


def test_cohens_d_positive():
    diffs = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = calculate_cohens_d(diffs)
    assert d > 0.0
    assert pytest.approx(d, rel=1e-2) == (3.0 / np.std(diffs, ddof=1))


def test_hedges_g_small_sample():
    d = 1.0
    n = 10
    g = calculate_hedges_g(d, n)
    assert g < d  # Hedges' g reduces overestimate for small sample
    assert calculate_hedges_g(d, 2) == d


def test_paired_bootstrap_ci_coverage():
    diffs = [0.05, 0.08, 0.12, 0.07, 0.10, 0.09, 0.06, 0.11]
    res = paired_bootstrap_ci(diffs, n_resamples=1000, confidence_level=0.95, seed=42)
    assert res.ci_lower < res.mean < res.ci_upper
    assert res.mean == pytest.approx(np.mean(diffs))
    assert res.ci_lower > 0.0  # Entire CI is strictly positive


def test_holm_bonferroni_family_step_down():
    # RQ1 family: 4 deltas
    np.random.seed(42)
    deltas = {
        "C5 - C1": np.random.normal(loc=0.15, scale=0.02, size=30),  # Strong positive
        "C2 - C1": np.random.normal(loc=0.04, scale=0.03, size=30),  # Moderate positive
        "C3 - C1": np.random.normal(loc=0.01, scale=0.05, size=30),  # Weak / null
        "C5 - C6": np.random.normal(loc=0.10, scale=0.02, size=30),  # Strong positive
    }
    
    results = holm_bonferroni_family(deltas, alpha=0.05)
    assert len(results) == 4
    
    # Check that corrected p >= raw p
    for label, comp in results.items():
        assert comp.corrected_p_value >= comp.raw_p_value
        assert comp.cohens_d != 0.0
        assert comp.ci_95[0] <= comp.mean_diff <= comp.ci_95[1]

    # Strong deltas should reject null
    assert results["C5 - C1"].reject_null is True
    assert results["C5 - C6"].reject_null is True

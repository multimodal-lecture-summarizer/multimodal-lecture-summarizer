"""Unit tests for chapter segmentation metrics (Collar F1, Pk, WindowDiff)."""

import pytest
from benchmarks.metrics.chapter_metrics import (
    collar_f1,
    compute_multi_collar_f1,
    boundaries_to_binary_sequence,
    pk_metric,
    window_diff,
)


def test_collar_f1_perfect_match():
    gold = [10.0, 25.0, 60.0]
    pred = [10.0, 25.0, 60.0]
    metric = collar_f1(gold, pred, tolerance_sec=3.0)
    assert metric.f1 == 1.0
    assert metric.precision == 1.0
    assert metric.recall == 1.0
    assert metric.true_positives == 3


def test_collar_f1_within_tolerance():
    gold = [10.0, 25.0, 60.0]
    pred = [12.0, 23.5, 62.9]  # all within ±3s
    metric = collar_f1(gold, pred, tolerance_sec=3.0)
    assert metric.f1 == 1.0
    assert metric.true_positives == 3


def test_collar_f1_outside_tolerance():
    gold = [10.0, 25.0, 60.0]
    pred = [15.0, 25.0, 65.0]  # only 25.0 matches within ±3s
    metric = collar_f1(gold, pred, tolerance_sec=3.0)
    assert metric.true_positives == 1
    assert metric.precision == pytest.approx(1 / 3)
    assert metric.recall == pytest.approx(1 / 3)
    assert metric.f1 == pytest.approx(1 / 3)


def test_collar_f1_empty():
    assert collar_f1([], []).f1 == 1.0
    assert collar_f1([10.0], []).f1 == 0.0
    assert collar_f1([], [10.0]).f1 == 0.0


def test_compute_multi_collar_f1():
    gold = [10.0, 30.0]
    pred = [14.0, 30.0]  # 14.0 is outside 3s, but inside 5s and 10s
    res = compute_multi_collar_f1(gold, pred, tolerances=[3.0, 5.0, 10.0])
    assert res["collar_3s_f1"] == pytest.approx(0.5)  # 1 out of 2 matches
    assert res["collar_5s_f1"] == 1.0                 # both match
    assert res["collar_10s_f1"] == 1.0


def test_boundaries_to_binary_sequence():
    b = [2.0, 5.0]
    seq = boundaries_to_binary_sequence(b, total_duration_sec=6.0, step_sec=1.0)
    assert seq[2] == 1
    assert seq[5] == 1
    assert sum(seq) == 2


def test_pk_and_window_diff_perfect():
    ref = [0, 0, 1, 0, 0, 1, 0, 0]
    hyp = [0, 0, 1, 0, 0, 1, 0, 0]
    assert pk_metric(ref, hyp, k=2) == 0.0
    assert window_diff(ref, hyp, k=2) == 0.0


def test_pk_and_window_diff_imperfect():
    ref = [0, 0, 1, 0, 0, 1, 0, 0]
    hyp = [0, 0, 0, 0, 0, 0, 0, 0]  # Missed all boundaries
    assert pk_metric(ref, hyp, k=2) > 0.0
    assert window_diff(ref, hyp, k=2) > 0.0

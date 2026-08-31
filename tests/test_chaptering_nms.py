"""
Unit test to verify 1D Non-Maximum Suppression (Peak Finding) in Chaptering models.
Ensures that boundary clustering artifacts (e.g. 744s, 806s, 868s) are eliminated.
"""

import pytest
import torch
import numpy as np
from benchmarks.models.chaptering import C5_TemporalCrossAttentionTransformer, BaseChapteringModel


def test_nms_eliminates_clusters():
    base = BaseChapteringModel()
    
    # 100 timesteps, 1 per second (0 to 99s)
    T = 100
    timestamps = torch.arange(0, T, dtype=torch.float32).unsqueeze(0) # [1, 100]
    
    # Create a probability sequence with a broad plateau and two sharp peaks
    # Plateau around t=20..30 (all > 0.6), but peak at t=25
    # Sharp peak at t=70 (p=0.95)
    probs = torch.zeros(1, T, dtype=torch.float32)
    probs[0, 20:30] = 0.65
    probs[0, 25] = 0.85 # True local maximum
    
    probs[0, 68:73] = 0.55
    probs[0, 70] = 0.95 # True local maximum
    
    # Run upgraded extract_boundaries with min_interval_sec = 15.0s
    boundaries = base.extract_boundaries(
        probabilities=probs,
        timestamps=timestamps,
        threshold=0.5,
        min_interval_sec=15.0,
        smooth_sigma=1.0
    )[0]
    
    # Should only find 2 clean boundaries (around 25s and 70s), NOT a cluster of points from the plateau
    assert len(boundaries) == 2, f"Expected 2 boundaries, got {len(boundaries)}: {boundaries}"
    assert abs(boundaries[0] - 25.0) <= 2.0
    assert abs(boundaries[1] - 70.0) <= 2.0


def test_nms_suppresses_close_peaks():
    base = BaseChapteringModel()
    T = 60
    timestamps = torch.arange(0, T, dtype=torch.float32).unsqueeze(0)
    
    # Two close peaks: t=10 (p=0.7) and t=15 (p=0.9)
    probs = torch.zeros(1, T, dtype=torch.float32)
    probs[0, 10] = 0.70
    probs[0, 15] = 0.90
    
    # With min_interval_sec = 20.0s, the higher peak at 15s should suppress the peak at 10s
    boundaries = base.extract_boundaries(
        probabilities=probs,
        timestamps=timestamps,
        threshold=0.5,
        min_interval_sec=20.0,
        smooth_sigma=0.0 # disable smoothing to test pure peak suppression
    )[0]
    
    assert len(boundaries) == 1
    assert boundaries[0] == 15.0

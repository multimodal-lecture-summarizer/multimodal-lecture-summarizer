"""Unit tests for evaluation metrics used by thesis Bang 1–12."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.evaluation.datasets import tvsum_important_windows, tvsum_scene_boundaries
from experiments.evaluation.metrics import (
    asr_wer_cer,
    boundary_prf,
    caption_hallucination_flags,
    false_cut_rate,
    interval_prf,
    rag_hit_at_k,
    rouge_l_f1,
    rtf,
    wer,
)
from experiments.evaluation.runners import (
    _extractive_summary,
    _reference_summary_from_utterances,
    eval_chapters,
    eval_keyframe_filter,
    eval_ocr_items,
)
from experiments.evaluation.schemas import normalize_boundaries, normalize_intervals


def test_wer_identical():
    assert wer("hello world", "Hello World") == 0.0


def test_asr_wer_cer_dict():
    scores = asr_wer_cer("the cat sat", "the cat sat")
    assert scores["wer"] == 0.0
    assert scores["cer"] == 0.0


def test_rtf():
    assert abs(rtf(10.0, 2.0) - 0.2) < 1e-9


def test_interval_prf_perfect():
    ref = [(0.0, 5.0), (10.0, 15.0)]
    pred = [(0.1, 4.9), (10.2, 14.8)]
    m = interval_prf(pred, ref, iou_threshold=0.5)
    assert m["f1"] == 1.0


def test_false_cut_rate():
    ref = [(0.0, 10.0)]
    pred = [(0.0, 5.0)]
    assert abs(false_cut_rate(pred, ref) - 0.5) < 1e-9


def test_boundary_prf_tolerance():
    ref = [30.0, 90.0]
    pred = [32.0, 100.0]
    m = boundary_prf(pred, ref, tolerance_sec=15.0)
    assert m["tp"] == 2.0
    assert m["f1"] == 1.0


def test_normalize_intervals_and_boundaries():
    assert normalize_intervals([{"start": 1, "end": 3}]) == [(1.0, 3.0)]
    assert normalize_boundaries({"boundaries": [10, 20]}) == [10.0, 20.0]


def test_keyframe_and_ocr_and_chapter():
    kf = eval_keyframe_filter(["a", "b", "c"], ["a", "c"], ["a", "c"], video="v1")
    assert kf["f1"] == 1.0
    assert abs(kf["compression_ratio"] - 2 / 3) < 1e-9

    ocr = eval_ocr_items([{"image": "s1", "reference": "Hello\nWorld", "hypothesis": "Hello\nWorld"}])
    assert ocr[0]["cer"] == 0.0

    ch = eval_chapters(
        {"boundaries": [45.0, 120.0]},
        {"boundaries": [40.0, 125.0]},
        tolerance_sec=15.0,
        video="v1",
    )
    assert ch["f1"] == 1.0


def test_tvsum_windows():
    scores = [1] * 10 + [4] * 20 + [1] * 10
    wins = tvsum_important_windows(scores, fps=10.0, threshold=3.0, min_dur=0.5)
    assert len(wins) == 1
    bounds = tvsum_scene_boundaries([3, 3, 3, 1, 1, 1, 3, 3], fps=2.0, drop=0.5, min_gap_sec=0.1)
    assert isinstance(bounds, list)


def test_caption_and_rag_and_rouge():
    flags = caption_hallucination_flags("Keyframe for Scene 3", "Neural Networks")
    assert flags["hallucinated"] is True
    flags_empty = caption_hallucination_flags("A lecture talk video frame", "")
    assert isinstance(flags_empty["hallucinated"], bool)
    assert rag_hit_at_k(["a", "b", "c"], ["c", "z"], k=3) == 1.0
    assert rag_hit_at_k(["a", "b"], ["z"], k=3) == 0.0
    assert rouge_l_f1("the quick brown fox", "the quick brown fox") == 1.0


def test_extractive_and_reference_summary():
    text = "Alpha models learn patterns. Beta models need more data. Gamma models are compact."
    hyp = _extractive_summary(text, max_sents=2)
    assert "models" in hyp.lower()
    utts = [
        {"start": 0.0, "end": 5.0, "text": "First idea is stated here."},
        {"start": 6.0, "end": 10.0, "text": "More detail follows."},
        {"start": 45.0, "end": 50.0, "text": "Second block starts now."},
    ]
    ref = _reference_summary_from_utterances(utts, window=40.0)
    assert "First idea" in ref
    assert "Second block" in ref


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"OK {name}")
    print("ALL TESTS PASSED")

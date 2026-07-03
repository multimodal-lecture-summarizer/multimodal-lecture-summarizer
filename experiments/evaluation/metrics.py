"""Benchmark metrics for lecture pipeline evaluation.

Migrated from: src/mls/benchmarks/metrics.py
Script tính WER, ROUGE-L, BERTScore, F-score
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ASRMetrics:
    wer: float | None = None
    cer: float | None = None
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
    keyframe_slide_recall: float | None = None


@dataclass
class SemanticMetrics:
    ocr_char_accuracy: float | None = None
    clip_alignment_score: float | None = None
    caption_faithfulness: float | None = None


@dataclass
class SummaryMetrics:
    rouge_l: float | None = None
    bertscore_f1: float | None = None
    factuality_rate: float | None = None
    human_rating_1_5: float | None = None


@dataclass
class TimelineMetrics:
    slide_sync_mae_sec: float | None = None
    chapter_boundary_f1: float | None = None
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
    asr: ASRMetrics
    diarization: DiarizationMetrics
    visual: VisualMetrics
    semantic: SemanticMetrics
    summary: SummaryMetrics
    timeline: TimelineMetrics
    system: SystemMetrics

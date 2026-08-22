"""Pipeline evaluation suite for thesis tables (Bang 1–12).

Implements metrics and stage runners described in the TTTN/DATN evaluation plan:
ASR, VAD, Scene, Keyframe, OCR, Caption, Timeline, Chapter, Summary, RAG, Ablation.
"""

from experiments.evaluation.metrics import (
    asr_wer_cer,
    boundary_prf,
    interval_prf,
    rag_hit_at_k,
    rouge_l_f1,
    summary_text_metrics,
    timestamp_mae,
)

__all__ = [
    "asr_wer_cer",
    "boundary_prf",
    "interval_prf",
    "rag_hit_at_k",
    "rouge_l_f1",
    "summary_text_metrics",
    "timestamp_mae",
]

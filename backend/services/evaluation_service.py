"""Re-export alias for evaluation service."""

from app.services.evaluation_service import (
    calculate_wer_cer,
    calculate_summarization_metrics,
    get_benchmark_comparison_data,
    evaluate_video_ground_truth,
)

__all__ = [
    "calculate_wer_cer",
    "calculate_summarization_metrics",
    "get_benchmark_comparison_data",
    "evaluate_video_ground_truth",
]

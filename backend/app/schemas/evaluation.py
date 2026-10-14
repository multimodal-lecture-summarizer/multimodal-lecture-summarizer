from datetime import datetime
from typing import List, Optional, Any
from pydantic import Field
from app.schemas.base import CamelModel


class ASRBenchmarkItemDTO(CamelModel):
    model: str = Field(..., description="ASR Model variant name")
    params: str = Field(..., description="Parameter count e.g. 74M, 244M")
    wer: float = Field(..., description="Word Error Rate percentage (0-100%)")
    cer: float = Field(..., description="Character Error Rate percentage (0-100%)")
    rtf: float = Field(..., description="Real-Time Factor")
    relative_speed: str = Field(..., description="Speed relative to production base.en")
    is_production: bool = Field(default=False, description="Whether this is the current active model")
    notes: str = Field(..., description="Evaluation notes and trade-off justification")


class SummarizationBenchmarkItemDTO(CamelModel):
    method_id: str = Field(..., description="Unique method identifier")
    name: str = Field(..., description="Descriptive method name")
    rouge_1: float = Field(..., description="ROUGE-1 F1 score (0-100%)")
    rouge_2: float = Field(..., description="ROUGE-2 F1 score (0-100%)")
    rouge_l: float = Field(..., description="ROUGE-L F1 score (0-100%)")
    bertscore_f1: float = Field(..., description="BERTScore F1 score (0-100%)")
    avg_latency_sec: float = Field(..., description="Average generation latency in seconds")
    is_proposed: bool = Field(default=False, description="Highlighted as the proposed thesis contribution")
    description: str = Field(..., description="Technical architecture summary of the method")


class VisionBenchmarkItemDTO(CamelModel):
    model: str = Field(..., description="Vision/captioning model name")
    clip_score: float = Field(..., description="CLIPScore measuring visual alignment")
    latency_per_frame_sec: float = Field(..., description="Inference latency per keyframe")
    generic_rate: float = Field(..., description="Hallucination / generic caption rate percentage")
    is_production: bool = Field(default=False, description="Whether this is production default")


class BenchmarksDataDTO(CamelModel):
    asr_benchmarks: List[ASRBenchmarkItemDTO] = Field(default=[], description="Speech-to-text model comparison")
    summarization_benchmarks: List[SummarizationBenchmarkItemDTO] = Field(
        default=[], description="Summarization method comparison"
    )
    vision_benchmarks: List[VisionBenchmarkItemDTO] = Field(
        default=[], description="Keyframe captioning model comparison"
    )
    summary_conclusions: List[str] = Field(
        default=[], description="Key thesis takeaways from the empirical evaluation"
    )


class VideoEvaluationRequest(CamelModel):
    video_id: str = Field(..., description="Target video identifier to evaluate")
    reference_transcript: Optional[str] = Field(
        default=None, description="Ground truth transcript to measure WER/CER against"
    )
    reference_summary: Optional[str] = Field(
        default=None, description="Ground truth reference summary to measure ROUGE/BERTScore against"
    )


class ASREvalMetricsDTO(CamelModel):
    wer: float = Field(..., description="Word Error Rate fraction")
    cer: float = Field(..., description="Character Error Rate fraction")
    wer_pct: float = Field(..., description="Word Error Rate as percentage")
    cer_pct: float = Field(..., description="Character Error Rate as percentage")
    word_accuracy: float = Field(..., description="Word accuracy percentage (1 - WER)")


class SummaryEvalMetricsDTO(CamelModel):
    rouge_1: float = Field(..., description="ROUGE-1 unigram overlap score (0-100%)")
    rouge_2: float = Field(..., description="ROUGE-2 bigram overlap score (0-100%)")
    rouge_l: float = Field(..., description="ROUGE-L Longest Common Subsequence score (0-100%)")
    bertscore_f1: float = Field(..., description="BERTScore F1 semantic similarity (0-100%)")


class VideoEvaluationResultDTO(CamelModel):
    eval_id: str = Field(..., description="Unique evaluation result record ID")
    video_id: str = Field(..., description="Target video ID")
    video_title: str = Field(..., description="Target video title")
    has_transcript: bool = Field(default=False, description="Whether transcript was evaluated")
    has_summary: bool = Field(default=False, description="Whether summary was evaluated")
    asr_metrics: Optional[ASREvalMetricsDTO] = Field(default=None, description="ASR WER/CER evaluation")
    summary_metrics: Optional[SummaryEvalMetricsDTO] = Field(default=None, description="ROUGE & BERTScore evaluation")
    reference_transcript_preview: Optional[str] = Field(default=None)
    reference_summary_preview: Optional[str] = Field(default=None)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)

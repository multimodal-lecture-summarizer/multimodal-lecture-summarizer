from datetime import datetime
from typing import List, Optional, Any, Dict
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


class ObjectDetectionBenchmarkItemDTO(CamelModel):
    model: str = Field(..., description="OD Model variant name (e.g. YOLOv8n, YOLO11n, YOLOv8s)")
    params: str = Field(..., description="Parameter count e.g. 3.2M, 2.6M, 11.2M")
    map50: float = Field(..., description="mAP@50 score on lecture benchmark")
    map50_95: float = Field(..., description="mAP@50:95 score on lecture benchmark")
    precision: float = Field(..., description="Precision percentage (0-100%)")
    recall: float = Field(..., description="Recall percentage (0-100%)")
    latency_ms: float = Field(..., description="Inference latency per frame in milliseconds")
    fps: float = Field(..., description="Frames per second processing throughput")
    is_production: bool = Field(default=False, description="Whether this is the current production model")
    notes: str = Field(..., description="Trade-off justification")


class BenchmarksDataDTO(CamelModel):
    asr_benchmarks: List[ASRBenchmarkItemDTO] = Field(default=[], description="Speech-to-text model comparison")
    summarization_benchmarks: List[SummarizationBenchmarkItemDTO] = Field(
        default=[], description="Summarization method comparison"
    )
    vision_benchmarks: List[VisionBenchmarkItemDTO] = Field(
        default=[], description="Keyframe captioning model comparison"
    )
    object_detection_benchmarks: List[ObjectDetectionBenchmarkItemDTO] = Field(
        default=[], description="Object detection and tracking model comparison"
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
    reference_objects: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Ground truth reference object bounding boxes per scene"
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


class ODEvalMetricsDTO(CamelModel):
    od_precision: Optional[float] = Field(None, description="Object Detection precision metric")
    od_recall: Optional[float] = Field(None, description="Object Detection recall metric")
    od_map50: Optional[float] = Field(None, description="Mean Average Precision at IoU 0.50 (mAP@50)")
    od_map50_95: Optional[float] = Field(None, description="Mean Average Precision at IoU 0.50:0.95 (mAP@50-95)")


class VideoEvaluationResultDTO(CamelModel):
    eval_id: str = Field(..., description="Unique evaluation result record ID")
    video_id: str = Field(..., description="Target video ID")
    video_title: str = Field(..., description="Target video title")
    has_transcript: bool = Field(default=False, description="Whether transcript was evaluated")
    has_summary: bool = Field(default=False, description="Whether summary was evaluated")
    has_object_detection: bool = Field(default=False, description="Whether object detection was evaluated")
    asr_metrics: Optional[ASREvalMetricsDTO] = Field(default=None, description="ASR WER/CER evaluation")
    summary_metrics: Optional[SummaryEvalMetricsDTO] = Field(default=None, description="ROUGE & BERTScore evaluation")
    od_metrics: Optional[ODEvalMetricsDTO] = Field(default=None, description="Object Detection evaluation")
    od_precision: Optional[float] = Field(default=None, description="Object Detection precision")
    od_recall: Optional[float] = Field(default=None, description="Object Detection recall")
    od_map50: Optional[float] = Field(default=None, description="Object Detection mAP@50")
    od_map50_95: Optional[float] = Field(default=None, description="Object Detection mAP@50-95")
    od_details: Optional[Dict[str, Any]] = Field(default=None, description="Per-class AP and tracking breakdown")
    reference_transcript_preview: Optional[str] = Field(default=None)
    reference_summary_preview: Optional[str] = Field(default=None)
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationDTO(CamelModel):
    eval_id: str = Field(..., description="Unique evaluation result record ID")
    video_id: Optional[str] = Field(None, description="Target video ID")
    wer: Optional[float] = Field(None, description="Word Error Rate")
    cer: Optional[float] = Field(None, description="Character Error Rate")
    rouge_1: Optional[float] = Field(None, description="ROUGE-1 score")
    rouge_2: Optional[float] = Field(None, description="ROUGE-2 score")
    rouge_l: Optional[float] = Field(None, description="ROUGE-L score")
    bertscore_f1: Optional[float] = Field(None, description="BERTScore F1 score")
    od_precision: Optional[float] = Field(None, description="Object Detection precision")
    od_recall: Optional[float] = Field(None, description="Object Detection recall")
    od_map50: Optional[float] = Field(None, description="Object Detection mAP@50")
    od_map50_95: Optional[float] = Field(None, description="Object Detection mAP@50-95")
    reference_transcript: Optional[str] = Field(None, description="Ground truth transcript reference")
    reference_summary: Optional[str] = Field(None, description="Ground truth summary reference")
    details_json: Optional[Any] = Field(None, description="Audit breakdown details")
    created_at: Optional[datetime] = Field(None, description="Evaluation timestamp")


EvaluationResponse = EvaluationDTO

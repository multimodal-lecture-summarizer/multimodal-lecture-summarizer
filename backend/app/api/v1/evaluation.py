import logging
import uuid
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.exceptions import AppException, NotFoundException
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas.base import BaseDTO
from app.schemas.evaluation import (
    BenchmarksDataDTO,
    VideoEvaluationRequest,
    VideoEvaluationResultDTO,
    ObjectDetectionBenchmarkItemDTO,
    EvaluationDTO,
)
from app.api.deps import check_admin
from app.models.user import User
from app.models.evaluation import EvaluationLog
from app.services.evaluation_service import (
    get_benchmark_comparison_data,
    evaluate_video_ground_truth,
)

logger = logging.getLogger(__name__)

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.get(
    "/benchmarks",
    response_model=BaseDTO[BenchmarksDataDTO],
    summary="Get speech, summarization, vision, and object detection benchmark comparison data",
    description=(
        "Returns synthesized benchmark comparison tables for ASR models (Whisper base/small/large-v3), "
        "Summarization methods (Extractive baseline, Unimodal LLM, and Proposed Multimodal Fusion), "
        "Keyframe captioning (BLIP vs Florence-2), and Object Detection & Tracking (YOLOv8n vs YOLO11n vs YOLOv8s)."
    ),
)
def get_benchmarks(
    current_user: User = Depends(check_admin),
) -> BaseDTO[BenchmarksDataDTO]:
    """Retrieve benchmark comparison matrices across speech recognition, summarization, and vision/OD."""
    try:
        data = get_benchmark_comparison_data()
        return BaseDTO(
            success=True,
            data=data,
            code=200,
            message="Lấy dữ liệu benchmark so sánh mô hình thành công",
        )
    except Exception as exc:
        logger.error(f"Error fetching benchmark data: {exc}", exc_info=True)
        raise AppException(message=f"Không thể lấy dữ liệu benchmark: {str(exc)}")


@router.get(
    "/object-detection/benchmarks",
    response_model=BaseDTO[List[ObjectDetectionBenchmarkItemDTO]],
    summary="Get object detection and tracking benchmark comparison",
    description="Returns benchmark performance comparison between YOLOv8n, YOLO11n, and YOLOv8s on lecture dataset.",
)
def get_object_detection_benchmarks(
    current_user: User = Depends(check_admin),
) -> BaseDTO[List[ObjectDetectionBenchmarkItemDTO]]:
    """Retrieve benchmark comparison matrices for object detection and tracking."""
    try:
        data = get_benchmark_comparison_data()
        return BaseDTO(
            success=True,
            data=data.object_detection_benchmarks,
            code=200,
            message="Lấy bảng benchmark Object Detection thành công",
        )
    except Exception as exc:
        logger.error(f"Error fetching OD benchmark data: {exc}", exc_info=True)
        raise AppException(message=f"Không thể lấy dữ liệu benchmark OD: {str(exc)}")


@router.post(
    "/evaluate-video",
    response_model=BaseDTO[VideoEvaluationResultDTO],
    summary="Evaluate an existing video against ground-truth references",
    description=(
        "Computes real-time WER/CER for transcript, ROUGE-1/2/L and BERTScore-F1 for summary, "
        "and IoU, Precision, Recall, mAP@50, mAP@50:95 for object detection against ground-truth references. "
        "Stores audit record in evaluation_logs."
    ),
)
def evaluate_video(
    request: VideoEvaluationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
) -> BaseDTO[VideoEvaluationResultDTO]:
    """Execute live ground-truth evaluation for a given video."""
    try:
        result = evaluate_video_ground_truth(
            db=db,
            video_id=request.video_id,
            reference_transcript=request.reference_transcript,
            reference_summary=request.reference_summary,
            reference_objects=request.reference_objects,
        )
        return BaseDTO(
            success=True,
            data=result,
            code=200,
            message="Chấm điểm đánh giá chất lượng video thành công",
        )
    except ValueError as val_err:
        raise NotFoundException(message=str(val_err))
    except Exception as exc:
        logger.error(f"Error evaluating video {request.video_id}: {exc}", exc_info=True)
        raise AppException(message=f"Lỗi khi đánh giá video: {str(exc)}")


@router.get(
    "/history/{video_id}",
    response_model=BaseDTO[List[EvaluationDTO]],
    summary="Get evaluation history for a specific video",
    description="Returns historical evaluation audit runs including WER, ROUGE, and Object Detection metrics.",
)
def get_evaluation_history(
    video_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_admin),
) -> BaseDTO[List[EvaluationDTO]]:
    """Retrieve all evaluation log records for a given video."""
    try:
        video_uuid = uuid.UUID(video_id)
    except Exception:
        raise NotFoundException(message=f"Định dạng video ID không hợp lệ: {video_id}")

    try:
        logs = (
            db.query(EvaluationLog)
            .filter(EvaluationLog.video_id == video_uuid)
            .order_by(EvaluationLog.created_at.desc())
            .all()
        )
        data = [
            EvaluationDTO(
                eval_id=str(log.eval_id),
                video_id=str(log.video_id) if log.video_id else None,
                wer=log.wer,
                cer=log.cer,
                rouge_1=log.rouge_1,
                rouge_2=log.rouge_2,
                rouge_l=log.rouge_l,
                bertscore_f1=log.bertscore_f1,
                od_precision=log.od_precision,
                od_recall=log.od_recall,
                od_map50=log.od_map50,
                od_map50_95=log.od_map50_95,
                reference_transcript=log.reference_transcript,
                reference_summary=log.reference_summary,
                details_json=log.details_json,
                created_at=log.created_at,
            )
            for log in logs
        ]
        return BaseDTO(
            success=True,
            data=data,
            code=200,
            message="Lấy lịch sử đánh giá video thành công",
        )
    except Exception as exc:
        logger.error(f"Error fetching evaluation history for video {video_id}: {exc}", exc_info=True)
        raise AppException(message=f"Không thể lấy lịch sử đánh giá: {str(exc)}")

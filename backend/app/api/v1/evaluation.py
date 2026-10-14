import logging
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
)
from app.api.deps import check_admin
from app.models.user import User
from app.services.evaluation_service import (
    get_benchmark_comparison_data,
    evaluate_video_ground_truth,
)

logger = logging.getLogger(__name__)

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.get(
    "/benchmarks",
    response_model=BaseDTO[BenchmarksDataDTO],
    summary="Get speech and summarization benchmark comparison data",
    description=(
        "Returns synthesized benchmark comparison tables for ASR models (Whisper base/small/large-v3) "
        "and Summarization methods (Extractive baseline, Unimodal LLM, and Proposed Multimodal Fusion)."
    ),
)
def get_benchmarks(
    current_user: User = Depends(check_admin),
) -> BaseDTO[BenchmarksDataDTO]:
    """Retrieve benchmark comparison matrices for speech recognition and summarization."""
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


@router.post(
    "/evaluate-video",
    response_model=BaseDTO[VideoEvaluationResultDTO],
    summary="Evaluate an existing video against ground-truth references",
    description=(
        "Computes real-time WER/CER for transcript and ROUGE-1/2/L and BERTScore-F1 for summary "
        "against user-provided ground-truth references. Stores audit record in evaluation_logs."
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

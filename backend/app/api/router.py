from fastapi import APIRouter
from app.api.v1 import auth, videos, jobs, summaries, qa, stats, users, evaluation
from app.schemas.evaluation import BenchmarksDataDTO, VideoEvaluationResultDTO

api_router = APIRouter()

# Register sub-routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(videos.router, prefix="/videos", tags=["Videos Management"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Background Jobs"])
api_router.include_router(
    summaries.router, prefix="/summaries", tags=["AI Summarization"]
)
api_router.include_router(qa.router, prefix="/qa", tags=["Interactive Q&A"])
api_router.include_router(stats.router, prefix="/stats", tags=["Admin Reports"])
api_router.include_router(users.router, prefix="/users", tags=["Users Management"])
api_router.include_router(
    evaluation.router, prefix="/evaluation", tags=["Evaluation & Benchmarks"]
)

# Admin router alias for /admin/* endpoints
admin_router = APIRouter(prefix="/admin", tags=["Admin Settings"])
admin_router.add_api_route(
    "/video-standards",
    videos.get_standards,
    methods=["GET"],
    response_model=videos.BaseDTO[videos.VideoStandardDTO],
    summary="Retrieve active video standards (Admin)",
)
admin_router.add_api_route(
    "/video-standards",
    videos.update_standards,
    methods=["PUT"],
    response_model=videos.BaseDTO[videos.VideoStandardDTO],
    summary="Update video standards (Admin)",
)
admin_router.add_api_route(
    "/evaluation/benchmarks",
    evaluation.get_benchmarks,
    methods=["GET"],
    response_model=videos.BaseDTO[BenchmarksDataDTO],
    summary="Evaluation Benchmarks Comparison (Admin Alias)",
)
admin_router.add_api_route(
    "/evaluation/evaluate-video",
    evaluation.evaluate_video,
    methods=["POST"],
    response_model=videos.BaseDTO[VideoEvaluationResultDTO],
    summary="Live Video Evaluation (Admin Alias)",
)
api_router.include_router(admin_router)


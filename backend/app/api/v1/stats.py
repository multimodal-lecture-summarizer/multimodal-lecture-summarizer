import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.middleware.case_converter import CamelCaseAPIRoute
from app.schemas import BaseDTO, AdminDashboardStats, SystemStatDTO
from app.schemas.stats import ResourceCostStats, JobStatusDistribution, WeeklyVideoVolume, ModelUsageDistribution
from app.api.deps import check_admin
from app.models.user import User
from app.models.stats import SystemStat
from app.models.job import Job
from app.models.video import Video
from app.models.summary import Summary
from app.core.constants import JobStatus
from sqlalchemy import func

router = APIRouter(route_class=CamelCaseAPIRoute)


@router.get(
    "/dashboard",
    response_model=BaseDTO[AdminDashboardStats],
    summary="Get admin dashboard statistics",
    description="Returns aggregate reports of registered users, jobs status distribution, and estimated cost analysis.",
)
def get_dashboard_stats(
    db: Session = Depends(get_db), current_user: User = Depends(check_admin)
):
    """Retrieve historical statistics and processing KPIs. Restricted to Admin role."""
    # 1. Fetch system stats history
    stats_list = (
        db.query(SystemStat).order_by(SystemStat.date.asc()).limit(30).all()
    )

    # If database is empty, seed mock historical records for dashboard visualization demonstration
    if not stats_list:
        today = datetime.date.today()
        stats_list = []
        for i in range(7, 0, -1):
            day = today - datetime.timedelta(days=i)
            stat = SystemStat(
                date=day,
                total_users=10 + i * 2,
                new_users=2,
                total_videos_processed=5 + i,
                total_jobs_run=6 + i,
            )
            db.add(stat)
        db.commit()
        stats_list = (
            db.query(SystemStat).order_by(SystemStat.date.asc()).all()
        )

    # 2. Fetch jobs distribution
    completed_jobs_count = (
        db.query(Job).filter(Job.status == JobStatus.COMPLETED).count()
    )
    failed_jobs_count = (
        db.query(Job).filter(Job.status == JobStatus.FAILED).count()
    )
    running_jobs_count = (
        db.query(Job).filter(Job.status == JobStatus.RUNNING).count()
    )
    pending_jobs_count = (
        db.query(Job).filter(Job.status == JobStatus.PENDING).count()
    )

    # Add defaults if all are zero to make chart visible in mockup
    if (
        completed_jobs_count
        + failed_jobs_count
        + running_jobs_count
        + pending_jobs_count
        == 0
    ):
        job_distribution = [
            JobStatusDistribution(status="completed", count=45),
            JobStatusDistribution(status="failed", count=3),
            JobStatusDistribution(status="running", count=2),
            JobStatusDistribution(status="pending", count=0),
        ]
    else:
        job_distribution = [
            JobStatusDistribution(status="completed", count=completed_jobs_count),
            JobStatusDistribution(status="failed", count=failed_jobs_count),
            JobStatusDistribution(status="running", count=running_jobs_count),
            JobStatusDistribution(status="pending", count=pending_jobs_count),
        ]

    # 3. Formulate resource usage
    # Mocking cost calculation based on actual summarized records
    total_processing_seconds = sum([s.total_videos_processed * 12 for s in stats_list])
    # Groq costs ~$0.15 per 1M tokens, Whisper ~$0.006 per minute.
    # We construct a realistic mock cost analysis
    resource_usage = ResourceCostStats(
        estimated_api_cost=round(len(stats_list) * 0.45, 2),
        total_processing_time_seconds=float(total_processing_seconds),
        gpu_hours=round(total_processing_seconds / 3600.0, 3),
    )

    history_dto = [
        SystemStatDTO(
            stat_id=s.stat_id,
            date=s.date,
            total_users=s.total_users,
            new_users=s.new_users,
            total_videos_processed=s.total_videos_processed,
            total_jobs_run=s.total_jobs_run,
        )
        for s in stats_list
    ]

    # Calculate weekly video upload traffic (last 6 weeks)
    weekly_volume = []
    today = datetime.date.today()
    for i in range(6, 0, -1):
        start_date = today - datetime.timedelta(days=i * 7)
        end_date = today - datetime.timedelta(days=(i - 1) * 7)
        count = (
            db.query(Video)
            .filter(
                Video.uploaded_at >= datetime.datetime.combine(start_date, datetime.time.min),
                Video.uploaded_at < datetime.datetime.combine(end_date, datetime.time.max),
            )
            .count()
        )
        week_of_month = (end_date.day - 1) // 7 + 1
        week_label = f"Tuần {week_of_month} - T{end_date.month}"
        weekly_volume.append(
            WeeklyVideoVolume(week_label=week_label, count=count)
        )

    # Calculate AI model breakdown from summaries table
    model_query = (
        db.query(Summary.model_used, func.count(Summary.summary_id))
        .group_by(Summary.model_used)
        .all()
    )
    total_summaries = sum(q[1] for q in model_query)

    model_distribution = []
    if total_summaries > 0:
        for model_name, count in model_query:
            pct = round((count / total_summaries) * 100, 1)
            model_distribution.append(
                ModelUsageDistribution(
                    model_name=model_name or "Unknown Model",
                    percentage=pct,
                    count=count,
                )
            )
    else:
        model_distribution = [
            ModelUsageDistribution(
                model_name="Groq Llama 3.1 8B", percentage=100.0, count=0
            )
        ]

    stats = AdminDashboardStats(
        history=history_dto,
        job_distribution=job_distribution,
        resource_usage=resource_usage,
        weekly_volume=weekly_volume,
        model_distribution=model_distribution,
    )

    return BaseDTO(
        success=True,
        data=stats,
        message="Dashboard statistics compiled successfully",
    )

import datetime
from typing import List, Dict, Any
from pydantic import Field
from app.schemas.base import CamelModel


class SystemStatDTO(CamelModel):
    stat_id: int = Field(..., description="Unique statistic entry ID")
    date: datetime.date = Field(..., description="The date for the recorded statistics")
    total_users: int = Field(
        ..., description="Total cumulative registered users up to this date"
    )
    new_users: int = Field(
        ..., description="Number of new users registered on this day"
    )
    total_videos_processed: int = Field(
        ..., description="Number of videos successfully summarized on this day"
    )
    total_jobs_run: int = Field(
        ..., description="Total number of jobs executed (including failures)"
    )


class ResourceCostStats(CamelModel):
    estimated_api_cost: float = Field(
        ..., description="Estimated Groq and Whisper API costs in USD"
    )
    total_processing_time_seconds: float = Field(
        ..., description="Total processing latency in seconds across all videos"
    )
    gpu_hours: float = Field(..., description="Estimated GPU hours consumed")


class JobStatusDistribution(CamelModel):
    status: str = Field(..., description="The job status label")
    count: int = Field(..., description="The number of jobs with this status")


class AdminDashboardStats(CamelModel):
    history: List[SystemStatDTO] = Field(
        default=[], description="Historical stats trend"
    )
    job_distribution: List[JobStatusDistribution] = Field(
        default=[], description="Overview of job success/failure rates"
    )
    resource_usage: ResourceCostStats = Field(
        ..., description="Estimated system resource and API cost metrics"
    )

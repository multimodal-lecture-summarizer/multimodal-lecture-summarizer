import uuid
from datetime import datetime
from typing import Optional
from pydantic import Field
from app.schemas.base import CamelModel
from app.core.constants import JobType, JobStatus


class JobBase(CamelModel):
    job_type: JobType = Field(..., description="The type of the processing job")
    status: JobStatus = Field(
        JobStatus.PENDING, description="The current execution status of the job"
    )


class JobDTO(JobBase):
    job_id: uuid.UUID = Field(..., description="The unique UUID of the job")
    video_id: uuid.UUID = Field(
        ..., description="The UUID of the video associated with this job"
    )
    started_at: Optional[datetime] = Field(
        None, description="The timestamp when the job started execution"
    )
    completed_at: Optional[datetime] = Field(
        None, description="The timestamp when the job finished execution"
    )
    error_log: Optional[str] = Field(
        None, description="The detailed error log if the job failed"
    )
    progress: Optional[int] = Field(
        None, description="The execution progress percentage (0-100)"
    )
    stage: Optional[str] = Field(
        None, description="The current execution stage name"
    )

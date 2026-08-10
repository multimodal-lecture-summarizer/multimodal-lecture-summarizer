"""Regression tests for job runtime status metadata."""

from __future__ import annotations

import unittest
import uuid
from unittest.mock import patch

from app.api.v1.jobs import sync_job_status
from app.core.constants import JobStatus


class _FakeAsyncResult:
    state = "SUCCESS"
    result = {
        "stage": "completed",
        "progress": 100,
        "logs": ["[10:00:00] Đang yêu cầu mô hình AI sinh bản tóm tắt.", "[10:00:10] Hoàn tất xử lý video và lưu kết quả."],
    }
    info = result


class _FakeJob:
    def __init__(self):
        self.job_id = uuid.uuid4()
        self.video_id = uuid.uuid4()
        self.status = JobStatus.COMPLETED
        self.progress = None
        self.stage = None
        self.logs = ["old log"]


class JobRuntimeLogsTests(unittest.TestCase):
    def test_completed_job_keeps_logs_from_celery_success_result(self):
        job = _FakeJob()

        with patch("celery.result.AsyncResult", return_value=_FakeAsyncResult()):
            sync_job_status(job, db=None)

        self.assertEqual(job.stage, "completed")
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.logs, _FakeAsyncResult.result["logs"])


if __name__ == "__main__":
    unittest.main()

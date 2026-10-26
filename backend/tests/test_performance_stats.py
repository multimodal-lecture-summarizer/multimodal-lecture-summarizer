import unittest
from unittest.mock import MagicMock
from app.api.v1.stats import get_performance_stats
from app.models.user import User
from app.models.job import Job
from app.core.constants import UserRole


class TestPerformanceStatsEndpoint(unittest.TestCase):
    def test_get_performance_stats_structure(self):
        mock_db = MagicMock()
        # Mock completed_jobs query returning empty to exercise benchmark seeds
        mock_query = MagicMock()
        mock_query.join.return_value = mock_query
        mock_query.options.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.count.return_value = 10
        mock_db.query.return_value = mock_query

        mock_user = User(user_id=None, email="admin@test.com", role=UserRole.ADMIN)

        res = get_performance_stats(db=mock_db, current_user=mock_user)

        self.assertTrue(res.success)
        self.assertIsNotNone(res.data)
        data = res.data

        # Check summary metrics
        self.assertGreater(data.summary_metrics.avg_rtf, 0)
        self.assertGreater(data.summary_metrics.avg_processing_time_sec, 0)
        self.assertGreater(data.summary_metrics.total_processed_hours, 0)
        self.assertGreater(data.summary_metrics.total_jobs_evaluated, 0)

        # Check scatter data points
        self.assertGreater(len(data.scatter_data), 5)
        first_point = data.scatter_data[0]
        self.assertTrue(hasattr(first_point, "video_id"))
        self.assertTrue(hasattr(first_point, "duration_sec"))
        self.assertTrue(hasattr(first_point, "processing_sec"))
        self.assertTrue(hasattr(first_point, "rtf"))
        self.assertTrue(hasattr(first_point, "resolution"))

        # Check duration buckets
        self.assertEqual(len(data.duration_buckets), 3)
        buckets = {b.bucket: b for b in data.duration_buckets}
        self.assertIn("short", buckets)
        self.assertIn("medium", buckets)
        self.assertIn("long", buckets)
        self.assertGreater(buckets["short"].count, 0)
        self.assertGreater(buckets["medium"].count, 0)
        self.assertGreater(buckets["long"].count, 0)
        self.assertGreater(buckets["short"].success_rate, 80.0)


if __name__ == "__main__":
    unittest.main()

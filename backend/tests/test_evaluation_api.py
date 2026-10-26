import unittest
import uuid
from unittest.mock import MagicMock

from app.api.v1.evaluation import get_benchmarks, evaluate_video
from app.models.user import User
from app.models.video import Video, VideoScene
from app.models.summary import Summary
from app.core.constants import UserRole
from app.schemas.evaluation import VideoEvaluationRequest
from app.services.evaluation_service import (
    calculate_wer_cer,
    calculate_summarization_metrics,
    get_benchmark_comparison_data,
)


class TestEvaluationServiceAndApi(unittest.TestCase):
    def test_calculate_wer_cer_identical(self):
        ref = "Welcome to the introduction to artificial intelligence lecture."
        hyp = "Welcome to the introduction to artificial intelligence lecture."
        res = calculate_wer_cer(ref, hyp)
        self.assertEqual(res["wer"], 0.0)
        self.assertEqual(res["cer"], 0.0)
        self.assertEqual(res["word_accuracy"], 100.0)

    def test_calculate_wer_cer_differences(self):
        ref = "Deep learning uses neural networks."
        hyp = "Deep learning uses deep neural network."
        res = calculate_wer_cer(ref, hyp)
        self.assertGreater(res["wer"], 0.0)
        self.assertLessEqual(res["wer"], 1.0)
        self.assertGreater(res["cer"], 0.0)

    def test_calculate_summarization_metrics_identical(self):
        ref = "This lecture covers gradient descent and backpropagation optimization."
        hyp = "This lecture covers gradient descent and backpropagation optimization."
        res = calculate_summarization_metrics(ref, hyp)
        self.assertAlmostEqual(res["rouge_1"], 100.0, places=1)
        self.assertAlmostEqual(res["rouge_2"], 100.0, places=1)
        self.assertAlmostEqual(res["rouge_l"], 100.0, places=1)
        self.assertGreaterEqual(res["bertscore_f1"], 90.0)

    def test_get_benchmark_comparison_data(self):
        benchmarks = get_benchmark_comparison_data()
        self.assertIsNotNone(benchmarks)

        # Check ASR models
        self.assertEqual(len(benchmarks.asr_benchmarks), 3)
        models = [m.model for m in benchmarks.asr_benchmarks]
        self.assertTrue(any("base" in m.lower() for m in models))
        self.assertTrue(any("small" in m.lower() for m in models))
        self.assertTrue(any("large" in m.lower() for m in models))

        # Check Summarization methods
        self.assertEqual(len(benchmarks.summarization_benchmarks), 3)
        method_ids = [m.method_id for m in benchmarks.summarization_benchmarks]
        self.assertIn("extractive_baseline", method_ids)
        self.assertIn("unimodal_llm", method_ids)
        self.assertIn("multimodal_fusion", method_ids)

        proposed = [m for m in benchmarks.summarization_benchmarks if m.is_proposed]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(proposed[0].method_id, "multimodal_fusion")
        self.assertGreater(proposed[0].rouge_1, 45.0)
        self.assertGreater(proposed[0].bertscore_f1, 90.0)

        # Check Vision models
        self.assertGreater(len(benchmarks.vision_benchmarks), 0)

    def test_get_benchmarks_api_handler(self):
        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        res = get_benchmarks(current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertIsNotNone(res.data)
        self.assertEqual(len(res.data.summarization_benchmarks), 3)

    def test_evaluate_video_api_handler(self):
        video_id = uuid.uuid4()
        mock_video = Video(
            video_id=video_id,
            title="AI Introduction Lecture",
            file_path="/storage/videos/ai_intro.mp4",
        )
        mock_scene = VideoScene(
            scene_id=uuid.uuid4(),
            video_id=video_id,
            scene_index=0,
            script="Today we will learn machine learning algorithms and neural network architectures.",
        )
        mock_summary = Summary(
            summary_id=uuid.uuid4(),
            video_id=video_id,
            summary_text="Overview of machine learning algorithms and neural networks.",
        )

        mock_db = MagicMock()
        mock_query = MagicMock()

        def query_side_effect(model_cls):
            q = MagicMock()
            if model_cls == Video:
                q.filter.return_value.first.return_value = mock_video
            elif model_cls == VideoScene:
                q.filter.return_value.order_by.return_value.all.return_value = [mock_scene]
            elif model_cls == Summary:
                q.filter.return_value.first.return_value = mock_summary
            return q

        mock_db.query.side_effect = query_side_effect

        mock_user = User(user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN)
        req = VideoEvaluationRequest(
            video_id=str(video_id),
            reference_transcript="Today we will learn machine learning algorithms and neural networks.",
            reference_summary="Overview of machine learning algorithms and deep neural networks.",
        )

        res = evaluate_video(request=req, db=mock_db, current_user=mock_user)
        self.assertTrue(res.success)
        self.assertEqual(res.code, 200)
        self.assertIsNotNone(res.data)
        self.assertEqual(res.data.video_id, str(video_id))
        self.assertIsNotNone(res.data.asr_metrics)
        self.assertIsNotNone(res.data.summary_metrics)
        self.assertGreater(res.data.summary_metrics.rouge_1, 0.0)
        self.assertGreater(res.data.summary_metrics.bertscore_f1, 0.0)
        self.assertLessEqual(res.data.asr_metrics.wer, 1.0)


if __name__ == "__main__":
    unittest.main()

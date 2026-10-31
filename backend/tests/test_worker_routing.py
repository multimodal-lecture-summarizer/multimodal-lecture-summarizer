"""Video jobs go to the GPU queue only when a GPU worker is online and can download the video."""

import unittest
from unittest.mock import MagicMock

from app.services.worker_routing import CPU_QUEUE, GPU_QUEUE, choose_video_queue


def _celery_with_queues(active_queues):
    celery = MagicMock()
    celery.control.inspect.return_value.active_queues.return_value = active_queues
    return celery


class ChooseVideoQueueTest(unittest.TestCase):
    def test_gpu_worker_online_with_remote_video_uses_gpu_queue(self):
        celery = _celery_with_queues({"colab@host": [{"name": GPU_QUEUE}]})
        self.assertEqual(choose_video_queue(celery, "https://pub-x.r2.dev/videos/a.mp4"), GPU_QUEUE)

    def test_only_cpu_worker_online_uses_cpu_queue(self):
        celery = _celery_with_queues({"local@pc": [{"name": CPU_QUEUE}]})
        self.assertEqual(choose_video_queue(celery, "https://youtu.be/abc"), CPU_QUEUE)

    def test_no_worker_reply_uses_cpu_queue(self):
        celery = _celery_with_queues(None)
        self.assertEqual(choose_video_queue(celery, "https://youtu.be/abc"), CPU_QUEUE)

    def test_local_file_never_goes_to_gpu_queue(self):
        celery = _celery_with_queues({"colab@host": [{"name": GPU_QUEUE}]})
        self.assertEqual(choose_video_queue(celery, "C:/storage/mock_r2_bucket/videos/a.mp4"), CPU_QUEUE)
        celery.control.inspect.assert_not_called()

    def test_broker_error_uses_cpu_queue(self):
        celery = MagicMock()
        celery.control.inspect.side_effect = ConnectionError("redis down")
        self.assertEqual(choose_video_queue(celery, "https://youtu.be/abc"), CPU_QUEUE)


if __name__ == "__main__":
    unittest.main()

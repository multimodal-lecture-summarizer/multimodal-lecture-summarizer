"""Video jobs go to the GPU queue only when a GPU worker is online and can download the video."""

import unittest
from unittest.mock import MagicMock

from app.services.worker_routing import (
    CPU_QUEUE,
    GPU_QUEUE,
    PREPARE_YOUTUBE_TASK,
    PROCESS_VIDEO_TASK,
    choose_video_queue,
    dispatch_video_job,
)


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


class DispatchVideoJobTest(unittest.TestCase):
    YOUTUBE_URL = "https://www.youtube.com/watch?v=abc"
    R2_URL = "https://pub-x.r2.dev/videos/a.mp4"
    PROCESS_KWARGS = {"output_language": "en"}

    def _dispatch(self, celery, video_path, r2_enabled=True):
        return dispatch_video_job(
            celery, "job-1", video_path, self.PROCESS_KWARGS, max_duration=3600, r2_enabled=r2_enabled
        )

    def test_youtube_with_gpu_worker_is_downloaded_locally_first(self):
        celery = _celery_with_queues({"colab@host": [{"name": GPU_QUEUE}]})

        self.assertEqual(self._dispatch(celery, self.YOUTUBE_URL), CPU_QUEUE)

        celery.send_task.assert_called_once_with(
            PREPARE_YOUTUBE_TASK,
            args=["job-1", self.YOUTUBE_URL, GPU_QUEUE],
            kwargs={"output_language": "en", "max_duration": 3600},
            task_id="job-1",
            queue=CPU_QUEUE,
        )

    def test_youtube_on_cpu_worker_keeps_direct_processing(self):
        celery = _celery_with_queues({"local@pc": [{"name": CPU_QUEUE}]})

        self.assertEqual(self._dispatch(celery, self.YOUTUBE_URL), CPU_QUEUE)

        self.assertEqual(celery.send_task.call_args.args[0], PROCESS_VIDEO_TASK)

    def test_youtube_without_r2_keeps_direct_processing(self):
        celery = _celery_with_queues({"colab@host": [{"name": GPU_QUEUE}]})

        self.assertEqual(self._dispatch(celery, self.YOUTUBE_URL, r2_enabled=False), GPU_QUEUE)

        self.assertEqual(celery.send_task.call_args.args[0], PROCESS_VIDEO_TASK)

    def test_r2_video_goes_straight_to_gpu_worker(self):
        celery = _celery_with_queues({"colab@host": [{"name": GPU_QUEUE}]})

        self.assertEqual(self._dispatch(celery, self.R2_URL), GPU_QUEUE)

        celery.send_task.assert_called_once_with(
            PROCESS_VIDEO_TASK,
            args=["job-1", self.R2_URL],
            kwargs=self.PROCESS_KWARGS,
            task_id="job-1",
            queue=GPU_QUEUE,
        )


if __name__ == "__main__":
    unittest.main()

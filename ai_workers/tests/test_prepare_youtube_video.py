"""A YouTube video is fetched on the local worker, published to R2, then handed to the GPU queue."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from celery.exceptions import Ignore

from ai_workers import tasks
from ai_workers.core import youtube_ingest
from ai_workers.core.config import worker_settings

URL = "https://www.youtube.com/watch?v=abc"
R2_URL = "https://pub-x.r2.dev/videos/youtube_job-1.mp4"


def _fake_download(_url, dest_path, _max_duration):
    with open(dest_path, "wb") as handle:
        handle.write(b"video")
    return dest_path


class PrepareYoutubeVideoTests(unittest.TestCase):
    def setUp(self):
        for patcher in (
            patch.object(tasks.prepare_youtube_video, "update_state"),
            patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", "key"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _run(self):
        return tasks.prepare_youtube_video.run(
            "job-1", URL, "gpu", max_duration=3600, output_language="en"
        )

    def test_uploads_then_hands_job_to_gpu_queue_under_same_task_id(self):
        uploaded_from = []

        def fake_upload(local_path, object_key):
            uploaded_from.append(local_path)
            self.assertTrue(os.path.exists(local_path))
            self.assertEqual(object_key, "videos/youtube_job-1.mp4")
            return R2_URL

        with patch("ai_workers.tasks.download_youtube", side_effect=_fake_download) as download, patch(
            "ai_workers.tasks.upload_video_to_r2", side_effect=fake_upload
        ), patch.object(tasks.app, "send_task") as send_task:
            with self.assertRaises(Ignore):
                self._run()

        self.assertEqual(download.call_args.args[2], 3600)
        send_task.assert_called_once_with(
            "ai_workers.process_video",
            args=["job-1", R2_URL],
            kwargs={"output_language": "en"},
            task_id="job-1",
            queue="gpu",
        )
        self.assertFalse(os.path.exists(uploaded_from[0]), "temp download must be removed")

    def test_download_failure_fails_the_job_without_handoff(self):
        with patch(
            "ai_workers.tasks.download_youtube", side_effect=Exception("Sign in to confirm")
        ), patch.object(tasks.app, "send_task") as send_task:
            with self.assertRaisesRegex(RuntimeError, "Sign in to confirm"):
                self._run()

        send_task.assert_not_called()

    def test_upload_failure_does_not_hand_off_an_unreachable_video(self):
        with patch("ai_workers.tasks.download_youtube", side_effect=_fake_download), patch(
            "ai_workers.tasks.upload_video_to_r2", side_effect=Exception("R2 denied")
        ), patch.object(tasks.app, "send_task") as send_task:
            with self.assertRaisesRegex(RuntimeError, "R2 denied"):
                self._run()

        send_task.assert_not_called()

    def test_missing_r2_config_fails_before_downloading(self):
        with patch.object(worker_settings, "CF_R2_ACCESS_KEY_ID", ""), patch(
            "ai_workers.tasks.download_youtube"
        ) as download:
            with self.assertRaisesRegex(RuntimeError, "R2"):
                self._run()

        download.assert_not_called()


class DownloadYoutubeTests(unittest.TestCase):
    def _ydl(self, duration):
        ydl = MagicMock()
        ydl.extract_info.return_value = {"duration": duration}
        context = MagicMock()
        context.__enter__.return_value = ydl
        return ydl, context

    def test_video_over_limit_is_rejected_before_media_download(self):
        ydl, context = self._ydl(4000)

        with patch("yt_dlp.YoutubeDL", return_value=context):
            with self.assertRaisesRegex(ValueError, "3600"):
                youtube_ingest.download_youtube(URL, "video.mp4", max_duration=3600)

        ydl.download.assert_not_called()

    def test_without_limit_skips_metadata_probe(self):
        ydl, context = self._ydl(4000)

        with patch("yt_dlp.YoutubeDL", return_value=context):
            youtube_ingest.download_youtube(URL, "missing-video.mp4")

        ydl.extract_info.assert_not_called()
        ydl.download.assert_called_once_with([URL])


if __name__ == "__main__":
    unittest.main()

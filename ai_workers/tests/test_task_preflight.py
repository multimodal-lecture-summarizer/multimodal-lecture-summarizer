from __future__ import annotations

import unittest
from unittest.mock import mock_open, patch

from ai_workers import tasks
from ai_workers.core import resource_cleanup


class TaskPreflightTests(unittest.TestCase):
    def test_process_video_preflight_failure_does_not_write_malformed_failure_meta(self):
        error = RuntimeError("Insufficient RAM to start video processing: 100 MB available")

        with patch("ai_workers.tasks.ensure_process_memory_available", side_effect=error), patch.object(
            tasks.process_video, "update_state"
        ) as update_state:
            with self.assertRaises(RuntimeError):
                tasks.process_video.run("job-1", "input.mp4")

        update_state.assert_called_once()
        self.assertEqual(update_state.call_args.kwargs["state"], "PROGRESS")
        self.assertEqual(update_state.call_args.kwargs["meta"]["stage"], "preflight")
        self.assertIn("error", update_state.call_args.kwargs["meta"])

    def test_release_worker_resources_clears_cuda_cache_when_available(self):
        with patch("gc.collect") as collect, patch(
            "torch.cuda.is_available", return_value=True
        ), patch("torch.cuda.synchronize") as synchronize, patch(
            "torch.cuda.empty_cache"
        ) as empty_cache, patch("torch.cuda.ipc_collect") as ipc_collect, patch(
            "torch.cuda.reset_peak_memory_stats"
        ) as reset_peak, patch(
            "ai_workers.core.resource_cleanup.get_available_memory_mb", return_value=8192
        ), patch(
            "ai_workers.core.resource_cleanup.get_cuda_memory_snapshot", return_value=(5000, 6144)
        ):
            resource_cleanup.release_worker_resources("test")

        collect.assert_called_once()
        synchronize.assert_called_once()
        empty_cache.assert_called_once()
        ipc_collect.assert_called_once()
        reset_peak.assert_called_once()

    def test_memory_guard_allows_soft_floor_after_retry_budget(self):
        with patch(
            "ai_workers.core.resource_cleanup.get_available_memory_mb",
            return_value=3526,
        ), patch("ai_workers.core.resource_cleanup.release_worker_resources"), patch(
            "time.sleep"
        ):
            available_mb = resource_cleanup.ensure_process_memory_available(
                4096,
                soft_min_available_mb=3072,
                retry_seconds=0,
                retry_interval_seconds=1,
            )

        self.assertEqual(available_mb, 3526)

    def test_linux_memory_fallback_uses_mem_available_instead_of_free_pages(self):
        meminfo = "\n".join(
            [
                "MemTotal:        7460000 kB",
                "MemFree:         1392000 kB",
                "Buffers:          120000 kB",
                "Cached:          2600000 kB",
                "MemAvailable:    5440000 kB",
            ]
        )

        with patch("sys.platform", "linux"), patch("builtins.open", mock_open(read_data=meminfo)):
            self.assertEqual(resource_cleanup._read_linux_mem_available_mb(), 5312)

    def test_memory_guard_rejects_below_soft_floor(self):
        with patch(
            "ai_workers.core.resource_cleanup.get_available_memory_mb",
            return_value=2048,
        ), patch("ai_workers.core.resource_cleanup.release_worker_resources"), patch(
            "time.sleep"
        ):
            with self.assertRaisesRegex(RuntimeError, "hard floor"):
                resource_cleanup.ensure_process_memory_available(
                    4096,
                    soft_min_available_mb=3072,
                    retry_seconds=0,
                    retry_interval_seconds=1,
                )


if __name__ == "__main__":
    unittest.main()

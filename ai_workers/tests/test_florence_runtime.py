from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ai_workers.modules.visual_v2.florence_runtime import (
    FlorenceDeterminism,
    FLORENCE_ASSET_SHA256,
    FlorenceRuntime,
    FlorenceRuntimeError,
    resolve_florence_runtime,
    SUPPORTED_PACKAGES,
    validate_florence_environment,
    verify_florence_model,
)
from ai_workers.core.config import WorkerSettings


class FlorenceRuntimeTests(unittest.TestCase):
    def test_installed_environment_matches_contract(self):
        validate_florence_environment()

    def test_default_worker_setting_keeps_florence_on_cpu(self):
        settings = WorkerSettings(_env_file=None)
        self.assertEqual(settings.FLORENCE_DEVICE, "cpu")

    def test_version_drift_is_rejected(self):
        with patch.dict(SUPPORTED_PACKAGES, {"transformers": "0.0.0"}, clear=True):
            with self.assertRaisesRegex(FlorenceRuntimeError, "Unsupported Florence-2 runtime"):
                validate_florence_environment()

    def test_cpu_is_resolved_without_cuda_dependency(self):
        with patch("torch.cuda.is_available", return_value=False):
            runtime = resolve_florence_runtime("cpu")

        self.assertEqual(runtime.device, "cpu")
        self.assertEqual(runtime.dtype, torch.float32)
        self.assertEqual(runtime.attention_implementation, "eager")

    def test_cuda_request_fails_when_cuda_is_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(FlorenceRuntimeError, "CUDA is not available"):
                resolve_florence_runtime("cuda")

    def test_invalid_device_fails_fast(self):
        with self.assertRaisesRegex(FlorenceRuntimeError, "FLORENCE_DEVICE"):
            resolve_florence_runtime("auto")

    def test_verified_checkout_matches_asset_manifest(self):
        model_dir = Path(__file__).resolve().parents[1] / "modules" / "visual_v2" / "florence2_vendor"
        verify_florence_model(model_dir)

    def test_text_asset_hash_is_cross_platform_newline_stable(self):
        expected = hashlib.sha256(b"line1\nline2\n").hexdigest()
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_bytes(b"line1\r\nline2\r\n")
            with patch.dict(FLORENCE_ASSET_SHA256, {"config.json": expected}, clear=True):
                verify_florence_model(temp_dir)

    def test_missing_non_weight_asset_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                FLORENCE_ASSET_SHA256,
                {"config.json": FLORENCE_ASSET_SHA256["config.json"]},
                clear=True,
            ):
                with self.assertRaisesRegex(FlorenceRuntimeError, "vendored model checkout"):
                    verify_florence_model(temp_dir)

    def test_missing_checkpoint_reports_git_lfs_action(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FlorenceRuntimeError, "git lfs pull"):
                verify_florence_model(temp_dir)

    def test_checkpoint_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "model.safetensors").write_bytes(b"not-the-model")
            with self.assertRaisesRegex(FlorenceRuntimeError, "SHA-256 mismatch"):
                verify_florence_model(temp_dir)

    def test_determinism_guard_serializes_threads(self):
        runtime = FlorenceRuntime(device="cpu")
        first = FlorenceDeterminism(runtime)
        second_acquired = threading.Event()

        first.enable()

        def run_second_guard():
            second = FlorenceDeterminism(runtime)
            second.enable()
            second_acquired.set()
            second.restore()

        thread = threading.Thread(target=run_second_guard)
        thread.start()
        self.assertFalse(second_acquired.wait(0.1))

        first.restore()
        self.assertTrue(second_acquired.wait(2.0))
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())

    def test_determinism_setup_failure_restores_state(self):
        torch.manual_seed(4321)
        rng_state = torch.random.get_rng_state().clone()
        guard = FlorenceDeterminism(FlorenceRuntime(device="cpu"))

        with patch("torch.use_deterministic_algorithms", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                guard.enable()

        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
        self.assertFalse(guard.lock_acquired)

    def test_determinism_state_is_restored(self):
        previous = torch.are_deterministic_algorithms_enabled()
        torch.manual_seed(1234)
        rng_state = torch.random.get_rng_state().clone()
        runtime = FlorenceRuntime(device="cpu")
        guard = FlorenceDeterminism(runtime)

        guard.enable()
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        torch.rand(1)
        guard.restore()

        self.assertEqual(torch.are_deterministic_algorithms_enabled(), previous)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))


if __name__ == "__main__":
    unittest.main()

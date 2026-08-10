"""Reproducible runtime contract for the vendored Florence-2 model."""

from __future__ import annotations

import hashlib
import os
import random
import sys
import threading
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from packaging.version import Version

from ai_workers.core.resource_cleanup import get_available_memory_mb as get_system_available_memory_mb


FLORENCE_ASSET_SHA256 = {
    "model.safetensors": "03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060",
    "config.json": "c666d0fe0172d46e115e8fba6cd93cd83714575b33a73005cab8d24ce2a3aa8f",
    "configuration_florence2.py": "4b351d78012df1bcd0f21ce8b8cb3a82fa6f1790a65b2985d21d6e626f76b603",
    "modeling_florence2.py": "d0bd64290a87a46cbb9f4eccfa8c58227b1fd301b9039b1b687d413ead6effac",
    "preprocessor_config.json": "2f5921bbc53c7cc04251e1027b45b1cec726276be6db23d1bb40641bfbe2cf29",
    "processing_florence2.py": "f146023a507c009f425a49ee39aa037f4f25c64e14336e3e4f3f1d7377a68e98",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "tokenizer_config.json": "79ffcf43af8ebda99d165f61d243180da2e2639952e41e71e11611c18770489c",
    "vocab.json": "394fdc63c71aabe0a9b97117f5d62fb5fcc4d59b2b3ea929a3929e6a53217b3c",
}
FLORENCE_SEED = 0
FLORENCE_INFERENCE_LOCK = threading.Lock()
SUPPORTED_PYTHON = {(3, 10), (3, 11)}
SUPPORTED_PACKAGES = {
    "torch": "2.5.1",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
    "huggingface-hub": "0.36.2",
    "Pillow": "12.2.0",
    "numpy": "1.26.4",
    "einops": "0.8.2",
    "timm": "1.0.27",
    "torchvision": "0.20.1",
}
DEFAULT_MIN_AVAILABLE_MEMORY_MB = 6144
DEFAULT_MIN_AVAILABLE_VRAM_MB = 4096


class FlorenceRuntimeError(RuntimeError):
    """Raised when the runtime cannot satisfy the Florence reproducibility contract."""


class FlorenceResourceError(RuntimeError):
    """Raised when Florence inference would exceed local machine resource limits."""


@dataclass(frozen=True)
class FlorenceRuntime:
    device: str
    dtype: torch.dtype = torch.float32
    attention_implementation: str = "eager"


def validate_florence_environment() -> None:
    """Fail fast when a host uses an unverified Python or package version."""
    python_version = sys.version_info[:2]
    if python_version not in SUPPORTED_PYTHON:
        supported = ", ".join(f"{major}.{minor}" for major, minor in sorted(SUPPORTED_PYTHON))
        raise FlorenceRuntimeError(
            f"Florence-2 requires Python {supported}; found {sys.version.split()[0]}."
        )

    mismatches = []
    for package, expected in SUPPORTED_PACKAGES.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            mismatches.append(f"{package} is not installed (expected {expected})")
            continue

        if Version(installed).public != Version(expected).public:
            mismatches.append(f"{package}=={installed} (expected {expected}, local build suffix allowed)")

    if mismatches:
        details = "; ".join(mismatches)
        raise FlorenceRuntimeError(
            f"Unsupported Florence-2 runtime: {details}. "
            "Reinstall ai_workers/requirements.txt and restart the worker."
        )


def resolve_florence_runtime(requested_device: str) -> FlorenceRuntime:
    """Resolve Florence device; CPU is the reproducible and portable fallback."""
    validate_florence_environment()
    device = requested_device.strip().lower()
    if device not in {"cpu", "cuda"}:
        raise FlorenceRuntimeError(
            f"FLORENCE_DEVICE must be 'cpu' or 'cuda', received {requested_device!r}."
        )
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "[Florence Runtime] FLORENCE_DEVICE=cuda was requested, but CUDA is not available. "
            "Falling back to CPU so the worker can run on CPU-only hosts."
        )
        device = "cpu"
    if device == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return FlorenceRuntime(device=device)


def get_available_memory_mb() -> int | None:
    """Return available system memory in MB when the platform exposes it."""
    return get_system_available_memory_mb()


def assert_florence_memory_available(
    min_available_mb: int = DEFAULT_MIN_AVAILABLE_MEMORY_MB,
) -> int | None:
    """Fail before loading Florence when available RAM is below the configured floor."""
    available_mb = get_available_memory_mb()
    if available_mb is not None and available_mb < min_available_mb:
        raise FlorenceResourceError(
            f"Skipping Florence-2 captioning: only {available_mb} MB RAM available "
            f"(requires at least {min_available_mb} MB before model load)."
        )
    return available_mb


def get_cuda_memory_mb(device: str = "cuda") -> tuple[int, int] | None:
    """Return free and total CUDA memory in MB for the selected device."""
    if not torch.cuda.is_available():
        return None

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception:
        return None

    return int(free_bytes / (1024 * 1024)), int(total_bytes / (1024 * 1024))


def assert_florence_cuda_memory_available(
    runtime: FlorenceRuntime,
    min_available_vram_mb: int = DEFAULT_MIN_AVAILABLE_VRAM_MB,
) -> tuple[int, int] | None:
    """Fail before loading Florence on CUDA when free VRAM is below the configured floor."""
    if runtime.device != "cuda":
        return None

    torch.cuda.empty_cache()
    cuda_memory = get_cuda_memory_mb(runtime.device)
    if cuda_memory is None:
        return None

    free_mb, total_mb = cuda_memory
    if free_mb < min_available_vram_mb:
        raise FlorenceResourceError(
            f"Skipping Florence-2 CUDA captioning: only {free_mb}/{total_mb} MB VRAM free "
            f"(requires at least {min_available_vram_mb} MB before model load)."
        )
    return cuda_memory


def verify_florence_model(model_dir: str | Path) -> None:
    """Verify all executable, configuration, tokenizer, and weight assets."""
    model_root = Path(model_dir)
    for filename, expected in FLORENCE_ASSET_SHA256.items():
        asset_path = model_root / filename
        if not asset_path.is_file():
            action = "Run git lfs pull." if filename == "model.safetensors" else "Restore the vendored model checkout."
            raise FlorenceRuntimeError(
                f"Florence-2 asset is missing at {asset_path}. {action}"
            )

        stat = asset_path.stat()
        actual = _sha256_file(
            str(asset_path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
            normalize_newlines=filename != "model.safetensors",
        )
        if actual != expected:
            raise FlorenceRuntimeError(
                f"Florence-2 asset SHA-256 mismatch for {filename}. "
                f"Expected {expected}, found {actual}. Restore the verified checkout."
            )


@lru_cache(maxsize=32)
def _sha256_file(
    path: str,
    size: int,
    modified_ns: int,
    normalize_newlines: bool,
) -> str:
    del size, modified_ns
    if normalize_newlines:
        data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        return hashlib.sha256(data).hexdigest()

    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlorenceDeterminism:
    """Temporarily enable deterministic inference without leaking global state."""

    def __init__(self, runtime: FlorenceRuntime):
        self.runtime = runtime
        self.enabled = False
        self.lock_acquired = False

    def enable(self) -> None:
        FLORENCE_INFERENCE_LOCK.acquire()
        self.lock_acquired = True
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        self.deterministic_enabled = torch.are_deterministic_algorithms_enabled()
        self.warn_only_enabled = torch.is_deterministic_algorithms_warn_only_enabled()

        self.cuda_states = None
        if self.runtime.device == "cuda":
            self.cuda_states = torch.cuda.get_rng_state_all()
            self.cudnn_deterministic = torch.backends.cudnn.deterministic
            self.cudnn_benchmark = torch.backends.cudnn.benchmark
            self.matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            self.cudnn_tf32 = torch.backends.cudnn.allow_tf32

        self.enabled = True
        try:
            random.seed(FLORENCE_SEED)
            np.random.seed(FLORENCE_SEED)
            torch.manual_seed(FLORENCE_SEED)
            torch.use_deterministic_algorithms(True)

            if self.runtime.device == "cuda":
                torch.cuda.manual_seed_all(FLORENCE_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
        except BaseException:
            self.restore()
            raise

    def restore(self) -> None:
        if not self.enabled:
            if self.lock_acquired:
                self.lock_acquired = False
                FLORENCE_INFERENCE_LOCK.release()
            return

        try:
            if self.runtime.device == "cuda" and self.cuda_states is not None:
                torch.cuda.set_rng_state_all(self.cuda_states)
                torch.backends.cudnn.deterministic = self.cudnn_deterministic
                torch.backends.cudnn.benchmark = self.cudnn_benchmark
                torch.backends.cuda.matmul.allow_tf32 = self.matmul_tf32
                torch.backends.cudnn.allow_tf32 = self.cudnn_tf32

            torch.random.set_rng_state(self.torch_state)
            np.random.set_state(self.numpy_state)
            random.setstate(self.random_state)
            torch.use_deterministic_algorithms(
                self.deterministic_enabled,
                warn_only=self.warn_only_enabled,
            )
        finally:
            self.enabled = False
            if self.lock_acquired:
                self.lock_acquired = False
                FLORENCE_INFERENCE_LOCK.release()

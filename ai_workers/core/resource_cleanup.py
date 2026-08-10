"""Resource cleanup and memory guards for long-running AI workers."""

from __future__ import annotations

import gc
import os
import sys
import time


def _read_linux_mem_available_mb() -> int | None:
    if not sys.platform.startswith("linux"):
        return None

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(int(parts[1]) / 1024)
    except Exception:
        return None

    return None


def get_available_memory_mb() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        pass

    linux_available_mb = _read_linux_mem_available_mb()
    if linux_available_mb is not None:
        return linux_available_mb

    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys / (1024 * 1024))
        except Exception:
            return None

    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return int((pages * page_size) / (1024 * 1024))
    except Exception:
        return None

    return None


def get_cuda_memory_snapshot() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024)), int(total_bytes / (1024 * 1024))
    except Exception:
        return None


def release_worker_resources(label: str) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception as cleanup_err:
        print(f"[Cleanup] Resource cleanup warning after {label}: {cleanup_err}", flush=True)

    available_mb = get_available_memory_mb()
    cuda_memory = get_cuda_memory_snapshot()
    ram_text = f"RAM available={available_mb} MB" if available_mb is not None else "RAM available=unknown"
    vram_text = ""
    if cuda_memory is not None:
        free_mb, total_mb = cuda_memory
        vram_text = f", CUDA VRAM free={free_mb}/{total_mb} MB"
    print(f"[Cleanup] Completed after {label}: {ram_text}{vram_text}", flush=True)


def ensure_process_memory_available(
    min_available_mb: int,
    *,
    soft_min_available_mb: int,
    retry_seconds: int,
    retry_interval_seconds: int,
) -> int | None:
    deadline = time.monotonic() + max(0, retry_seconds)
    attempt = 0

    while True:
        attempt += 1
        release_worker_resources(f"preflight attempt {attempt}")
        available_mb = get_available_memory_mb()
        if available_mb is None or available_mb >= min_available_mb:
            return available_mb

        if time.monotonic() >= deadline:
            if available_mb >= soft_min_available_mb:
                print(
                    f"[Preflight] RAM available {available_mb} MB is below target "
                    f"{min_available_mb} MB but above soft floor {soft_min_available_mb} MB. Continuing.",
                    flush=True,
                )
                return available_mb
            raise RuntimeError(
                f"Insufficient RAM to start video processing: {available_mb} MB available "
                f"(requires at least {soft_min_available_mb} MB hard floor; target is {min_available_mb} MB). "
                "Close other apps or restart the worker."
            )

        wait_seconds = max(1, retry_interval_seconds)
        print(
            f"[Preflight] RAM available {available_mb} MB below target {min_available_mb} MB. "
            f"Waiting {wait_seconds}s before retry...",
            flush=True,
        )
        time.sleep(wait_seconds)

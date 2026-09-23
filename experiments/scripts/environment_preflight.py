"""Record real local environment and storage evidence for benchmark execution."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Sequence

from experiments.scripts.benchmark_artifacts import (
    parse_context,
    write_json_artifact,
    write_run_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "experiments" / "datasets"
GIB = 1024**3


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _verify_writable(path: Path) -> bool:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".preflight-write-probe"
    try:
        probe.write_text("write-probe\n", encoding="utf-8")
        return probe.read_text(encoding="utf-8") == "write-probe\n"
    finally:
        probe.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        context = parse_context(__doc__ or "", "00a_environment_preflight", argv)
        roots = {
            "project": PROJECT_ROOT,
            "data": DATA_ROOT,
            "artifact": context.namespace_root,
        }
        artifact_writable = _verify_writable(context.namespace_root)
        disk = {}
        for name, root in roots.items():
            usage = shutil.disk_usage(root)
            reserve_bytes = max(100 * GIB, int(usage.total * 0.15))
            disk[name] = {
                "root_key": name,
                "free_bytes": usage.free,
                "total_bytes": usage.total,
                "reserve_bytes": reserve_bytes,
                "reserve_gate_passed": usage.free >= reserve_bytes,
            }
        datasets = sorted(path.name for path in DATA_ROOT.iterdir() if path.is_dir())
        environment = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "project_root": ".",
            "data_root": "experiments/datasets",
            "cache_root": ".cache",
            "packages": {
                name: _package_version(name)
                for name in ("nbclient", "nbformat", "pyyaml", "numpy", "pandas")
            },
            "executables": {
                name: shutil.which(name) is not None for name in ("git", "ffmpeg", "ffprobe")
            },
            "local_datasets": datasets,
        }
        checks = {
            "data_root_exists": DATA_ROOT.is_dir(),
            "artifact_root_writable": artifact_writable,
            "disk": disk,
            "disk_gate_passed": all(item["reserve_gate_passed"] for item in disk.values()),
            "python_executable": Path(sys.executable).name,
        }
        if not all(
            (checks["data_root_exists"], checks["artifact_root_writable"], checks["disk_gate_passed"])
        ):
            raise RuntimeError(f"environment preflight failed: {checks}")
        write_json_artifact(context, "environment.profile", [environment])
        write_json_artifact(context, "environment.preflight_receipt", [checks])
        write_run_manifest(context)
        print(f"[OK] preflight: {len(datasets)} local dataset directories")
        return 0
    except Exception as exc:
        print(f"[FAIL] environment preflight: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Inventory local datasets and hash a deterministic real-data smoke subset."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Sequence

from experiments.scripts.benchmark_artifacts import (
    parse_context,
    read_json_artifact,
    sha256_file,
    write_json_artifact,
    write_run_manifest,
)
from experiments.scripts.tedlium_qualification import qualify_tedlium


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "experiments" / "datasets"
SMOKE_LIMIT = 20
SMOKE_MAX_FILE_BYTES = 256 * 1024**2
SMOKE_MAX_TOTAL_BYTES = 1024**3
ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".bz2", ".7z", ".rar"}


def _dataset_files(root: Path) -> Iterator[Path]:
    for directory, names, files in os.walk(root):
        names.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                yield path


def _readability(path: Path) -> tuple[bool, bool, str]:
    try:
        if path.suffix.lower() == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            return True, True, "json_parsed"
        with path.open("rb") as handle:
            handle.read(4096)
        return True, False, "bytes_read_not_decoded"
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, False, type(exc).__name__


def _smoke_files(root: Path, files: list[Path]) -> list[Path]:
    selected: list[Path] = []
    selected_bytes = 0
    for path in files:
        relative = path.relative_to(root)
        size = path.stat().st_size
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() in ARCHIVE_SUFFIXES or size > SMOKE_MAX_FILE_BYTES:
            continue
        if selected_bytes + size > SMOKE_MAX_TOTAL_BYTES:
            continue
        selected.append(path)
        selected_bytes += size
        if len(selected) == SMOKE_LIMIT:
            break
    return selected


def _inventory_dataset(root: Path, profile: str) -> dict:
    all_files = list(_dataset_files(root))
    selected = all_files if profile == "full" else _smoke_files(root, all_files)
    samples = []
    for path in selected:
        readable, decoded, probe_status = _readability(path)
        samples.append(
            {
                "unit_id": path.relative_to(root).as_posix(),
                "relative_path": path.relative_to(DATA_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "extension": path.suffix.lower() or "<none>",
                "sha256": sha256_file(path),
                "readable": readable,
                "decoded": decoded,
                "probe_status": probe_status,
            }
        )
    return {
        "dataset_id": root.name,
        "source": "local_filesystem",
        "acquisition_level": "D0_local_inventory" if profile == "smoke" else "D1_local_full_hash",
        "profile": profile,
        "root": f"experiments/datasets/{root.relative_to(DATA_ROOT).as_posix()}",
        "total_files": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "selected_files": len(samples),
        "readable_files": sum(bool(item["readable"]) for item in samples),
        "selection_policy": (
            "lexicographic_nonhidden_max20_max256MiB_each_max1GiB_total"
            if profile == "smoke"
            else "all_files"
        ),
        "files": samples,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        context = parse_context(__doc__ or "", "00b_dataset_feasibility", argv)
        read_json_artifact(context, "environment.profile")
        preflight = read_json_artifact(context, "environment.preflight_receipt")
        if not preflight["records"][0].get("disk_gate_passed"):
            raise RuntimeError("upstream disk gate did not pass")
        inventories = []
        for path in sorted(DATA_ROOT.iterdir()):
            if not path.is_dir() or not any(_dataset_files(path)):
                continue
            if context.profile == "full" and path.name == "tedlium3":
                inventory = qualify_tedlium(
                    path,
                    run_root=context.namespace_root,
                    run_id=context.run_id,
                    config_sha256=context.config_sha256,
                    code_sha256=context.source_sha256,
                )
                if inventory["status"] != "completed":
                    raise RuntimeError(
                        "TED-LIUM D1 pilot did not complete; resume from operational checkpoint"
                    )
                inventories.append(inventory)
            else:
                inventories.append(_inventory_dataset(path, context.profile))
        if not inventories:
            raise RuntimeError(f"no local datasets found under {DATA_ROOT}")
        feasibility = [
            {
                "dataset_id": item["dataset_id"],
                "acquisition_level": item["acquisition_level"],
                "selected_files": item["selected_files"],
                "readable_files": item["readable_files"],
                "readability_rate": (
                    item["readable_files"] / item["selected_files"]
                    if item["selected_files"]
                    else 0.0
                ),
                "eligible_for_profiling": (
                    item["selected_files"] > 0
                    and item["readable_files"] == item["selected_files"]
                    and (
                        item.get("dataset_id") != "tedlium3"
                        or (
                            item.get("status") == "completed"
                            and item.get("decode_gate_passed") is True
                        )
                    )
                ),
                "scientific_selection_eligible": False,
                "eligibility_reason": item.get(
                    "eligibility_reason",
                    "bounded local file probe; rights, task GT and media decode remain unadjudicated",
                ),
                "network_used": False,
                "decode_success_rate": item.get("decode_success_rate"),
                "qualification_status": item.get("status", "completed"),
            }
            for item in inventories
        ]
        if not any(item["eligible_for_profiling"] for item in feasibility):
            raise RuntimeError("no dataset passed the local readability probe")
        write_json_artifact(context, "dataset.feasibility_receipts", feasibility)
        write_json_artifact(context, "dataset.acquisition_inventory", inventories)
        write_run_manifest(context)
        print(f"[OK] feasibility: {len(inventories)} datasets, network_used=false")
        return 0
    except Exception as exc:
        print(f"[FAIL] dataset feasibility: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

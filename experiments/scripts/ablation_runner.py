"""Execute registry-defined ablations with explicit, fail-closed adapters.

Only RQ0_OD currently has a real adapter. Other tasks are rejected until their
benchmark entrypoints can consume and echo an effective per-combination config.
Dry runs are planning evidence only: they never contain metrics or pass status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.registry_loader import (  # noqa: E402
    AblationSpec,
    Candidate,
    TaskID,
    load_registry,
)


@dataclass(frozen=True)
class AblationCombo:
    combo_id: str
    base_id: str
    toggle_values: Dict[str, Any]
    base_kwargs: Dict[str, Any]
    merged_kwargs: Dict[str, Any]
    timestamp_iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class ComboResult:
    combo_id: str
    base_id: str
    toggle_values: Dict[str, Any]
    metrics: Dict[str, float]
    runtime_sec: float
    peak_vram_mb: float
    success: bool
    error_message: Optional[str] = None
    seed: int = 42
    evidence_status: str = "failed"
    command: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    config_hash: str = ""
    output_sha256: Optional[str] = None
    timestamp_iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


_OD_CANDIDATE_IDS = {
    "RQ0_OD.OD-A_YOLOv8n": "OD-A",
    "RQ0_OD.OD-B_YOLOv11n": "OD-B",
    "RQ0_OD.OD-C_RTDETR_r18": "OD-C",
}
_SUPPORTED_TOGGLES = {TaskID.RQ0_OD: {"conf_threshold", "imgsz"}}
_REQUIRED_OD_METRICS = {"map50", "precision", "recall"}
_REQUIRED_PROVENANCE = {
    "run_id",
    "config_hash",
    "dataset_id",
    "dataset_revision",
    "dataset_checksum",
    "ground_truth_revision",
    "model_revision",
    "runner_version",
    "raw_prediction_count",
    "frame_count",
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _make_combo_id(base_id: str, toggle_values: Dict[str, Any]) -> str:
    return hashlib.sha1(base_id.encode("utf-8") + b"|" + _json_bytes(toggle_values)).hexdigest()[:8]


def _compute_total_combos(spec: AblationSpec) -> int:
    total_per_base = math.prod(len(values) for values in spec.toggles.values())
    return total_per_base * len(spec.base_candidates)


def _stable_seed(seed: int, base_id: str) -> int:
    digest = hashlib.sha256(base_id.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:4], "big")


def generate_ablation_grid(
    spec: AblationSpec,
    candidates: Dict[TaskID, List[Candidate]],
    *,
    seed: int = 42,
    max_combinations_per_base: Optional[int] = None,
) -> List[AblationCombo]:
    cap = max_combinations_per_base or spec.max_combinations_per_base
    available = {candidate.id: candidate for candidate in candidates.get(spec.task, [])}
    toggle_keys = list(spec.toggles)
    value_grid = list(itertools.product(*(spec.toggles[key] for key in toggle_keys)))
    combos: List[AblationCombo] = []
    for base_id in spec.base_candidates:
        base = available.get(base_id)
        if base is None:
            raise ValueError(f"base candidate {base_id} not found in registry")
        selected = value_grid
        if len(value_grid) > cap:
            selected = random.Random(_stable_seed(seed, base_id)).sample(value_grid, cap)
        for values in selected:
            toggles = dict(zip(toggle_keys, values))
            combos.append(
                AblationCombo(
                    combo_id=_make_combo_id(base_id, toggles),
                    base_id=base_id,
                    toggle_values=toggles,
                    base_kwargs=base.model_kwargs,
                    merged_kwargs={**base.model_kwargs, **toggles},
                )
            )
    return combos


def _output_path(
    task: TaskID,
    combo: AblationCombo,
    seed: int,
    output_root: Path,
    attempt_id: Optional[str] = None,
) -> Path:
    suffix = f"-attempt-{attempt_id}" if attempt_id else ""
    return output_root / task.value / f"{combo.combo_id}-seed-{seed}{suffix}.json"


def build_command(
    combo: AblationCombo,
    task: TaskID,
    *,
    seed: int,
    output_root: Path,
    attempt_id: Optional[str] = None,
) -> tuple[List[str], Path, str]:
    supported = _SUPPORTED_TOGGLES.get(task)
    if supported is None:
        raise ValueError(f"task {task.value} has no real ablation adapter")
    unknown_toggles = set(combo.toggle_values) - supported
    missing_toggles = supported - set(combo.toggle_values)
    if unknown_toggles or missing_toggles:
        raise ValueError(
            f"task {task.value} toggle contract mismatch; missing={sorted(missing_toggles)}, "
            f"unknown={sorted(unknown_toggles)}"
        )
    candidate_id = _OD_CANDIDATE_IDS.get(combo.base_id)
    if candidate_id is None:
        raise ValueError(f"unsupported RQ0_OD candidate {combo.base_id}")
    conf_threshold = combo.toggle_values["conf_threshold"]
    imgsz = combo.toggle_values["imgsz"]
    if not isinstance(conf_threshold, (int, float)) or not 0 < float(conf_threshold) <= 1:
        raise ValueError("conf_threshold must be in (0, 1]")
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    output_path = _output_path(task, combo, seed, output_root, attempt_id)
    effective_config = {
        "task": task.value,
        "candidate_id": combo.base_id,
        "toggle_values": combo.toggle_values,
        "seed": seed,
    }
    command = [
        sys.executable,
        "-m",
        "benchmarks.scripts.run_object_detection_benchmark",
        "--models",
        candidate_id,
        "--frames-per-video",
        "8",
        "--imgsz",
        str(imgsz),
        "--conf-threshold",
        str(conf_threshold),
        "--seed",
        str(seed),
        "--candidate-id",
        combo.base_id,
        "--output-json",
        str(output_path),
        "--run-id",
        attempt_id or "plan-only",
        "--config-hash",
        _sha256_json(effective_config),
    ]
    return command, output_path, _sha256_json(effective_config)


def _failed_result(
    combo: AblationCombo,
    seed: int,
    message: str,
    *,
    runtime_sec: float = 0.0,
    command: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    config_hash: str = "",
) -> ComboResult:
    return ComboResult(
        combo_id=combo.combo_id,
        base_id=combo.base_id,
        toggle_values=combo.toggle_values,
        metrics={},
        runtime_sec=runtime_sec,
        peak_vram_mb=0.0,
        success=False,
        error_message=message,
        seed=seed,
        evidence_status="failed",
        command=command or [],
        output_path=str(output_path) if output_path else None,
        config_hash=config_hash,
    )


def _validate_output(
    output_path: Path,
    combo: AblationCombo,
    task: TaskID,
    seed: int,
    config_hash: str,
    run_id: str,
) -> tuple[Dict[str, float], float, str]:
    if not output_path.is_file():
        raise ValueError(f"expected output missing: {output_path}")
    raw = output_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid output JSON: {exc}") from exc
    expected_contract = {
        "task": task.value,
        "candidate_id": combo.base_id,
        "toggle_values": combo.toggle_values,
        "seed": seed,
    }
    if payload.get("ablation_contract") != expected_contract:
        raise ValueError("output ablation_contract does not match requested config")
    if payload.get("evidence_status") != "real":
        raise ValueError("output is not real benchmark evidence")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not _REQUIRED_PROVENANCE.issubset(provenance):
        raise ValueError("output provenance is incomplete")
    string_fields = _REQUIRED_PROVENANCE - {"raw_prediction_count", "frame_count"}
    if any(not isinstance(provenance[field], str) or not provenance[field] for field in string_fields):
        raise ValueError("output provenance strings must be non-empty")
    if provenance["run_id"] != run_id or provenance.get("config_hash") != config_hash:
        raise ValueError("output provenance is not bound to this run/config")
    checksum = provenance["dataset_checksum"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("output dataset checksum is invalid")
    for count_field in ("raw_prediction_count", "frame_count"):
        count = provenance[count_field]
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"output {count_field} must be positive")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, dict) or not raw_metrics:
        raise ValueError("output metrics must be non-empty")
    metrics: Dict[str, float] = {}
    for name, value in raw_metrics.items():
        if not isinstance(name, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"metric {name!r} is not numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"metric {name!r} is not finite")
        metrics[name] = numeric
    if task is TaskID.RQ0_OD and not _REQUIRED_OD_METRICS.issubset(metrics):
        raise ValueError("RQ0_OD output lacks map50/precision/recall")
    if task is TaskID.RQ0_OD:
        invalid_quality = {
            name: metrics[name] for name in _REQUIRED_OD_METRICS if not 0.0 <= metrics[name] <= 1.0
        }
        if invalid_quality:
            raise ValueError(f"RQ0_OD quality metrics must be in [0, 1]: {invalid_quality}")
    peak_vram = float(metrics.get("peak_vram_mb", 0.0))
    return metrics, peak_vram, hashlib.sha256(raw).hexdigest()


def execute_combo(
    combo: AblationCombo,
    task: TaskID,
    *,
    seed: int = 42,
    timeout_sec: int = 1800,
    dry_run: bool = False,
    output_root: Path = REPO_ROOT / "experiments" / "outputs" / "ablations",
) -> ComboResult:
    started = time.monotonic()
    attempt_id = None if dry_run else uuid.uuid4().hex
    try:
        command, output_path, config_hash = build_command(
            combo, task, seed=seed, output_root=output_root, attempt_id=attempt_id
        )
    except (TypeError, ValueError) as exc:
        return _failed_result(combo, seed, str(exc))

    if dry_run:
        return ComboResult(
            combo_id=combo.combo_id,
            base_id=combo.base_id,
            toggle_values=combo.toggle_values,
            metrics={},
            runtime_sec=0.0,
            peak_vram_mb=0.0,
            success=False,
            error_message=None,
            seed=seed,
            evidence_status="not_run",
            command=command,
            output_path=str(output_path),
            config_hash=config_hash,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.unlink(missing_ok=True)
    except OSError as exc:
        return _failed_result(
            combo, seed, f"cannot invalidate prior output: {exc}", command=command,
            output_path=output_path, config_hash=config_hash,
        )
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return _failed_result(
            combo, seed, f"timeout after {timeout_sec}s",
            runtime_sec=time.monotonic() - started, command=command,
            output_path=output_path, config_hash=config_hash,
        )
    except OSError as exc:
        return _failed_result(
            combo, seed, f"process launch failed: {exc}",
            runtime_sec=time.monotonic() - started, command=command,
            output_path=output_path, config_hash=config_hash,
        )

    runtime = time.monotonic() - started
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "non-zero exit")[-500:]
        return _failed_result(
            combo, seed, detail, runtime_sec=runtime, command=command,
            output_path=output_path, config_hash=config_hash,
        )
    try:
        metrics, peak_vram, output_sha256 = _validate_output(
            output_path, combo, task, seed, config_hash, attempt_id or "plan-only"
        )
    except (OSError, ValueError) as exc:
        return _failed_result(
            combo, seed, str(exc), runtime_sec=runtime, command=command,
            output_path=output_path, config_hash=config_hash,
        )
    return ComboResult(
        combo_id=combo.combo_id,
        base_id=combo.base_id,
        toggle_values=combo.toggle_values,
        metrics=metrics,
        runtime_sec=runtime,
        peak_vram_mb=peak_vram,
        success=True,
        seed=seed,
        evidence_status="real",
        command=command,
        output_path=str(output_path),
        config_hash=config_hash,
        output_sha256=output_sha256,
    )


def execute_grid(
    combos: List[AblationCombo],
    task: TaskID,
    *,
    seed: int = 42,
    timeout_sec: int = 1800,
    dry_run: bool = False,
    parallel: bool = False,
    max_workers: int = 4,
    output_root: Path = REPO_ROOT / "experiments" / "outputs" / "ablations",
) -> List[ComboResult]:
    if parallel and not dry_run:
        return [
            _failed_result(combo, seed, "parallel execution is unsafe until frame inputs are immutable")
            for combo in combos
        ]
    return [
        execute_combo(
            combo, task, seed=seed, timeout_sec=timeout_sec,
            dry_run=dry_run, output_root=output_root,
        )
        for combo in combos
    ]


def log_combo_audit(
    combos: List[AblationCombo],
    results: List[ComboResult],
    audit_path: Path,
) -> None:
    """Append exactly one audit row per result; combos is retained for API compatibility."""
    del combos
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    has_rows = audit_path.exists() and audit_path.stat().st_size > 0
    with audit_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if not has_rows:
            writer.writerow(
                [
                    "combo_id", "base_id", "toggle_values_json", "seed",
                    "evidence_status", "runtime_sec", "peak_vram_mb",
                    "metrics_json", "config_hash", "command_json", "output_path",
                    "output_sha256", "error", "timestamp",
                ]
            )
        for result in results:
            writer.writerow(
                [
                    result.combo_id,
                    result.base_id,
                    json.dumps(result.toggle_values, sort_keys=True),
                    result.seed,
                    result.evidence_status,
                    f"{result.runtime_sec:.3f}",
                    f"{result.peak_vram_mb:.1f}",
                    json.dumps(result.metrics, sort_keys=True),
                    result.config_hash,
                    json.dumps(result.command),
                    result.output_path or "",
                    result.output_sha256 or "",
                    result.error_message or "",
                    result.timestamp_iso,
                ]
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Task ID, currently only RQ0_OD")
    parser.add_argument("--seeds", default="42", help="Comma-separated integer seeds")
    parser.add_argument("--max-per-base", type=int, default=None)
    parser.add_argument(
        "--audit-path", type=Path,
        default=Path("experiments/scratch/ablation_audit.csv"),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("experiments/outputs/ablations"),
    )
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args(argv)

    try:
        task = TaskID(args.task)
        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        if not seeds:
            raise ValueError("at least one seed is required")
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seeds are not independent replicates")
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1

    candidates, specs, _ = load_registry()
    spec = specs.get(task)
    if spec is None:
        print(f"[FAIL] no ablation spec for {task.value}")
        return 1
    if task is TaskID.RQ0_OD and not args.dry_run and len(seeds) > 1:
        print("[FAIL] RQ0_OD is deterministic; real mode accepts one seed until stochastic sources are defined")
        return 1
    try:
        combos = generate_ablation_grid(
            spec, candidates, seed=42, max_combinations_per_base=args.max_per_base
        )
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1
    if not combos:
        print("[FAIL] generated no ablation combinations")
        return 1

    all_results: List[ComboResult] = []
    for seed in seeds:
        all_results.extend(
            execute_grid(
                combos,
                task,
                seed=seed,
                timeout_sec=args.timeout_sec,
                dry_run=args.dry_run,
                parallel=args.parallel,
                max_workers=args.max_workers,
                output_root=args.output_root,
            )
        )
    log_combo_audit(combos, all_results, args.audit_path)
    status_counts = {
        status: sum(result.evidence_status == status for result in all_results)
        for status in ("real", "not_run", "failed")
    }
    print(f"[OK] planned={len(all_results)} status={status_counts} audit={args.audit_path}")
    if args.dry_run:
        return 0 if status_counts["not_run"] == len(all_results) else 1
    return 0 if status_counts["real"] == len(all_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

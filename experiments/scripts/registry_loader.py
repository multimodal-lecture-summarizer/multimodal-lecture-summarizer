"""registry_loader.py — Load YAML candidates/ablation/weights → typed dataclass.

Usage:
    from experiments.scripts.registry_loader import load_registry

    candidates, ablation_specs, weights = load_registry()

CLI:
    python experiments/scripts/registry_loader.py --validate
    python experiments/scripts/registry_loader.py --task RQ1
    python experiments/scripts/registry_loader.py --list-tasks
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add parent to allow `python experiments/scripts/registry_loader.py` direct
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.registry import REGISTRY_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskID(str, Enum):
    RQ0_A = "RQ0_A"
    RQ0_V = "RQ0_V"
    RQ0_OD = "RQ0_OD"
    RQ1 = "RQ1"
    RQ2 = "RQ2"
    RQ3 = "RQ3"
    RQ4 = "RQ4"


# ---------------------------------------------------------------------------
# Dataclasses (typed schema)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComputeBudget:
    """VRAM + latency caps. 0 = unlimited."""
    max_vram_mb: int = 0
    max_latency_ms: int = 0


@dataclass(frozen=True)
class Candidate:
    """1 model candidate từ registry."""
    id: str
    task: TaskID
    display_name: str
    model_class: str
    model_kwargs: Dict[str, Any]
    required_features: List[str]
    compute_budget: ComputeBudget
    family: str
    reference_paper: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AblationSpec:
    """2^k toggle grid cho 1 task."""
    task: TaskID
    base_candidates: List[str]
    toggles: Dict[str, List[Any]]
    max_combinations_per_base: int
    description: str = ""


@dataclass(frozen=True)
class LeaderboardWeights:
    """Multi-metric weights + directions per task."""
    task: TaskID
    metrics: Dict[str, float]
    directions: Dict[str, str]
    note: str = ""


# ---------------------------------------------------------------------------
# YAML loaders (with validation)
# ---------------------------------------------------------------------------

def _validate_task(task_str: str) -> TaskID:
    try:
        return TaskID(task_str)
    except ValueError as e:
        valid = ", ".join(t.value for t in TaskID)
        raise ValueError(f"Unknown task {task_str!r}. Valid: {valid}") from e


def _parse_compute_budget(raw: Optional[Dict[str, Any]]) -> ComputeBudget:
    if raw is None:
        return ComputeBudget()
    return ComputeBudget(
        max_vram_mb=int(raw.get("max_vram_mb", 0)),
        max_latency_ms=int(raw.get("max_latency_ms", 0)),
    )


def _parse_candidate(raw: Dict[str, Any]) -> Candidate:
    return Candidate(
        id=str(raw["id"]),
        task=_validate_task(raw["task"]),
        display_name=str(raw["display_name"]),
        model_class=str(raw["model_class"]),
        model_kwargs=dict(raw.get("model_kwargs", {})),
        required_features=list(raw.get("required_features", [])),
        compute_budget=_parse_compute_budget(raw.get("compute_budget")),
        family=str(raw.get("family", "unknown")),
        reference_paper=raw.get("reference_paper"),
        tags=list(raw.get("tags", [])),
    )


def _parse_ablation(raw: Dict[str, Any], task: TaskID) -> AblationSpec:
    return AblationSpec(
        task=task,
        base_candidates=list(raw["base_candidates"]),
        toggles={k: list(vals) for k, vals in raw["toggles"].items()},
        max_combinations_per_base=int(raw.get("max_combinations_per_base", 8)),
        description=str(raw.get("description", "")),
    )


def _parse_weights(raw: Dict[str, Any], task: TaskID) -> LeaderboardWeights:
    return LeaderboardWeights(
        task=task,
        metrics=dict(raw["metrics"]),
        directions=dict(raw["directions"]),
        note=str(raw.get("note", "")),
    )


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file with error context."""
    if not path.exists():
        raise FileNotFoundError(f"YAML not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error in {path.name}: {e}") from e


# ---------------------------------------------------------------------------
# Main load function
# ---------------------------------------------------------------------------

def load_registry(
    registry_dir: Path = REGISTRY_DIR,
) -> Tuple[Dict[TaskID, List[Candidate]], Dict[TaskID, AblationSpec], Dict[TaskID, LeaderboardWeights]]:
    """
    Load all 3 YAMLs and return typed dataclasses.
    Returns:
        candidates: {TaskID: [Candidate, ...]} per task
        ablation_specs: {TaskID: AblationSpec} per task
        weights: {TaskID: LeaderboardWeights} per task
    """
    candidates_raw = load_yaml(registry_dir / "candidates.yaml")
    ablation_raw = load_yaml(registry_dir / "ablation_specs.yaml")
    weights_raw = load_yaml(registry_dir / "leaderboard_weights.yaml")

    # Parse candidates — group by task
    candidates: Dict[TaskID, List[Candidate]] = {t: [] for t in TaskID}
    for raw_c in candidates_raw.get("candidates", []):
        cand = _parse_candidate(raw_c)
        candidates[cand.task].append(cand)

    # Parse ablation specs — keyed by task in YAML
    ablation_specs: Dict[TaskID, AblationSpec] = {}
    for task_str, raw_spec in (ablation_raw.get("ablation_specs", {})).items():
        task = _validate_task(task_str)
        ablation_specs[task] = _parse_ablation(raw_spec, task)

    # Parse weights — keyed by task
    weights: Dict[TaskID, LeaderboardWeights] = {}
    for task_str, raw_w in (weights_raw.get("leaderboard_weights", {})).items():
        task = _validate_task(task_str)
        weights[task] = _parse_weights(raw_w, task)

    return candidates, ablation_specs, weights


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_registry(
    registry_dir: Path = REGISTRY_DIR,
) -> List[str]:
    """
    Check schema + referential integrity across 3 YAMLs.
    Returns list of error messages (empty = valid).
    """
    errors: List[str] = []

    # 1. Load all
    try:
        candidates, ablation_specs, weights = load_registry(registry_dir)
    except Exception as e:
        return [f"Load failed: {e}"]

    # 2. Check candidates per task
    candidate_ids_by_task: Dict[TaskID, set] = {t: set() for t in TaskID}
    for task, cands in candidates.items():
        if not cands:
            errors.append(f"No candidates defined for task {task.value}")
        for c in cands:
            if c.id in candidate_ids_by_task[task]:
                errors.append(f"Duplicate candidate ID: {c.id}")
            candidate_ids_by_task[task].add(c.id)
            # Check fields
            if not c.model_class:
                errors.append(f"{c.id}: model_class is empty")
            if not c.required_features:
                errors.append(f"{c.id}: required_features is empty")
            if c.compute_budget.max_vram_mb < 0:
                errors.append(f"{c.id}: max_vram_mb must be >= 0")
            if c.compute_budget.max_latency_ms < 0:
                errors.append(f"{c.id}: max_latency_ms must be >= 0")

    # 3. Check ablation specs reference valid base candidates
    for task, spec in ablation_specs.items():
        available = candidate_ids_by_task.get(task, set())
        for base_id in spec.base_candidates:
            if base_id not in available:
                errors.append(
                    f"ablation_specs[{task.value}]: base_candidate '{base_id}' "
                    f"not found in candidates.yaml"
                )
        # Check cap value
        if spec.max_combinations_per_base < 1:
            errors.append(
                f"ablation_specs[{task.value}]: max_combinations_per_base must be >= 1"
            )
        # Check toggles not empty
        if not spec.toggles:
            errors.append(
                f"ablation_specs[{task.value}]: toggles dict is empty"
            )

    # 4. Check weights sum to ~1.0
    for task, w in weights.items():
        total = sum(w.metrics.values())
        if abs(total - 1.0) > 0.01:
            errors.append(
                f"leaderboard_weights[{task.value}]: weights sum to {total:.4f} "
                f"(expected 1.0 ± 0.01)"
            )
        # Check directions match metric keys
        for metric in w.metrics:
            if metric not in w.directions:
                errors.append(
                    f"leaderboard_weights[{task.value}]: metric '{metric}' "
                    f"missing direction"
                )
        # Check directions valid values
        valid_directions = {"higher", "lower", "binary"}
        for metric, direction in w.directions.items():
            if direction not in valid_directions:
                errors.append(
                    f"leaderboard_weights[{task.value}]: metric '{metric}' "
                    f"has invalid direction '{direction}' (must be one of {valid_directions})"
                )

    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true",
                        help="Validate registry schema + cross-references (exit 0 if OK)")
    parser.add_argument("--task", type=str, default=None,
                        help="Show candidates for one task (e.g., RQ1)")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all tasks + candidate counts")
    parser.add_argument("--registry-dir", type=Path, default=REGISTRY_DIR,
                        help="Path to registry directory")
    args = parser.parse_args(argv)

    if args.validate:
        errors = validate_registry(args.registry_dir)
        if errors:
            print(f"[FAIL] {len(errors)} validation error(s):")
            for e in errors:
                print(f"  - {e}")
            return 1
        print("[OK] Registry valid")
        return 0

    candidates, ablation_specs, weights = load_registry(args.registry_dir)

    if args.list_tasks:
        print("=" * 60)
        print("REGISTRY SUMMARY")
        print("=" * 60)
        for task in TaskID:
            n = len(candidates.get(task, []))
            has_ablation = task in ablation_specs
            has_weights = task in weights
            print(
                f"  {task.value:8s}  candidates={n:3d}  "
                f"ablation={'Y' if has_ablation else 'N'}  "
                f"weights={'Y' if has_weights else 'N'}"
            )
        total = sum(len(c) for c in candidates.values())
        print(f"\n  TOTAL: {total} candidates")
        return 0

    if args.task:
        try:
            task = _validate_task(args.task)
        except ValueError as e:
            print(f"[FAIL] {e}")
            return 1
        cands = candidates.get(task, [])
        print(f"\nTASK: {task.value} ({len(cands)} candidates)\n")
        for c in cands:
            print(f"  [{c.id}]")
            print(f"    Display:  {c.display_name}")
            print(f"    Class:    {c.model_class}")
            print(f"    Family:   {c.family}")
            print(f"    Budget:   VRAM<={c.compute_budget.max_vram_mb}MB, "
                  f"latency<={c.compute_budget.max_latency_ms}ms")
            print(f"    Features: {c.required_features}")
            print(f"    Tags:     {c.tags}")
            print()
        return 0

    # No flags: print usage
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
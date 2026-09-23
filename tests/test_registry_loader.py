"""Unit tests for registry_loader.py — Phase 7 verification."""
import pytest
from pathlib import Path

from experiments.scripts.registry_loader import (
    load_registry,
    validate_registry,
    TaskID,
    Candidate,
    ComputeBudget,
    AblationSpec,
    LeaderboardWeights,
)


def test_load_registry_returns_dataclass_objects():
    """Registry load returns typed dataclasses, not raw dicts."""
    candidates, specs, weights = load_registry()
    # All 7 tasks should be present
    for task in TaskID:
        assert task in candidates, f"Missing task {task.value}"
        assert task in specs, f"Missing ablation spec for {task.value}"
        assert task in weights, f"Missing weights for {task.value}"


def test_candidate_count_meets_plan_target():
    """Plan target: ≥30 candidates. Actual should be ≥35."""
    candidates, _, _ = load_registry()
    total = sum(len(c) for c in candidates.values())
    assert total >= 30, f"Expected ≥30 candidates, got {total}"


def test_all_candidates_have_required_fields():
    """Every candidate has id, task, model_class, required_features, compute_budget."""
    candidates, _, _ = load_registry()
    for task, cands in candidates.items():
        for c in cands:
            assert c.id, f"Empty ID in {task.value}"
            assert c.task == task, f"Task mismatch for {c.id}"
            assert c.model_class, f"Empty model_class for {c.id}"
            assert c.required_features, f"Empty required_features for {c.id}"
            assert isinstance(c.compute_budget, ComputeBudget)


def test_validate_registry_passes():
    """Default registry is valid."""
    errors = validate_registry()
    assert errors == [], f"Validation errors: {errors}"


def test_validate_detects_missing_candidate_in_ablation():
    """If ablation_spec references unknown candidate, validate catches it."""
    # Create a temp registry with bad cross-reference
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # Copy valid registry then modify
        src = Path("experiments/registry")
        for f in ["candidates.yaml", "ablation_specs.yaml", "leaderboard_weights.yaml"]:
            shutil.copy(src / f, tmpdir / f)
        # Modify ablation_specs to reference missing candidate
        ab_path = tmpdir / "ablation_specs.yaml"
        ab = ab_path.read_text(encoding="utf-8")
        ab_bad = ab.replace("RQ1.C5_CrossAttentionProposed", "RQ1.FAKE_CANDIDATE_999")
        ab_path.write_text(ab_bad, encoding="utf-8")

        errors = validate_registry(tmpdir)
        assert any("FAKE_CANDIDATE_999" in e for e in errors), \
            f"Expected error for fake candidate, got: {errors}"


def test_validate_detects_invalid_direction():
    """Direction not in {higher, lower, binary} triggers error."""
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        src = Path("experiments/registry")
        for f in ["candidates.yaml", "ablation_specs.yaml", "leaderboard_weights.yaml"]:
            shutil.copy(src / f, tmpdir / f)
        lw_path = tmpdir / "leaderboard_weights.yaml"
        lw = lw_path.read_text(encoding="utf-8")
        lw_bad = lw.replace("directions:\n      wer: lower", "directions:\n      wer: sideways")
        lw_path.write_text(lw_bad, encoding="utf-8")

        errors = validate_registry(tmpdir)
        assert any("sideways" in e for e in errors), \
            f"Expected error for invalid direction, got: {errors}"
        assert any("invalid direction" in e for e in errors), \
            f"Expected explicit 'invalid direction' message"


def test_validate_detects_weights_not_summing_to_one():
    """If weights don't sum to ~1.0, validate catches it."""
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        src = Path("experiments/registry")
        for f in ["candidates.yaml", "ablation_specs.yaml", "leaderboard_weights.yaml"]:
            shutil.copy(src / f, tmpdir / f)
        lw_path = tmpdir / "leaderboard_weights.yaml"
        lw = lw_path.read_text(encoding="utf-8")
        # Change wer weight from 0.40 to 0.90 (totals now 1.4)
        lw_bad = lw.replace("wer: 0.40", "wer: 0.90")
        lw_path.write_text(lw_bad, encoding="utf-8")

        errors = validate_registry(tmpdir)
        assert any("weights sum" in e for e in errors), \
            f"Expected weights sum error, got: {errors}"


def test_task_id_enum_complete():
    """All TaskID values are valid strings."""
    expected = {"RQ0_A", "RQ0_V", "RQ0_OD", "RQ1", "RQ2", "RQ3", "RQ4"}
    actual = {t.value for t in TaskID}
    assert actual == expected, f"TaskID mismatch: missing {expected - actual}, extra {actual - expected}"


def test_candidate_class_immutable():
    """Candidate is frozen dataclass — can't mutate fields."""
    candidates, _, _ = load_registry()
    c = candidates[TaskID.RQ1][0]
    with pytest.raises(Exception):  # FrozenInstanceError
        c.display_name = "modified"
"""Contract tests for the fail-closed ablation runner."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import experiments.scripts.ablation_runner as ablation_runner

from experiments.scripts.ablation_runner import (
    AblationCombo,
    _compute_total_combos,
    _make_combo_id,
    _stable_seed,
    build_command,
    execute_combo,
    execute_grid,
    generate_ablation_grid,
    log_combo_audit,
    main,
)
from experiments.scripts.registry_loader import AblationSpec, TaskID, load_registry


def _combo(**toggles: object) -> AblationCombo:
    values = {"conf_threshold": 0.25, "imgsz": 640, **toggles}
    return AblationCombo(
        combo_id=_make_combo_id("RQ0_OD.OD-A_YOLOv8n", values),
        base_id="RQ0_OD.OD-A_YOLOv8n",
        toggle_values=values,
        base_kwargs={},
        merged_kwargs=values,
    )


def test_combo_id_is_order_independent_and_value_sensitive() -> None:
    left = _make_combo_id("RQ1.C5", {"visual_snapping": True, "pos_weight": 4.0})
    reordered = _make_combo_id("RQ1.C5", {"pos_weight": 4.0, "visual_snapping": True})
    changed = _make_combo_id("RQ1.C5", {"pos_weight": 8.0, "visual_snapping": True})
    assert left == reordered
    assert left != changed


def test_compute_total_combos() -> None:
    spec = AblationSpec(
        task=TaskID.RQ1,
        base_candidates=["A", "B", "C"],
        toggles={"a": [1, 2], "b": [1, 2], "c": [1, 2, 3]},
        max_combinations_per_base=8,
        description="test",
    )
    assert _compute_total_combos(spec) == 36


def test_grid_sampling_is_stable_and_capped() -> None:
    candidates, specs, _ = load_registry()
    spec = specs[TaskID.RQ1]
    first = generate_ablation_grid(spec, candidates, seed=42, max_combinations_per_base=2)
    second = generate_ablation_grid(spec, candidates, seed=42, max_combinations_per_base=2)
    assert [combo.combo_id for combo in first] == [combo.combo_id for combo in second]
    assert all(sum(item.base_id == combo.base_id for item in first) <= 2 for combo in first)
    expected = 42 ^ int.from_bytes(hashlib.sha256(b"RQ1.C5").digest()[:4], "big")
    assert _stable_seed(42, "RQ1.C5") == expected


def test_unknown_base_candidate_fails_closed() -> None:
    spec = AblationSpec(
        task=TaskID.RQ1,
        base_candidates=["RQ1.FAKE_NOT_REAL"],
        toggles={"a": [True, False]},
        max_combinations_per_base=8,
        description="test",
    )
    candidates, _, _ = load_registry()
    with pytest.raises(ValueError, match="FAKE_NOT_REAL"):
        generate_ablation_grid(spec, candidates)


def test_build_command_propagates_effective_config(tmp_path: Path) -> None:
    combo = _combo(conf_threshold=0.35, imgsz=416)
    command, output_path, config_hash = build_command(
        combo, TaskID.RQ0_OD, seed=1337, output_root=tmp_path
    )
    assert command[:3] == [
        sys.executable,
        "-m",
        "benchmarks.scripts.run_object_detection_benchmark",
    ]
    assert command[command.index("--models") + 1] == "OD-A"
    assert command[command.index("--conf-threshold") + 1] == "0.35"
    assert command[command.index("--imgsz") + 1] == "416"
    assert command[command.index("--seed") + 1] == "1337"
    assert str(output_path) == command[command.index("--output-json") + 1]
    assert len(config_hash) == 64


def test_unsupported_task_fails_closed(tmp_path: Path) -> None:
    result = execute_combo(_combo(), TaskID.RQ1, dry_run=True, output_root=tmp_path)
    assert not result.success
    assert result.evidence_status == "failed"
    assert "no real ablation adapter" in (result.error_message or "")


def test_dry_run_is_plan_only_without_metrics(tmp_path: Path) -> None:
    result = execute_combo(_combo(), TaskID.RQ0_OD, dry_run=True, output_root=tmp_path)
    assert not result.success
    assert result.evidence_status == "not_run"
    assert result.metrics == {}
    assert result.command
    assert result.output_path


def _write_valid_output(command: list[str], *, metrics: dict | None = None) -> None:
    def arg(name: str) -> str:
        return command[command.index(name) + 1]

    output_path = Path(arg("--output-json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "ablation_contract": {
                    "task": "RQ0_OD",
                    "candidate_id": arg("--candidate-id"),
                    "toggle_values": {
                        "conf_threshold": float(arg("--conf-threshold")),
                        "imgsz": int(arg("--imgsz")),
                    },
                    "seed": int(arg("--seed")),
                },
                "evidence_status": "real",
                "provenance": {
                    "run_id": arg("--run-id"),
                    "config_hash": arg("--config-hash"),
                    "dataset_id": "released-od-set",
                    "dataset_revision": "v1",
                    "dataset_checksum": "c" * 64,
                    "ground_truth_revision": "labels-v1",
                    "model_revision": "weights-v1",
                    "runner_version": "rq0_od_v2",
                    "raw_prediction_count": 25,
                    "frame_count": 10,
                },
                "metrics": {
                    "map50": 0.51,
                    "precision": 0.62,
                    "recall": 0.58,
                    "fps": 31.5,
                    "peak_vram_mb": 512.0,
                } if metrics is None else metrics,
            }
        ),
        encoding="utf-8",
    )


def test_real_execution_requires_matching_nonempty_finite_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _write_valid_output(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = execute_combo(_combo(), TaskID.RQ0_OD, seed=7, output_root=tmp_path)
    assert result.success
    assert result.evidence_status == "real"
    assert result.metrics["fps"] == 31.5
    assert result.output_sha256 and len(result.output_sha256) == 64


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({}, "non-empty"),
        ({"fps": float("nan")}, "not finite"),
        ({"map50": -1.0, "precision": 2.0, "recall": 0.5}, "must be in [0, 1]"),
    ],
)
def test_invalid_metrics_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics: dict,
    expected: str,
) -> None:
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _write_valid_output(command, metrics=metrics)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path)
    assert not result.success
    assert expected in (result.error_message or "")


def test_diagnostic_or_incomplete_provenance_cannot_be_real(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def diagnostic_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _write_valid_output(command)
        path = Path(command[command.index("--output-json") + 1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["evidence_status"] = "diagnostic"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", diagnostic_run)
    result = execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path)
    assert not result.success
    assert "not real benchmark evidence" in (result.error_message or "")

    def incomplete_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        _write_valid_output(command)
        path = Path(command[command.index("--output-json") + 1])
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["provenance"]["ground_truth_revision"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", incomplete_run)
    result = execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path)
    assert not result.success
    assert "provenance is incomplete" in (result.error_message or "")


def test_missing_output_nonzero_timeout_and_oserror_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    assert "expected output missing" in (
        execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path).error_message or ""
    )

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, stdout="", stderr="bad"),
    )
    assert "bad" in (execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path).error_message or "")

    def timeout(*_: object, **__: object) -> None:
        raise subprocess.TimeoutExpired("cmd", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    assert "timeout" in (execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path).error_message or "")

    def os_error(*_: object, **__: object) -> None:
        raise OSError("cannot launch")

    monkeypatch.setattr(subprocess, "run", os_error)
    assert "cannot launch" in (
        execute_combo(_combo(), TaskID.RQ0_OD, output_root=tmp_path).error_message or ""
    )


def test_stale_output_is_invalidated_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    combo = _combo()
    command, output_path, _ = build_command(
        combo, TaskID.RQ0_OD, seed=42, output_root=tmp_path, attempt_id="fixed"
    )
    _write_valid_output(command)
    assert output_path.exists()
    monkeypatch.setattr(
        ablation_runner.uuid,
        "uuid4",
        lambda: type("FixedUUID", (), {"hex": "fixed"})(),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    result = execute_combo(combo, TaskID.RQ0_OD, output_root=tmp_path)
    assert not result.success
    assert "expected output missing" in (result.error_message or "")
    assert not output_path.exists()


def test_parallel_real_execution_fails_closed(tmp_path: Path) -> None:
    results = execute_grid([_combo()], TaskID.RQ0_OD, parallel=True, output_root=tmp_path)
    assert results[0].evidence_status == "failed"
    assert "parallel execution is unsafe" in (results[0].error_message or "")


def test_audit_keeps_every_seed_result(tmp_path: Path) -> None:
    combos = [_combo(), _combo(conf_threshold=0.35)]
    results = []
    for seed in (42, 1337):
        results.extend(execute_grid(combos, TaskID.RQ0_OD, seed=seed, dry_run=True, output_root=tmp_path))
    audit_path = tmp_path / "audit.csv"
    log_combo_audit(combos, results, audit_path)
    with audit_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert {row["seed"] for row in rows} == {"42", "1337"}
    assert {row["evidence_status"] for row in rows} == {"not_run"}
    assert all(json.loads(row["metrics_json"]) == {} for row in rows)


def test_cli_requires_all_real_runs_or_all_plans(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert main(
        [
            "--task", "RQ0_OD", "--max-per-base", "1", "--seeds", "42,1337",
            "--dry-run", "--audit-path", str(tmp_path / "dry.csv"),
            "--output-root", str(tmp_path / "outputs"),
        ]
    ) == 0
    assert main(
        [
            "--task", "RQ1", "--max-per-base", "1", "--dry-run",
            "--audit-path", str(tmp_path / "unsupported.csv"),
        ]
    ) == 1
    assert main(
        [
            "--task", "RQ0_OD", "--seeds", "42,42", "--dry-run",
            "--audit-path", str(tmp_path / "duplicates.csv"),
        ]
    ) == 1

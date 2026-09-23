from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import nbformat

from benchmarks.scripts.execute_all_notebooks import (
    _entry_config_sha256,
    _fallback_only_errors,
    _sha256,
    _scientifically_empty_errors,
    _validate_payload,
    build_plan,
    can_resume,
    load_manifest,
    main as executor_main,
    select_entries,
    validate_produced_artifacts,
    validate_required_artifacts,
)
from experiments.scripts import environment_preflight, probe_dataset_feasibility, profile_dataset
from experiments.scripts.benchmark_artifacts import (
    RunnerContext,
    write_json_artifact,
    write_run_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _runner_args(artifact_root: Path, notebook_id: str, run_id: str) -> list[str]:
    return [
        "--profile",
        "smoke",
        "--run-namespace",
        "official",
        "--evidence-status",
        "real",
        "--artifact-root",
        str(artifact_root),
        "--notebook-id",
        notebook_id,
        "--run-id",
        run_id,
        "--source-sha256",
        HASH_A,
        "--config-sha256",
        HASH_B,
        "--dependency-state-sha256",
        HASH_C,
    ]


def _context(run_id: str) -> dict[str, str]:
    return {
        "run_id": run_id,
        "status": "real",
        "source_sha256": HASH_A,
        "config_sha256": HASH_B,
        "dependency_state_sha256": HASH_C,
    }


def test_direct_and_module_executor_entrypoints_match() -> None:
    direct = subprocess.run(
        [sys.executable, "benchmarks/scripts/execute_all_notebooks.py", "--list", "--profile", "smoke"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    module = subprocess.run(
        [sys.executable, "-m", "benchmarks.scripts.execute_all_notebooks", "--list", "--profile", "smoke"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert direct.returncode == module.returncode == 0
    assert direct.stdout == module.stdout
    assert len(direct.stdout.strip().splitlines()) == 22


def test_generator_entrypoints_match_repository_notebooks() -> None:
    commands = (
        [sys.executable, "experiments/scripts/generate_refactored_notebook.py", "--check"],
        [sys.executable, "-m", "experiments.scripts.generate_refactored_notebook", "--check"],
    )
    results = [
        subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        for command in commands
    ]
    assert all(result.returncode == 0 for result in results)
    assert results[0].stdout == results[1].stdout
    assert "[OK] 0 notebook(s) would change" in results[0].stdout


def test_smoke_dry_plan_has_order_and_bounded_resource_policies() -> None:
    manifest = load_manifest()
    group_02 = build_plan(
        select_entries(manifest, group="02", profile="smoke"),
        "smoke",
        manifest["notebooks"],
    )
    group_07 = build_plan(
        select_entries(manifest, group="07", profile="smoke"),
        "smoke",
        manifest["notebooks"],
    )
    assert [item["id"] for item in group_02] == [
        "02a_scene_boundary_benchmark",
        "02b_keyframe_selection_benchmark",
        "02c_ocr_benchmark",
        "02d_visual_caption_benchmark",
    ]
    assert [item["id"] for item in group_07] == [
        "07a_efficiency_pareto",
        "07b_error_propagation",
    ]
    for item in group_02 + group_07:
        assert item["disk_budget"]["estimated_peak_bytes"] == 10 * 1024**3
        assert item["disk_budget"]["validated_by"] == "00a_environment_preflight"
        assert item["network_budget"]["policy"] == item["network_policy"]


def test_missing_upstream_fails_before_kernel(tmp_path: Path, monkeypatch, capsys) -> None:
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("kernel execution must not start")

    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks.execute_notebook", fail_if_called
    )
    code = executor_main(
        [
            "--notebook",
            "00b_dataset_feasibility",
            "--profile",
            "smoke",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--output-root",
            str(tmp_path / "outputs"),
        ]
    )
    assert code == 1
    assert called is False
    assert "required artifact validation failed" in capsys.readouterr().out


def test_runner_code_state_changes_config_hash_and_invalidates_resume(
    tmp_path: Path, monkeypatch
) -> None:
    entry = load_manifest()["notebooks"][0]
    source = tmp_path / "source.ipynb"
    destination = tmp_path / "executed.ipynb"
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('ok')")])
    nbformat.write(notebook, source)
    nbformat.write(notebook, destination)
    dependency_hash = "d" * 64
    produced_hash = "e" * 64
    receipt = {
        "schema_version": 1,
        "notebook_id": entry["id"],
        "profile": "smoke",
        "run_namespace": "official",
        "status": "real",
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
        "config_sha256": _entry_config_sha256(entry, "smoke", "official"),
        "dependency_state_sha256": dependency_hash,
        "produced_state_sha256": produced_hash,
    }
    destination.with_suffix(".run.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert can_resume(
        entry, source, destination, "smoke", "official",
        dependency_hash, "real", produced_hash,
    )[0]
    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks._code_state_sha256",
        lambda _entry: "f" * 64,
    )
    resumable, reason = can_resume(
        entry, source, destination, "smoke", "official",
        dependency_hash, "real", produced_hash,
    )
    assert resumable is False
    assert "config_sha256" in reason


def test_fallback_only_scientific_artifact_cannot_be_official_success(tmp_path: Path) -> None:
    context = RunnerContext(
        profile="smoke",
        run_namespace="official",
        status="real",
        artifact_root=tmp_path,
        notebook_id="fixture_notebook",
        run_id=str(uuid.uuid4()),
        source_sha256="a" * 64,
        config_sha256="b" * 64,
        dependency_state_sha256="c" * 64,
    )
    write_json_artifact(
        context,
        "asr.aggregate",
        [{"candidate": "fallback", "execution_mode": "fallback", "metric": 0.1}],
    )
    write_run_manifest(context)
    errors = validate_produced_artifacts(
        {"id": context.notebook_id, "produces": ["asr.aggregate"], "expected_schema_version": 1},
        tmp_path,
        "smoke",
        "official",
        {
            "run_id": context.run_id,
            "status": context.status,
            "source_sha256": context.source_sha256,
            "config_sha256": context.config_sha256,
            "dependency_state_sha256": context.dependency_state_sha256,
        },
    )
    assert any("fallback-only evidence" in error for error in errors)


def test_empty_metric_mapping_is_rejected(tmp_path: Path) -> None:
    payload = tmp_path / "empty-metrics.json"
    payload.write_text('{"metrics": {}}\n', encoding="utf-8")
    errors, record_count = _validate_payload(payload, "asr.aggregate")
    assert record_count == 0
    assert any("field 'metrics' is empty" in error for error in errors)


def test_nested_empty_or_null_metrics_are_scientifically_empty(tmp_path: Path) -> None:
    payloads = (
        {"records": [{"unit_id": "u1", "metrics": {}}]},
        {"metrics": {"wer": None}},
        {"metrics": {"wer": float("nan")}},
    )
    for index, value in enumerate(payloads):
        payload = tmp_path / f"empty-scientific-{index}.json"
        payload.write_text(json.dumps(value), encoding="utf-8")
        errors = _scientifically_empty_errors(payload, "asr.aggregate")
        assert any("no finite metric" in error for error in errors)


def test_payload_level_and_nested_fallback_markers_are_rejected(tmp_path: Path) -> None:
    payloads = (
        {"metrics": {"wer": 0.1}, "execution_mode": "fallback_cpu"},
        {"metrics": {"wer": 0.1}, "execution_mode": "fallback-cpu"},
        {"metrics": {"wer": 0.1}, "provenance": {"fallback_used": True}},
        {"records": [{"metrics": {"wer": 0.1}, "model_loaded": False}]},
    )
    for index, value in enumerate(payloads):
        payload = tmp_path / f"fallback-{index}.json"
        payload.write_text(json.dumps(value), encoding="utf-8")
        errors = _fallback_only_errors(payload, "asr.aggregate")
        assert any("fallback-only evidence" in error for error in errors)


def test_status_sentinels_cannot_be_official_scientific_evidence(tmp_path: Path) -> None:
    invalid_artifacts = (
        ("asr.aggregate", {"metrics": {"wer": "not_run"}}),
        ("asr.aggregate", {"metrics": {"wer": "NaN"}}),
        ("asr.aggregate", {"metrics": {"wer": None, "sample_count": 10}}),
        (
            "asr.aggregate",
            {
                "metrics": {
                    "status": "not eligible",
                    "frame_index": 12,
                    "timestamp_ms": 1000,
                    "rank": 1,
                    "offset": 2,
                    "wer_sample_count": 12,
                    "wer_seed": 42,
                }
            },
        ),
        ("asr.raw_predictions", {"predictions": "not_run"}),
        ("asr.raw_predictions", {"predictions": {"unit_id": "u1", "status": "real"}}),
        (
            "asr.raw_predictions",
            {"predictions": {"status": "not eligible", "unit_count": 5}},
        ),
        ("retrieval.rankings", {"rankings": ["blocked"]}),
    )
    for index, (artifact_id, payload) in enumerate(invalid_artifacts):
        artifact_root = tmp_path / str(index)
        context = RunnerContext(
            profile="smoke",
            run_namespace="official",
            status="real",
            artifact_root=artifact_root,
            notebook_id="fixture_notebook",
            run_id=str(uuid.uuid4()),
            source_sha256="a" * 64,
            config_sha256="b" * 64,
            dependency_state_sha256="c" * 64,
        )
        write_json_artifact(context, artifact_id, payload)
        write_run_manifest(context)
        errors = validate_produced_artifacts(
            {
                "id": context.notebook_id,
                "produces": [artifact_id],
                "expected_schema_version": 1,
            },
            artifact_root,
            "smoke",
            "official",
            {
                "run_id": context.run_id,
                "status": context.status,
                "source_sha256": context.source_sha256,
                "config_sha256": context.config_sha256,
                "dependency_state_sha256": context.dependency_state_sha256,
            },
        )
        assert any("no finite metric" in error for error in errors)


def test_finite_allowlisted_metric_is_scientifically_meaningful(tmp_path: Path) -> None:
    payload = tmp_path / "real-metric.json"
    payload.write_text(json.dumps({"metrics": {"wer": 0.125}}), encoding="utf-8")
    assert _scientifically_empty_errors(payload, "asr.aggregate") == []
    payload.write_text(
        json.dumps({"metrics": {"model_a": {"window_diff": {"mean": 0.2}}}}),
        encoding="utf-8",
    )
    assert _scientifically_empty_errors(payload, "chapter.aggregate") == []
    payload.write_text(
        json.dumps({"metrics": {"model_a": {"corpus_wer": 0.15}}}),
        encoding="utf-8",
    )
    assert _scientifically_empty_errors(payload, "asr.aggregate") == []


def test_00a_to_00c_real_local_artifact_chain(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "datasets"
    dataset = data_root / "fixture"
    (dataset / "train").mkdir(parents=True)
    (dataset / "test").mkdir(parents=True)
    (dataset / "train" / "one.json").write_text('{"value": 1}\n', encoding="utf-8")
    (dataset / "test" / "two.txt").write_text("real local evidence\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()

    monkeypatch.setattr(environment_preflight, "DATA_ROOT", data_root)
    monkeypatch.setattr(probe_dataset_feasibility, "DATA_ROOT", data_root)
    manifest = load_manifest()
    entries = {entry["id"]: entry for entry in manifest["notebooks"]}

    runners = (
        ("00a_environment_preflight", environment_preflight.main),
        ("00b_dataset_feasibility", probe_dataset_feasibility.main),
        ("00c_dataset_eda_and_freeze", profile_dataset.main),
    )
    for notebook_id, runner in runners:
        run_id = str(uuid.uuid4())
        assert runner(_runner_args(artifact_root, notebook_id, run_id)) == 0
        assert validate_produced_artifacts(
            entries[notebook_id], artifact_root, "smoke", "official", _context(run_id)
        ) == []

    frozen = artifact_root / "official" / "smoke" / "payloads" / "dataset.frozen_manifest.json"
    figure = artifact_root / "official" / "smoke" / "figures" / "selected-bytes-by-dataset.svg"
    assert frozen.is_file()
    assert figure.is_file()
    frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert all(record["freeze_status"] == "not_eligible" for record in frozen_payload["records"])
    assert all(record["unit_ids"] == [] for record in frozen_payload["records"])
    assert "fixture" in figure.read_text(encoding="utf-8")

    receipt_path = artifact_root / "official" / "smoke" / "dataset.feasibility_receipts.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_id"] = str(uuid.uuid4())
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    producers = {
        artifact: entry["id"]
        for entry in manifest["notebooks"]
        for artifact in entry["produces"]
    }
    downstream_errors = validate_required_artifacts(
        entries["01a_asr_model_benchmark"],
        artifact_root,
        "smoke",
        "official",
        producers,
    )
    assert any("no scientifically eligible frozen units" in error for error in downstream_errors)

    errors = validate_required_artifacts(
        entries["00c_dataset_eda_and_freeze"],
        artifact_root,
        "smoke",
        "official",
        producers,
    )
    assert any("does not match upstream run manifest" in error for error in errors)


def test_runner_rejects_non_real_official_status(tmp_path: Path) -> None:
    args = _runner_args(tmp_path, "00a_environment_preflight", str(uuid.uuid4()))
    args[args.index("real")] = "dry"
    assert environment_preflight.main(args) == 1


def test_smoke_selection_is_hidden_archive_and_byte_bounded(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / ".metadata").write_text("hidden", encoding="utf-8")
    (dataset / "archive.zip").write_bytes(b"archive")
    (dataset / "train.json").write_text("{}", encoding="utf-8")
    selected = probe_dataset_feasibility._smoke_files(
        dataset, list(probe_dataset_feasibility._dataset_files(dataset))
    )
    assert [path.name for path in selected] == ["train.json"]

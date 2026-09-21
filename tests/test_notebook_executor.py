"""Tests for manifest-driven notebook execution planning."""

from copy import deepcopy

import pytest

from benchmarks.scripts.execute_all_notebooks import (
    ManifestError,
    build_plan,
    can_resume,
    execute_notebook,
    load_manifest,
    main,
    select_entries,
    topological_order,
    validate_produced_artifacts,
    validate_required_artifacts,
)


def test_manifest_drives_all_22_notebooks_in_topological_order():
    manifest = load_manifest()
    ordered = select_entries(manifest, profile="smoke")
    assert len(ordered) == 22
    positions = {entry["id"]: index for index, entry in enumerate(ordered)}
    assert positions["00c_dataset_eda_and_freeze"] < positions["01a_asr_model_benchmark"]
    assert positions["08b_end_to_end_system"] < positions["09_cross_task_report"]


def test_selectors_use_manifest_ids_groups_and_tasks():
    manifest = load_manifest()
    assert [entry["id"] for entry in select_entries(manifest, notebook="02c_ocr_benchmark")] == [
        "02c_ocr_benchmark"
    ]
    assert len(select_entries(manifest, group="02")) == 4
    assert [entry["id"] for entry in select_entries(manifest, task="ASR_MODEL")] == [
        "01a_asr_model_benchmark"
    ]


def test_unknown_selector_fails_closed():
    with pytest.raises(ManifestError, match="No notebook matches"):
        select_entries(load_manifest(), notebook="does-not-exist")


def test_topological_order_rejects_cycle():
    entries = [
        {"id": "a", "requires": ["b.out"], "produces": ["a.out"]},
        {"id": "b", "requires": ["a.out"], "produces": ["b.out"]},
    ]
    with pytest.raises(ManifestError, match="cycle"):
        topological_order(entries)


def test_topological_order_preserves_manifest_order_when_dependencies_allow():
    entries = [
        {"id": "a", "requires": [], "produces": ["a.out"]},
        {"id": "b", "requires": ["a.out"], "produces": ["b.out"]},
        {"id": "c", "requires": [], "produces": ["c.out"]},
    ]
    assert [entry["id"] for entry in topological_order(entries)] == ["a", "b", "c"]


def test_dry_plan_exposes_resources_dependencies_and_missing_sources():
    manifest = load_manifest()
    entry = select_entries(manifest, notebook="05a_summarization_benchmark")
    plan = build_plan(entry, "smoke")
    assert plan[0]["profile"] == "smoke"
    assert plan[0]["timeout_seconds"] > 0
    assert plan[0]["network_policy"] == "registry_controlled"
    assert "04b_chaptering_modality_ablation" in plan[0]["upstream_notebooks"]
    assert plan[0]["source_exists"] is False


def test_cli_list_and_dry_plan_do_not_execute(capsys):
    assert main(["--group", "02", "--profile", "smoke", "--list"]) == 0
    listed = capsys.readouterr().out
    assert "02a_scene_boundary_benchmark" in listed
    assert "02d_visual_caption_benchmark" in listed
    assert main(["--task", "ASR_MODEL", "--dry-plan"]) == 0
    planned = capsys.readouterr().out
    assert '"id": "01a_asr_model_benchmark"' in planned
    assert '"disk_budget"' in planned


def test_malformed_manifest_returns_controlled_cli_error(tmp_path, capsys):
    manifest_path = tmp_path / "malformed.yaml"
    manifest_path.write_text("[", encoding="utf-8")
    assert main(["--manifest", str(manifest_path), "--list"]) == 2
    assert "[FAIL] Cannot load notebook manifest" in capsys.readouterr().out


def test_required_artifact_receipt_validates_schema_status_profile_and_hash(tmp_path):
    entry = deepcopy(load_manifest()["notebooks"][1])
    artifact_id = entry["requires"][0]
    entry["requires"] = [artifact_id]
    payload = tmp_path / "official" / "smoke" / "payload.json"
    payload.parent.mkdir(parents=True)
    payload.write_text('{"row": 1}', encoding="utf-8")
    import hashlib
    import json

    receipt = {
        "artifact_id": artifact_id,
        "schema_version": 1,
        "profile": "smoke",
        "run_namespace": "official",
        "status": "real",
        "payload_path": "official/smoke/payload.json",
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        "record_count": 1,
        "denominator": 1,
        "producer_notebook_id": "00a_environment_preflight",
        "run_id": "00000000-0000-4000-8000-000000000001",
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "dependency_state_sha256": "c" * 64,
    }
    receipt_path = tmp_path / "official" / "smoke" / f"{artifact_id}.receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert validate_required_artifacts(entry, tmp_path, "smoke", "official") == []
    receipt["status"] = "proxy"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert any(
        "official artifact status must be real" in error
        for error in validate_required_artifacts(entry, tmp_path, "smoke", "official")
    )
    receipt["status"] = "real"
    payload.write_bytes(b"")
    receipt["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert any(
        "payload is missing or empty" in error
        for error in validate_required_artifacts(entry, tmp_path, "smoke", "official")
    )


def test_produced_artifacts_require_receipts_and_runner_manifest(tmp_path):
    entry = deepcopy(load_manifest()["notebooks"][0])
    context = {
        "run_id": "run-1",
        "status": "real",
        "source_sha256": "source",
        "config_sha256": "config",
        "dependency_state_sha256": "dependencies",
    }
    errors = validate_produced_artifacts(entry, tmp_path, "smoke", "official", context)
    assert any("missing receipt" in error for error in errors)
    assert any("run manifest missing" in error for error in errors)


def test_profile_and_namespace_are_propagated_to_kernel(tmp_path, monkeypatch):
    notebook = tmp_path / "source.ipynb"
    destination = tmp_path / "executed.ipynb"
    notebook.write_text(
        '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}', encoding="utf-8"
    )
    captured = {}

    class FakeClient:
        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self, **kwargs):
            captured.update(kwargs["env"])

    monkeypatch.setattr("benchmarks.scripts.execute_all_notebooks.NotebookClient", FakeClient)
    entry = deepcopy(load_manifest()["notebooks"][0])
    ok, _, error = execute_notebook(
        entry, notebook, destination, 30, "smoke", "diagnostic", tmp_path,
        {
            "run_id": "run-1",
            "status": "dry",
            "source_sha256": "source",
            "config_sha256": "config",
            "dependency_state_sha256": "dependencies",
        },
    )
    assert ok and error is None
    assert captured["MLS_BENCHMARK_PROFILE"] == "smoke"
    assert captured["MLS_RUN_NAMESPACE"] == "diagnostic"
    assert captured["MLS_NOTEBOOK_ID"] == entry["id"]
    assert captured["MLS_ARTIFACT_ROOT"] == str(tmp_path.resolve())


def test_resume_rejects_empty_or_unreceipted_output(tmp_path):
    entry = deepcopy(load_manifest()["notebooks"][0])
    source = tmp_path / "source.ipynb"
    destination = tmp_path / "output.ipynb"
    source.write_text(
        '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}', encoding="utf-8"
    )
    destination.write_bytes(b"")
    assert can_resume(
        entry, source, destination, "smoke", "official", "dependency-hash", "real", "produced-hash"
    )[0] is False


def test_executor_does_not_overwrite_source_notebook(tmp_path, monkeypatch):
    manifest = load_manifest()
    entry = deepcopy(manifest["notebooks"][0])
    source = tmp_path / "source.ipynb"
    source.write_text("source", encoding="utf-8")
    entry["path"] = str(source)
    before = source.read_bytes()

    def fake_execute(
        entry_value, source_path, destination, timeout, profile, run_namespace,
        artifact_root, run_context,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            '{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}', encoding="utf-8"
        )
        return True, 0.1, None

    monkeypatch.setattr("benchmarks.scripts.execute_all_notebooks.PROJECT_ROOT", tmp_path)
    monkeypatch.setattr("benchmarks.scripts.execute_all_notebooks.execute_notebook", fake_execute)
    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks.validate_produced_artifacts",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks.produced_state_sha256",
        lambda *args, **kwargs: "produced-state",
    )
    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks.select_entries", lambda *args, **kwargs: [entry]
    )
    monkeypatch.setattr(
        "benchmarks.scripts.execute_all_notebooks.load_manifest", lambda *args, **kwargs: manifest
    )
    output_root = tmp_path / "outputs"
    assert main([
        "--notebook", entry["id"], "--output-root", str(output_root),
        "--artifact-root", str(tmp_path / "artifacts"),
    ]) == 0
    assert source.read_bytes() == before
    executed = output_root / "official" / "full" / f"{entry['id']}.ipynb"
    assert executed.is_file()
    assert executed.with_suffix(".run.json").is_file()

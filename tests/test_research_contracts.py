"""Tests for frozen research, metric, and notebook contracts."""

from copy import deepcopy

import pytest

from experiments.registry.contract_validator import (
    DEFAULT_DECISIONS,
    DEFAULT_METRICS,
    DEFAULT_NOTEBOOKS,
    EDA_OUTPUTS,
    load_yaml,
    validate_contracts,
    validate_metric_registry,
    validate_notebook_manifest,
    validate_research_decisions,
)


def test_default_contracts_are_valid():
    assert validate_contracts() == []


def test_research_decisions_freeze_user_choices():
    decisions = load_yaml(DEFAULT_DECISIONS)
    assert validate_research_decisions(decisions) == []
    assert decisions["language"]["primary"] == "en"
    assert decisions["summary_budgets"] == {
        "short": 0.05,
        "medium": 0.10,
        "long": 0.20,
        "text_denominator": "transcript_word_count",
        "visual_denominator": "source_video_duration",
    }
    assert decisions["ground_truth"]["project_manual_annotation_allowed"] is False
    assert decisions["speaker_diarization"]["official_benchmark"] is False


def test_research_decisions_freeze_phase2_dataset_execution():
    decisions = load_yaml(DEFAULT_DECISIONS)
    phase2 = decisions["phase2_dataset_execution"]
    assert phase2 == {
        "first_target": {
            "dataset_id": "tedlium3",
            "track_id": "asr_english",
            "qualification_compute_profile": "local_rtx4050_6gb",
            "acquisition_policy": "existing_local_integrity_scan_only",
            "source_of_truth": "local",
            "split_mapping": {
                "official_dev": "validation",
                "official_test": "final_test",
                "preserve_original_split": True,
            },
        },
        "artifact_formats": {
            "unit_level": "parquet",
            "aggregate": "csv",
            "manifests": "json",
        },
        "next_target": {"dataset_id": "iiit_cvid", "endpoint": "ocr"},
        "downstream_execution": {
            "prerequisite": "d2_validated",
            "asr_model_benchmark_compute_profile": "colab_t4",
            "asr_ablation_compute_profile": "colab_t4",
        },
        "artifact_transport": {
            "provider": "google_drive",
            "logical_root": "MyDrive",
            "role": "immutable_artifact_transport",
        },
        "run_control": {
            "target_run_minutes": 60,
            "checkpoint_policy": "adaptive",
            "log_format": "jsonl",
            "log_granularity": "per_task_detailed",
            "long_task_progress_required": True,
        },
    }


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("first_target", "dataset_id"), "other_dataset"),
        (("first_target", "track_id"), "ocr"),
        (("first_target", "qualification_compute_profile"), "colab_t4"),
        (("first_target", "acquisition_policy"), "redownload"),
        (("first_target", "source_of_truth"), "google_drive"),
        (("first_target", "split_mapping", "official_dev"), "train"),
        (("first_target", "split_mapping", "official_test"), "test"),
        (("first_target", "split_mapping", "preserve_original_split"), False),
        (("artifact_formats", "unit_level"), "json"),
        (("artifact_formats", "aggregate"), "parquet"),
        (("artifact_formats", "manifests"), "yaml"),
        (("next_target", "dataset_id"), "eduvidqa"),
        (("next_target", "endpoint"), "qa"),
        (("downstream_execution", "prerequisite"), "d1_validated"),
        (("downstream_execution", "asr_model_benchmark_compute_profile"), "local_rtx4050_6gb"),
        (("downstream_execution", "asr_ablation_compute_profile"), "local_rtx4050_6gb"),
        (("artifact_transport", "provider"), "local_disk"),
        (("artifact_transport", "logical_root"), "MyDrive/benchmark"),
        (("artifact_transport", "role"), "source_of_truth"),
        (("run_control", "target_run_minutes"), 120),
        (("run_control", "checkpoint_policy"), "per_unit"),
        (("run_control", "log_format"), "text"),
        (("run_control", "log_granularity"), "summary_only"),
        (("run_control", "long_task_progress_required"), False),
    ],
)
def test_research_decisions_reject_phase2_mutations(path, invalid_value):
    decisions = load_yaml(DEFAULT_DECISIONS)
    target = decisions["phase2_dataset_execution"]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = invalid_value
    errors = validate_research_decisions(decisions)
    assert any("phase2_dataset_execution" in error for error in errors)


def test_research_decisions_reject_manual_ground_truth():
    decisions = load_yaml(DEFAULT_DECISIONS)
    decisions["ground_truth"]["project_manual_annotation_allowed"] = True
    errors = validate_research_decisions(decisions)
    assert any("project_manual_annotation_allowed" in error for error in errors)


@pytest.mark.parametrize(
    ("section", "field", "invalid_value"),
    [
        ("summary_budgets", "text_denominator", "frame_count"),
        ("ground_truth", "policy", "project_authored"),
        ("ground_truth", "heuristic_references_allowed_for_official_evaluation", True),
        ("speaker_diarization", "selection_gate", True),
    ],
)
def test_research_decisions_reject_other_frozen_mutations(section, field, invalid_value):
    decisions = load_yaml(DEFAULT_DECISIONS)
    decisions[section][field] = invalid_value
    assert validate_research_decisions(decisions)


def test_research_decisions_reject_hardware_and_api_policy_mutations():
    decisions = load_yaml(DEFAULT_DECISIONS)
    decisions["compute_profiles"][0]["accelerator"] = "cpu"
    decisions["compute_profiles"][1]["leaderboard_scope"] = "combined"
    decisions["model_policy"]["llm_api_exception"]["justification_required"] = False
    decisions["model_policy"]["llm_api_exception"]["deterministic_fallback_required"] = False
    assert validate_research_decisions(decisions)


def test_metric_registry_has_one_valid_primary_per_required_track():
    metrics = load_yaml(DEFAULT_METRICS)
    assert validate_metric_registry(metrics) == []
    for track in metrics["tracks"]:
        if track["id"] == "multimodal_summary_package":
            assert track["primary_metric"] is None
            assert track["selection_mode"] == "pareto_or_gate"
            assert len(track["metric_vector"]) >= 2
        else:
            assert set(track["primary_metric"]) == {"name", "direction"}


def test_metric_registry_rejects_invalid_primary_direction():
    metrics = load_yaml(DEFAULT_METRICS)
    metrics["tracks"][0]["primary_metric"]["direction"] = "sideways"
    errors = validate_metric_registry(metrics)
    assert any("invalid primary direction" in error for error in errors)


def test_metric_registry_rejects_expanded_allowed_enums():
    metrics = load_yaml(DEFAULT_METRICS)
    metrics["allowed_directions"].append("sideways")
    metrics["allowed_failure_policies"].append("ignore_failures")
    metrics["tracks"][0]["primary_metric"]["direction"] = "sideways"
    metrics["tracks"][0]["failure_policy"] = "ignore_failures"
    errors = validate_metric_registry(metrics)
    assert any("allowed_directions must be exactly" in error for error in errors)
    assert any("allowed_failure_policies must be exactly" in error for error in errors)
    assert any("invalid primary direction" in error for error in errors)
    assert any("invalid failure_policy" in error for error in errors)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda track: track["scorer"].update(version=None),
        lambda track: track.update(practical_threshold=None),
        lambda track: track["primary_metric"].update(bogus="field"),
    ],
)
def test_metric_registry_rejects_fail_open_mutations(mutation):
    metrics = load_yaml(DEFAULT_METRICS)
    mutation(metrics["tracks"][0])
    assert validate_metric_registry(metrics)


def test_notebook_manifest_has_22_unique_acyclic_entries():
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    assert validate_notebook_manifest(manifest) == []
    assert len(manifest["notebooks"]) == 22
    assert len({entry["id"] for entry in manifest["notebooks"]}) == 22
    assert len({entry["path"] for entry in manifest["notebooks"]}) == 22


def test_dataset_eda_notebook_declares_full_artifact_contract():
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    eda = next(entry for entry in manifest["notebooks"] if entry["id"] == "00c_dataset_eda_and_freeze")
    assert EDA_OUTPUTS <= set(eda["produces"])
    assert "dataset.cross_dataset_comparison" in eda["produces"]
    assert "dataset.eda_report" in eda["produces"]


def test_notebook_manifest_rejects_output_collision():
    manifest = deepcopy(load_yaml(DEFAULT_NOTEBOOKS))
    manifest["notebooks"][1]["produces"].append("environment.profile")
    errors = validate_notebook_manifest(manifest)
    assert any("output collision" in error for error in errors)


def test_notebook_manifest_rejects_missing_dependency_and_cycle():
    manifest = deepcopy(load_yaml(DEFAULT_NOTEBOOKS))
    manifest["notebooks"][1]["requires"].append("missing.artifact")
    manifest["notebooks"][0]["requires"].append("system.final_selection")
    errors = validate_notebook_manifest(manifest)
    assert any("unresolved required artifact" in error for error in errors)
    assert any("dependency cycle" in error for error in errors)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda entry: entry["produces"].append(""),
        lambda entry: entry.update(path="README.md"),
        lambda entry: entry.update(network_policy="unbounded"),
        lambda entry: entry.update(timeout_seconds=0),
    ],
)
def test_notebook_manifest_rejects_invalid_entry_fields(mutation):
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    mutation(manifest["notebooks"][0])
    assert validate_notebook_manifest(manifest)


def test_notebook_manifest_rejects_expanded_allowed_enums():
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    manifest["allowed_profiles"].append("mock")
    manifest["allowed_network_policies"].append("unbounded")
    manifest["notebooks"][0]["profiles"] = ["mock"]
    manifest["notebooks"][0]["network_policy"] = "unbounded"
    errors = validate_notebook_manifest(manifest)
    assert any("allowed_profiles must be exactly" in error for error in errors)
    assert any("allowed_network_policies must be exactly" in error for error in errors)
    assert any("invalid profiles" in error for error in errors)
    assert any("invalid network policy" in error for error in errors)


def test_notebook_manifest_rejects_missing_eda_outputs():
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    eda = next(entry for entry in manifest["notebooks"] if entry["id"] == "00c_dataset_eda_and_freeze")
    eda["produces"].remove("dataset.eda_report")
    errors = validate_notebook_manifest(manifest)
    assert any("00c missing required EDA outputs" in error for error in errors)


def test_notebook_manifest_rejects_non_mapping_entry_without_exception():
    manifest = load_yaml(DEFAULT_NOTEBOOKS)
    manifest["notebooks"][0] = "not-a-mapping"
    errors = validate_notebook_manifest(manifest)
    assert any("notebook entry must be a mapping" in error for error in errors)


def test_load_yaml_rejects_missing_and_non_mapping(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "missing.yaml")
    invalid = tmp_path / "list.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(invalid)

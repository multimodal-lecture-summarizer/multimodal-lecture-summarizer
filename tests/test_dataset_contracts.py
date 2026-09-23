"""Contract tests for fail-closed D2 dataset freezing."""

from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path

import jsonschema
import pytest

from experiments.evaluation.dataset_contracts import (
    DatasetContractError,
    MAX_D2_PEAK_BYTES,
    SCHEMA_PATH,
    UNIVERSAL_D2_FIGURE_CONTRACTS,
    UNIVERSAL_D2_FIGURE_IDS,
    validate_dataset_characterization_bundle,
    validate_dataset_bundle,
)


def _valid_bundle() -> dict:
    dataset_id = "released-lecture-set"
    units = [
        {
            "dataset_id": dataset_id,
            "unit_id": "lecture-001",
            "source": "https://example.invalid/dataset/unit-2",
            "revision": "v1.2.0",
            "rights": "CC-BY-4.0",
            "split": "validation",
            "bytes": 1024,
            "sha256": "a" * 64,
            "decoded": True,
            "speaker_id": "speaker-a",
            "talk_id": "talk-a",
            "ground_truth": {
                "status": "released",
                "source": "official annotations",
                "revision": "v1.2.0",
                "annotation_protocol": "dataset card section 4",
            },
        },
        {
            "dataset_id": dataset_id,
            "unit_id": "lecture-002",
            "source": "https://example.invalid/dataset",
            "revision": "v1.2.0",
            "rights": "CC-BY-4.0",
            "split": "final_test",
            "bytes": 2048,
            "sha256": "b" * 64,
            "decoded": True,
            "speaker_id": "speaker-b",
            "talk_id": "talk-b",
            "ground_truth": {
                "status": "released",
                "source": "official annotations",
                "revision": "v1.2.0",
                "annotation_protocol": "dataset card section 4",
            },
        },
    ]
    return {
        "registry": [
            {
                "dataset_id": dataset_id,
                "source": "https://example.invalid/dataset",
                "revision": "v1.2.0",
                "rights": "CC-BY-4.0",
                "access_status": "available",
                "expected_bytes": 3072,
                "predicted_peak_bytes": 8192,
                "acquisition_level": "D2",
                "fallback_dataset": None,
                "data_root_key": "LECTURE_DATA_ROOT",
                "receipt_path": "receipts/released-lecture-set.json",
                "leakage_exception": None,
            }
        ],
        "feasibility_receipts": [
            {
                "dataset_id": dataset_id,
                "acquisition_level": "D2",
                "schema_success_rate": 1.0,
                "checksum_success_rate": 1.0,
                "download_success_rate": 0.95,
                "decode_success_rate": 1.0,
                "predicted_peak_bytes": 8192,
                "estimated_completion_hours": 1.5,
            }
        ],
        "units": units,
        "profiles": [
            {
                "dataset_id": dataset_id,
                "raw_file_count": 2,
                "eligible_file_count": 2,
                "excluded_file_count": 0,
                "unresolved_file_count": 0,
                "source": "https://example.invalid/dataset",
                "acquisition_level": "D2",
                "profile": "full",
                "selected_bytes": 3072,
                "selection_policy": "frozen IDs",
                "ground_truth_status": "released",
            }
        ],
        "attrition": [
            {
                "dataset_id": dataset_id,
                "raw": 2,
                "eligible": 2,
                "excluded": 0,
                "unresolved": 0,
                "scope": "full",
            }
        ],
        "frozen_manifest": [
            {
                "dataset_id": dataset_id,
                "freeze_status": "eligible",
                "scientific_selection_eligible": True,
                "unit_ids": ["lecture-001", "lecture-002"],
                "reason": "all D2 gates passed",
            }
        ],
        "endpoint_matrix": [
            {
                "dataset_id": dataset_id,
                "endpoint": "object_detection",
                "eligibility": "eligible",
                "scientific_selection_eligible": True,
                "reason": "released box/class GT",
            }
        ],
    }


def _valid_characterization_bundle(tmp_path: Path) -> dict:
    bundle = _valid_bundle()
    dataset_id = bundle["profiles"][0]["dataset_id"]
    source_hash = hashlib.sha256(
        json.dumps(bundle["units"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    comparison_record = {
        "dataset_id": dataset_id,
        "metric_id": "selected_bytes",
        "value": 3072,
        "definition_id": "physical_selected_file_bytes_v1",
        "unit": "bytes",
        "revision": "v1.2.0",
        "acquisition_level": "D2",
        "scope": "full",
        "denominator": 2,
        "source_table_sha256": source_hash,
    }
    comparison_hash = hashlib.sha256(
        json.dumps([comparison_record], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    figures = []
    for figure_id in sorted(UNIVERSAL_D2_FIGURE_IDS):
        path = tmp_path / f"{figure_id}.svg"
        path.write_text(f"<svg><title>{figure_id}</title></svg>", encoding="utf-8")
        figures.append(
            {
                "dataset_id": dataset_id,
                "figure_id": figure_id,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_table_sha256": comparison_hash,
                "revision": "v1.2.0",
                "acquisition_level": "D2",
                "split_scope": "validation+final_test",
                "profile": "full",
                "unit": "bytes",
                "n": 2,
                "missing_policy": "reported, not imputed",
                "exclusion_policy": "frozen before inference",
                "source_artifact": "dataset.cross_dataset_comparison",
                "code_revision": "d" * 64,
                "plot_config": {"kind": "fixture"},
            }
        )
    bundle.update(
        statistics=[
            {
                "dataset_id": dataset_id,
                "selected_files": 2,
                "selected_bytes": 3072,
                "file_size_bytes": {
                    "n": 2,
                    "missing_n": 0,
                    "min": 1024,
                    "p5": 1024,
                    "p25": 1024,
                    "median": 1536,
                    "p75": 1024,
                    "p95": 1024,
                    "max": 2048,
                    "mean": 1536,
                    "standard_deviation": 512,
                },
                "extension_counts": {".mp4": 2},
                "revision": "v1.2.0",
                "acquisition_level": "D2",
                "scope": "full",
                "denominator": 2,
                "source_table_sha256": source_hash,
            }
        ],
        split_statistics=[
            {
                "dataset_id": dataset_id,
                "split": split,
                "n": 1,
                "denominator": 2,
                "revision": "v1.2.0",
                "acquisition_level": "D2",
                "scope": "full",
            }
            for split in ("validation", "final_test")
        ],
        quality_issues=[
            {
                "dataset_id": dataset_id,
                "checked_units": 2,
                "issue_count": 0,
                "excluded_units": 0,
                "revision": "v1.2.0",
                "acquisition_level": "D2",
                "scope": "full",
                "issue_policy": "released media decoded before freeze",
            }
        ],
        leakage_audit=[
            {
                "dataset_id": dataset_id,
                "revision": "v1.2.0",
                "acquisition_level": "D2",
                "scope": "full",
                "exact_overlap_count": 0,
                "duplicate_group_count": 0,
                "near_duplicate_status": "adjudicated",
                "adjudication_status": "complete",
                "findings": [],
            }
        ],
        figure_manifest=figures,
        cross_dataset_comparison=[comparison_record],
        artifact_root=tmp_path,
    )
    source_payloads = {
        "dataset.statistics": bundle["statistics"][0],
        "dataset.split_statistics": bundle["split_statistics"],
        "dataset.quality_issues": bundle["quality_issues"][0],
        "dataset.attrition": bundle["attrition"][0],
        "dataset.leakage_audit": bundle["leakage_audit"][0],
        "dataset.cross_dataset_comparison": bundle["cross_dataset_comparison"],
    }
    for figure in bundle["figure_manifest"]:
        source, unit, kind, measure = UNIVERSAL_D2_FIGURE_CONTRACTS[figure["figure_id"]]
        figure.update(
            source_artifact=source,
            unit=unit,
            source_table_sha256=hashlib.sha256(
                json.dumps(source_payloads[source], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            plot_config={"kind": kind, "measure": measure},
        )
    return bundle


def _rebind_comparison_figures(bundle: dict) -> None:
    comparison_hash = hashlib.sha256(
        json.dumps(
            bundle["cross_dataset_comparison"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    for figure in bundle["figure_manifest"]:
        if figure["source_artifact"] == "dataset.cross_dataset_comparison":
            figure["source_table_sha256"] = comparison_hash


def _rebind_all_figure_sources(bundle: dict) -> None:
    source_payloads = {
        "dataset.statistics": bundle["statistics"][0],
        "dataset.split_statistics": bundle["split_statistics"],
        "dataset.quality_issues": bundle["quality_issues"][0],
        "dataset.attrition": bundle["attrition"][0],
        "dataset.leakage_audit": bundle["leakage_audit"][0],
        "dataset.cross_dataset_comparison": bundle["cross_dataset_comparison"],
    }
    for figure in bundle["figure_manifest"]:
        payload = source_payloads[figure["source_artifact"]]
        figure["source_table_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


def _mutate_comparison_value(bundle: dict) -> None:
    bundle["cross_dataset_comparison"][0]["value"] = 1
    _rebind_comparison_figures(bundle)


def _mutate_comparison_source_hash(bundle: dict) -> None:
    bundle["cross_dataset_comparison"][0]["source_table_sha256"] = "not-a-hash"
    _rebind_comparison_figures(bundle)


def test_valid_d2_bundle_passes() -> None:
    result = validate_dataset_bundle(**_valid_bundle())
    assert result == {"dataset_count": 1, "unit_count": 2, "d2_eligible_dataset_count": 1}


def test_valid_d2_characterization_bundle_passes(tmp_path: Path) -> None:
    result = validate_dataset_characterization_bundle(**_valid_characterization_bundle(tmp_path))
    assert result["d2_eligible_dataset_count"] == 1
    assert result["d2_figure_count"] == len(UNIVERSAL_D2_FIGURE_IDS)


def test_full_characterization_bundle_is_valid_against_root_schema(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle.pop("artifact_root")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(bundle, schema)


def test_characterization_rejects_empty_bundle(tmp_path: Path) -> None:
    bundle = {
        key: []
        for key in (
            "registry", "feasibility_receipts", "units", "profiles", "attrition",
            "frozen_manifest", "endpoint_matrix", "statistics", "split_statistics",
            "quality_issues", "leakage_audit", "figure_manifest",
            "cross_dataset_comparison",
        )
    }
    bundle["artifact_root"] = tmp_path
    with pytest.raises(DatasetContractError, match="no datasets"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_rejects_profile_byte_and_split_count_drift(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["profiles"][0]["selected_bytes"] += 1
    with pytest.raises(DatasetContractError, match="profile.*selected_bytes mismatch"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    third = copy.deepcopy(bundle["units"][0])
    third.update(
        unit_id="lecture-003",
        source="https://example.invalid/dataset/unit-3",
        sha256="e" * 64,
        speaker_id="speaker-c",
        talk_id="talk-c",
    )
    bundle["units"].append(third)
    bundle["frozen_manifest"][0]["unit_ids"].append("lecture-003")
    bundle["profiles"][0].update(raw_file_count=3, eligible_file_count=3, selected_bytes=4096)
    bundle["attrition"][0].update(raw=3, eligible=3)
    bundle["statistics"][0].update(
        selected_files=3,
        selected_bytes=4096,
        denominator=3,
        file_size_bytes={
            "n": 3, "missing_n": 0, "min": 1024, "p5": 1024, "p25": 1024,
            "median": 1024, "p75": 1024, "p95": 1024, "max": 2048,
            "mean": 4096 / 3, "standard_deviation": 482.71822929001644,
        },
        extension_counts={".mp4": 3},
    )
    bundle["statistics"][0]["source_table_sha256"] = hashlib.sha256(
        json.dumps(bundle["units"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    for row in bundle["split_statistics"]:
        row["denominator"] = 3
    # Actual counts are validation=2/final_test=1; report the inverse with the same total.
    bundle["split_statistics"][0]["n"] = 1
    bundle["split_statistics"][1]["n"] = 2
    bundle["quality_issues"][0]["checked_units"] = 3
    bundle["cross_dataset_comparison"][0].update(
        value=4096,
        denominator=3,
        source_table_sha256=bundle["statistics"][0]["source_table_sha256"],
    )
    _rebind_comparison_figures(bundle)
    for figure in bundle["figure_manifest"]:
        figure["n"] = 3
    with pytest.raises(DatasetContractError, match="counts disagree with unit profile"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_rejects_unfrozen_unit_and_incomplete_adjudication(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["units"].append(
        {"dataset_id": "released-lecture-set", "unit_id": "unvalidated-extra", "bytes": 1}
    )
    with pytest.raises(DatasetContractError, match="does not match frozen IDs"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    bundle["leakage_audit"][0]["adjudication_status"] = "incomplete"
    with pytest.raises(DatasetContractError, match="leakage audit is incomplete"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_rejects_unbound_or_aliased_figures(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["figure_manifest"][0]["source_artifact"] = "dataset.unregistered_table"
    with pytest.raises(DatasetContractError, match="unknown source_artifact"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    bundle["figure_manifest"][1].update(
        path=bundle["figure_manifest"][0]["path"],
        sha256=bundle["figure_manifest"][0]["sha256"],
    )
    with pytest.raises(DatasetContractError, match="duplicate figure path"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    outside = tmp_path.parent / "outside-characterization.svg"
    outside.write_text("<svg/>", encoding="utf-8")
    bundle["figure_manifest"][0].update(
        path=str(Path("..") / outside.name),
        sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
    )
    with pytest.raises(DatasetContractError, match="path escapes artifact root"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_rejects_mislabeled_or_identical_universal_figures(
    tmp_path: Path,
) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    first, second = bundle["figure_manifest"][:2]
    first["figure_id"], second["figure_id"] = second["figure_id"], first["figure_id"]
    with pytest.raises(DatasetContractError, match="universal semantic contract mismatch"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    first, second = bundle["figure_manifest"][:2]
    first_path = tmp_path / first["path"]
    second_path = tmp_path / second["path"]
    second_path.write_bytes(first_path.read_bytes())
    second["sha256"] = first["sha256"]
    with pytest.raises(DatasetContractError, match="duplicate universal figure content"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    bundle["figure_manifest"][0]["plot_config"] = {}
    with pytest.raises(DatasetContractError, match="invalid plot_config"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_recomputes_cross_split_exact_overlap(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["frozen_manifest"][0].update(
        freeze_status="not_eligible",
        scientific_selection_eligible=False,
        unit_ids=[],
        reason="diagnostic overlap fixture",
    )
    bundle["endpoint_matrix"][0].update(
        eligibility="not_eligible",
        scientific_selection_eligible=False,
        reason="diagnostic overlap fixture",
    )
    bundle["units"][1]["sha256"] = bundle["units"][0]["sha256"]
    unit_hash = hashlib.sha256(
        json.dumps(bundle["units"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle["statistics"][0]["source_table_sha256"] = unit_hash
    bundle["cross_dataset_comparison"][0]["source_table_sha256"] = unit_hash
    bundle["leakage_audit"][0]["duplicate_group_count"] = 1
    _rebind_all_figure_sources(bundle)
    with pytest.raises(DatasetContractError, match="exact overlap findings mismatch"):
        validate_dataset_characterization_bundle(**bundle)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_characterization_rejects_invalid_comparison_values(
    tmp_path: Path, value: float
) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["cross_dataset_comparison"][0]["value"] = value
    _rebind_comparison_figures(bundle)
    with pytest.raises(DatasetContractError, match="invalid value"):
        validate_dataset_characterization_bundle(**bundle)


def test_characterization_rejects_malformed_counts_and_accepts_multiple_metrics(tmp_path: Path) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    bundle["statistics"][0]["extension_counts"] = {".mp4": "2"}
    with pytest.raises(DatasetContractError, match="extension counts mismatch"):
        validate_dataset_characterization_bundle(**bundle)

    bundle = _valid_characterization_bundle(tmp_path)
    second = copy.deepcopy(bundle["cross_dataset_comparison"][0])
    second.update(metric_id="mean_file_size", definition_id="mean_file_size_bytes_v1", value=1536)
    bundle["cross_dataset_comparison"].append(second)
    _rebind_comparison_figures(bundle)
    result = validate_dataset_characterization_bundle(**bundle)
    assert result["d2_eligible_dataset_count"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda b: b["statistics"].clear(), "dataset IDs do not match"),
        (lambda b: b["statistics"][0].update(denominator=1), "unit denominator mismatch"),
        (lambda b: b["split_statistics"][0].update(n=0), "split counts do not reconcile"),
        (lambda b: b["quality_issues"][0].update(issue_count=1), "exclusions do not reconcile"),
        (
            lambda b: b["leakage_audit"][0].update(near_duplicate_status="not_run"),
            "leakage audit is incomplete",
        ),
        (lambda b: b["figure_manifest"].pop(), "missing universal figures"),
        (lambda b: b["figure_manifest"][0].update(sha256="0" * 64), "hash mismatch"),
        (
            _mutate_comparison_value,
            "value disagrees with statistics",
        ),
        (
            lambda b: b["statistics"][0]["file_size_bytes"].update(mean=-999),
            "numeric summary mismatch",
        ),
        (
            lambda b: b["figure_manifest"][0].update(source_table_sha256="not-a-hash"),
            "invalid source_table_sha256",
        ),
        (
            lambda b: b["figure_manifest"].append(dict(b["figure_manifest"][0])),
            "duplicate figure_id",
        ),
        (
            _mutate_comparison_source_hash,
            "invalid source table hash",
        ),
    ],
)
def test_characterization_artifacts_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    bundle = _valid_characterization_bundle(tmp_path)
    mutation(bundle)
    with pytest.raises(DatasetContractError, match=message):
        validate_dataset_characterization_bundle(**bundle)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("units", 0, "rights"), "", "missing required fields"),
        (("units", 0, "sha256"), "bad", "invalid sha256"),
        (("units", 0, "decoded"), False, "was not decoded"),
        (("units", 0, "ground_truth", "status"), "prediction", "GT is not released"),
        (("feasibility_receipts", 0, "download_success_rate"), 0.89, "expected rate"),
        (("feasibility_receipts", 0, "decode_success_rate"), 0.94, "expected rate"),
        (("feasibility_receipts", 0, "predicted_peak_bytes"), MAX_D2_PEAK_BYTES + 1, "50 GiB"),
        (("feasibility_receipts", 0, "estimated_completion_hours"), 12.1, "12 hours"),
    ],
)
def test_d2_rejects_invalid_evidence(path: tuple, value: object, message: str) -> None:
    bundle = _valid_bundle()
    target = bundle
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(DatasetContractError, match=message):
        validate_dataset_bundle(**bundle)


def test_d2_rejects_attrition_mismatch() -> None:
    bundle = _valid_bundle()
    bundle["attrition"][0]["raw"] = 3
    bundle["profiles"][0]["raw_file_count"] = 3
    with pytest.raises(DatasetContractError, match="raw != eligible"):
        validate_dataset_bundle(**bundle)


def test_d2_rejects_profile_or_frozen_count_mismatch() -> None:
    bundle = _valid_bundle()
    bundle["profiles"][0]["eligible_file_count"] = 1
    with pytest.raises(DatasetContractError, match="eligible_file_count"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["attrition"][0].update(eligible=1, excluded=1)
    bundle["profiles"][0].update(eligible_file_count=1, excluded_file_count=1)
    with pytest.raises(DatasetContractError, match="frozen IDs"):
        validate_dataset_bundle(**bundle)


def test_d2_rejects_missing_schema_keys_and_unexpected_gt_fields() -> None:
    bundle = _valid_bundle()
    del bundle["registry"][0]["fallback_dataset"]
    with pytest.raises(DatasetContractError, match="missing schema fields"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    del bundle["frozen_manifest"][0]["reason"]
    with pytest.raises(DatasetContractError, match="missing schema fields"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["units"][0]["ground_truth"]["prediction"] = "must never be GT"
    with pytest.raises(DatasetContractError, match="unexpected GT fields"):
        validate_dataset_bundle(**bundle)


def test_d2_rejects_wrong_types_revision_drift_and_d1_receipt() -> None:
    bundle = _valid_bundle()
    bundle["registry"][0]["source"] = 123
    with pytest.raises(DatasetContractError, match="non-empty strings"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["units"][0]["revision"] = "different-revision"
    with pytest.raises(DatasetContractError, match="revision disagrees"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["feasibility_receipts"][0]["acquisition_level"] = "D1"
    with pytest.raises(DatasetContractError, match="requires D2 receipt"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["feasibility_receipts"][0]["predicted_peak_bytes"] = 4096
    with pytest.raises(DatasetContractError, match="predicted peak disagrees"):
        validate_dataset_bundle(**bundle)


@pytest.mark.parametrize(
    ("artifact", "field", "value", "message"),
    [
        ("profiles", "source", 123, "non-empty strings"),
        ("profiles", "selected_bytes", "not-a-number", "invalid selected_bytes"),
        ("endpoint_matrix", "endpoint", 123, "non-empty strings"),
        ("endpoint_matrix", "reason", False, "non-empty strings"),
        (
            "endpoint_matrix",
            "scientific_selection_eligible",
            "yes",
            "must be boolean",
        ),
    ],
)
def test_strict_profile_and_endpoint_types(
    artifact: str, field: str, value: object, message: str
) -> None:
    bundle = _valid_bundle()
    bundle[artifact][0][field] = value
    with pytest.raises(DatasetContractError, match=message):
        validate_dataset_bundle(**bundle)


@pytest.mark.parametrize("field", ["unit_id", "sha256", "source", "speaker_id", "talk_id"])
def test_d2_rejects_validation_final_test_leakage(field: str) -> None:
    bundle = _valid_bundle()
    bundle["units"][1][field] = bundle["units"][0][field]
    if field == "unit_id":
        bundle["frozen_manifest"][0]["unit_ids"] = ["lecture-001"]
    with pytest.raises(DatasetContractError, match="overlap|duplicate unit"):
        validate_dataset_bundle(**bundle)


def test_noneligible_smoke_remains_fail_closed_without_d2_fields() -> None:
    bundle = _valid_bundle()
    bundle["registry"] = [{"dataset_id": "released-lecture-set"}]
    bundle["feasibility_receipts"] = []
    bundle["units"] = []
    bundle["profiles"][0].update(
        raw_file_count=2,
        eligible_file_count=0,
        excluded_file_count=0,
        unresolved_file_count=2,
    )
    bundle["attrition"][0].update(raw=2, eligible=0, excluded=0, unresolved=2)
    bundle["frozen_manifest"][0].update(
        freeze_status="not_eligible",
        scientific_selection_eligible=False,
        unit_ids=[],
        reason="D0 inventory only",
    )
    bundle["endpoint_matrix"][0]["scientific_selection_eligible"] = False

    result = validate_dataset_bundle(**bundle)
    assert result["d2_eligible_dataset_count"] == 0


def test_d2_requires_validation_final_test_and_leakage_identities() -> None:
    bundle = _valid_bundle()
    bundle["units"][1]["split"] = "train"
    with pytest.raises(DatasetContractError, match="validation and final_test"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["units"][0]["speaker_id"] = None
    with pytest.raises(DatasetContractError, match="leakage identity"):
        validate_dataset_bundle(**bundle)

    bundle["registry"][0]["leakage_exception"] = "dataset release omits speaker IDs; source/talk disjointness used"
    result = validate_dataset_bundle(**bundle)
    assert result["d2_eligible_dataset_count"] == 1


def test_d2_rejects_unsorted_or_unresolved_ids() -> None:
    bundle = _valid_bundle()
    bundle["frozen_manifest"][0]["unit_ids"] = ["lecture-002", "lecture-001"]
    with pytest.raises(DatasetContractError, match="sorted and unique"):
        validate_dataset_bundle(**bundle)

    bundle = _valid_bundle()
    bundle["frozen_manifest"][0]["unit_ids"] = ["lecture-001", "lecture-999"]
    with pytest.raises(DatasetContractError, match="unresolved unit IDs"):
        validate_dataset_bundle(**bundle)

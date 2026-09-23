"""Contract tests for fail-closed D2 dataset freezing."""

from __future__ import annotations

import pytest

from experiments.evaluation.dataset_contracts import (
    DatasetContractError,
    MAX_D2_PEAK_BYTES,
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


def test_valid_d2_bundle_passes() -> None:
    result = validate_dataset_bundle(**_valid_bundle())
    assert result == {"dataset_count": 1, "unit_count": 2, "d2_eligible_dataset_count": 1}


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

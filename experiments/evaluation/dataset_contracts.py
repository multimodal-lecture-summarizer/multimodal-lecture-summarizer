"""Fail-closed validation for D0/D1/D2 dataset evidence bundles."""

from __future__ import annotations

import re
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_PATH = Path(__file__).with_name("dataset_profile_schema.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_D2_PEAK_BYTES = 50 * 1024**3
MAX_D2_HOURS = 12.0


class DatasetContractError(ValueError):
    """Raised when dataset evidence cannot support its declared acquisition level."""


def _index(records: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        dataset_id = record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise DatasetContractError(f"{label}: missing dataset_id")
        if dataset_id in indexed:
            raise DatasetContractError(f"{label}: duplicate dataset_id {dataset_id}")
        indexed[dataset_id] = record
    return indexed


def _require(record: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    missing = [field for field in fields if record.get(field) in (None, "")]
    if missing:
        raise DatasetContractError(f"{label}: missing required fields {missing}")


def _schema_required(definition: str) -> tuple[str, ...]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        required = schema["$defs"][definition]["required"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetContractError(f"dataset schema is invalid: {exc}") from exc
    if not isinstance(required, list) or not all(isinstance(field, str) for field in required):
        raise DatasetContractError(f"dataset schema {definition}.required is invalid")
    return tuple(required)


def _require_schema_keys(record: Mapping[str, Any], definition: str, label: str) -> None:
    missing = [field for field in _schema_required(definition) if field not in record]
    if missing:
        raise DatasetContractError(f"{label}: missing schema fields {missing}")


def _schema_properties(definition: str) -> set[str]:
    try:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        properties = schema["$defs"][definition]["properties"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DatasetContractError(f"dataset schema is invalid: {exc}") from exc
    if not isinstance(properties, dict):
        raise DatasetContractError(f"dataset schema {definition}.properties is invalid")
    return set(properties)


def _reject_extra_keys(record: Mapping[str, Any], definition: str, label: str) -> None:
    extras = sorted(set(record) - _schema_properties(definition))
    if extras:
        raise DatasetContractError(f"{label}: unexpected schema fields {extras}")


def _require_nonempty_strings(record: Mapping[str, Any], fields: Iterable[str], label: str) -> None:
    invalid = [field for field in fields if not isinstance(record.get(field), str) or not record[field]]
    if invalid:
        raise DatasetContractError(f"{label}: fields must be non-empty strings {invalid}")


def _validate_rate(value: Any, minimum: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum or value > 1:
        raise DatasetContractError(f"{label}: expected rate in [{minimum}, 1], got {value!r}")


def _validate_d1_receipt(receipt: Mapping[str, Any], dataset_id: str) -> None:
    _require_schema_keys(receipt, "feasibility_receipt", f"receipt[{dataset_id}]")
    _reject_extra_keys(receipt, "feasibility_receipt", f"receipt[{dataset_id}]")
    _require(
        receipt,
        (
            "acquisition_level",
            "schema_success_rate",
            "checksum_success_rate",
            "download_success_rate",
            "decode_success_rate",
            "predicted_peak_bytes",
            "estimated_completion_hours",
        ),
        f"receipt[{dataset_id}]",
    )
    if receipt["acquisition_level"] not in {"D1", "D2"}:
        raise DatasetContractError(f"receipt[{dataset_id}]: D2 freeze requires D1/D2 receipt")
    _validate_rate(receipt["schema_success_rate"], 1.0, f"receipt[{dataset_id}].schema")
    _validate_rate(receipt["checksum_success_rate"], 1.0, f"receipt[{dataset_id}].checksum")
    _validate_rate(receipt["download_success_rate"], 0.90, f"receipt[{dataset_id}].download")
    _validate_rate(receipt["decode_success_rate"], 0.95, f"receipt[{dataset_id}].decode")
    peak = receipt["predicted_peak_bytes"]
    hours = receipt["estimated_completion_hours"]
    if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0 or peak > MAX_D2_PEAK_BYTES:
        raise DatasetContractError(f"receipt[{dataset_id}]: predicted peak exceeds 50 GiB")
    if not isinstance(hours, (int, float)) or isinstance(hours, bool) or hours < 0 or hours > MAX_D2_HOURS:
        raise DatasetContractError(f"receipt[{dataset_id}]: estimated completion exceeds 12 hours")


def _validate_d2_unit(unit: Mapping[str, Any], dataset_id: str) -> None:
    _require_schema_keys(unit, "unit_record", f"unit[{dataset_id}]")
    _reject_extra_keys(unit, "unit_record", f"unit[{dataset_id}]")
    _require(
        unit,
        ("unit_id", "source", "revision", "rights", "split", "bytes", "sha256", "ground_truth"),
        f"unit[{dataset_id}]",
    )
    if unit["split"] not in {"train", "validation", "final_test"}:
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: invalid split")
    _require_nonempty_strings(
        unit, ("dataset_id", "unit_id", "source", "revision", "rights"), f"unit[{dataset_id}]"
    )
    if not isinstance(unit["bytes"], int) or isinstance(unit["bytes"], bool) or unit["bytes"] <= 0:
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: bytes must be positive")
    if not isinstance(unit["sha256"], str) or not SHA256_RE.fullmatch(unit["sha256"]):
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: invalid sha256")
    if unit.get("decoded") is not True:
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: media was not decoded")
    ground_truth = unit["ground_truth"]
    if not isinstance(ground_truth, Mapping):
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: invalid ground_truth")
    _require(ground_truth, ("status", "source", "revision", "annotation_protocol"), "ground_truth")
    _require_nonempty_strings(
        ground_truth, ("status", "source", "revision", "annotation_protocol"), "ground_truth"
    )
    if ground_truth["status"] != "released":
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: GT is not released")
    allowed_gt_fields = {"status", "source", "revision", "annotation_protocol"}
    if set(ground_truth) - allowed_gt_fields:
        raise DatasetContractError(f"unit[{dataset_id}/{unit['unit_id']}]: unexpected GT fields")


def _reject_validation_test_leakage(units: Sequence[Mapping[str, Any]], dataset_id: str) -> None:
    tracked = ("unit_id", "sha256", "source", "speaker_id", "talk_id")
    values: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in tracked
    }
    for unit in units:
        split = unit.get("split")
        if split not in {"validation", "final_test"}:
            continue
        for field in tracked:
            value = unit.get(field)
            if value not in (None, "", "unknown"):
                values[field][str(value)].add(str(split))
    for field, index in values.items():
        overlaps = sorted(value for value, splits in index.items() if splits == {"validation", "final_test"})
        if overlaps:
            raise DatasetContractError(
                f"dataset[{dataset_id}]: validation/final_test {field} overlap: {overlaps[:3]}"
            )


def validate_dataset_bundle(
    *,
    registry: Sequence[Mapping[str, Any]],
    feasibility_receipts: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    attrition: Sequence[Mapping[str, Any]],
    frozen_manifest: Sequence[Mapping[str, Any]],
    endpoint_matrix: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    """Validate diagnostic records and apply strict cross-artifact gates to D2 claims."""

    registry_by_id = _index(registry, "registry")
    receipts_by_id = _index(feasibility_receipts, "feasibility_receipts")
    profiles_by_id = _index(profiles, "profiles")
    attrition_by_id = _index(attrition, "attrition")
    frozen_by_id = _index(frozen_manifest, "frozen_manifest")
    endpoint_by_id = _index(endpoint_matrix, "endpoint_matrix")
    units_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_unit_keys: set[tuple[str, str]] = set()
    for unit in units:
        _require(unit, ("dataset_id", "unit_id"), "unit")
        key = (str(unit["dataset_id"]), str(unit["unit_id"]))
        if key in seen_unit_keys:
            raise DatasetContractError(f"unit_profile: duplicate unit {key[0]}/{key[1]}")
        seen_unit_keys.add(key)
        units_by_dataset[key[0]].append(unit)

    dataset_ids = set(profiles_by_id)
    for label, indexed in (
        ("registry", registry_by_id),
        ("attrition", attrition_by_id),
        ("frozen_manifest", frozen_by_id),
        ("endpoint_matrix", endpoint_by_id),
    ):
        if set(indexed) != dataset_ids:
            raise DatasetContractError(f"{label}: dataset IDs do not match profiles")

    eligible_count = 0
    for dataset_id in sorted(dataset_ids):
        profile = profiles_by_id[dataset_id]
        attrition_record = attrition_by_id[dataset_id]
        endpoint_record = endpoint_by_id[dataset_id]
        _require_schema_keys(profile, "profile_record", f"profile[{dataset_id}]")
        _reject_extra_keys(profile, "profile_record", f"profile[{dataset_id}]")
        _require_schema_keys(attrition_record, "attrition_record", f"attrition[{dataset_id}]")
        _reject_extra_keys(attrition_record, "attrition_record", f"attrition[{dataset_id}]")
        _require_schema_keys(endpoint_record, "endpoint_record", f"endpoint_matrix[{dataset_id}]")
        _reject_extra_keys(endpoint_record, "endpoint_record", f"endpoint_matrix[{dataset_id}]")
        _require_nonempty_strings(
            profile,
            (
                "dataset_id", "source", "acquisition_level", "profile",
                "selection_policy", "ground_truth_status",
            ),
            f"profile[{dataset_id}]",
        )
        for count_field in (
            "raw_file_count", "eligible_file_count", "excluded_file_count",
            "unresolved_file_count", "selected_bytes",
        ):
            value = profile[count_field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DatasetContractError(f"profile[{dataset_id}]: invalid {count_field}")
        _require_nonempty_strings(attrition_record, ("dataset_id", "scope"), f"attrition[{dataset_id}]")
        _require_nonempty_strings(
            endpoint_record,
            ("dataset_id", "endpoint", "eligibility", "reason"),
            f"endpoint_matrix[{dataset_id}]",
        )
        if not isinstance(endpoint_record["scientific_selection_eligible"], bool):
            raise DatasetContractError(
                f"endpoint_matrix[{dataset_id}]: scientific_selection_eligible must be boolean"
            )
        raw = attrition_record.get("raw")
        eligible = attrition_record.get("eligible")
        excluded = attrition_record.get("excluded")
        unresolved = attrition_record.get("unresolved")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                   for value in (raw, eligible, excluded, unresolved)):
            raise DatasetContractError(f"attrition[{dataset_id}]: counts must be non-negative integers")
        if raw != eligible + excluded + unresolved:
            raise DatasetContractError(f"attrition[{dataset_id}]: raw != eligible + excluded + unresolved")
        if profile.get("raw_file_count") != raw:
            raise DatasetContractError(f"profile[{dataset_id}]: raw count disagrees with attrition")
        for profile_key, attrition_key in (
            ("eligible_file_count", "eligible"),
            ("excluded_file_count", "excluded"),
            ("unresolved_file_count", "unresolved"),
        ):
            if profile.get(profile_key) != attrition_record.get(attrition_key):
                raise DatasetContractError(
                    f"profile[{dataset_id}]: {profile_key} disagrees with attrition"
                )

        frozen = frozen_by_id[dataset_id]
        _require_schema_keys(frozen, "frozen_manifest_record", f"frozen_manifest[{dataset_id}]")
        _reject_extra_keys(frozen, "frozen_manifest_record", f"frozen_manifest[{dataset_id}]")
        _require(frozen, ("reason",), f"frozen_manifest[{dataset_id}]")
        status = frozen.get("freeze_status")
        scientific_eligible = frozen.get("scientific_selection_eligible")
        unit_ids = frozen.get("unit_ids")
        if status not in {"eligible", "not_eligible", "blocked_decision"}:
            raise DatasetContractError(f"frozen_manifest[{dataset_id}]: invalid freeze_status")
        if not isinstance(scientific_eligible, bool) or not isinstance(unit_ids, list):
            raise DatasetContractError(f"frozen_manifest[{dataset_id}]: invalid eligibility fields")
        if status != "eligible":
            if scientific_eligible or unit_ids:
                raise DatasetContractError(f"frozen_manifest[{dataset_id}]: non-eligible dataset has frozen IDs")
            if endpoint_by_id[dataset_id].get("scientific_selection_eligible") is True:
                raise DatasetContractError(f"endpoint_matrix[{dataset_id}]: contradicts frozen manifest")
            continue

        eligible_count += 1
        if scientific_eligible is not True or not unit_ids:
            raise DatasetContractError(f"frozen_manifest[{dataset_id}]: eligible D2 requires non-empty unit_ids")
        if unit_ids != sorted(set(unit_ids)):
            raise DatasetContractError(f"frozen_manifest[{dataset_id}]: unit_ids must be sorted and unique")
        registry_record = registry_by_id[dataset_id]
        _require_schema_keys(registry_record, "registry_record", f"registry[{dataset_id}]")
        _reject_extra_keys(registry_record, "registry_record", f"registry[{dataset_id}]")
        _require(
            registry_record,
            (
                "source", "revision", "rights", "access_status", "expected_bytes",
                "predicted_peak_bytes", "acquisition_level", "data_root_key", "receipt_path",
            ),
            f"registry[{dataset_id}]",
        )
        if registry_record["acquisition_level"] != "D2":
            raise DatasetContractError(f"registry[{dataset_id}]: eligible freeze requires D2 acquisition")
        if registry_record["access_status"] != "available":
            raise DatasetContractError(f"registry[{dataset_id}]: source is not available")
        _require_nonempty_strings(
            registry_record,
            ("dataset_id", "source", "revision", "rights", "data_root_key", "receipt_path"),
            f"registry[{dataset_id}]",
        )
        leakage_exception = registry_record["leakage_exception"]
        if leakage_exception is not None and (
            not isinstance(leakage_exception, str) or not leakage_exception.strip()
        ):
            raise DatasetContractError(f"registry[{dataset_id}]: invalid leakage_exception")
        for numeric_field in ("expected_bytes", "predicted_peak_bytes"):
            value = registry_record[numeric_field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise DatasetContractError(f"registry[{dataset_id}]: invalid {numeric_field}")
        receipt = receipts_by_id.get(dataset_id)
        if receipt is None:
            raise DatasetContractError(f"receipt[{dataset_id}]: missing D1/D2 receipt")
        _validate_d1_receipt(receipt, dataset_id)
        if receipt["acquisition_level"] != "D2":
            raise DatasetContractError(f"receipt[{dataset_id}]: eligible freeze requires D2 receipt")
        if receipt["predicted_peak_bytes"] != registry_record["predicted_peak_bytes"]:
            raise DatasetContractError(f"receipt[{dataset_id}]: predicted peak disagrees with registry")

        dataset_units = units_by_dataset.get(dataset_id, [])
        by_unit_id = {str(unit["unit_id"]): unit for unit in dataset_units}
        missing_ids = sorted(set(unit_ids) - set(by_unit_id))
        if missing_ids:
            raise DatasetContractError(f"frozen_manifest[{dataset_id}]: unresolved unit IDs {missing_ids[:3]}")
        selected_units = [by_unit_id[unit_id] for unit_id in unit_ids]
        if eligible != len(selected_units):
            raise DatasetContractError(
                f"dataset[{dataset_id}]: attrition eligible count disagrees with frozen IDs"
            )
        for unit in selected_units:
            _validate_d2_unit(unit, dataset_id)
            for field in ("revision", "rights"):
                if unit[field] != registry_record[field]:
                    raise DatasetContractError(
                        f"unit[{dataset_id}/{unit['unit_id']}]: {field} disagrees with registry"
                    )
        selected_splits = {str(unit["split"]) for unit in selected_units}
        if not {"validation", "final_test"}.issubset(selected_splits):
            raise DatasetContractError(
                f"dataset[{dataset_id}]: D2 requires non-empty validation and final_test splits"
            )
        if leakage_exception is None:
            for unit in selected_units:
                _require_nonempty_strings(
                    unit,
                    ("speaker_id", "talk_id"),
                    f"unit[{dataset_id}/{unit['unit_id']}] leakage identity",
                )
        _reject_validation_test_leakage(selected_units, dataset_id)
        if endpoint_by_id[dataset_id].get("scientific_selection_eligible") is not True:
            raise DatasetContractError(f"endpoint_matrix[{dataset_id}]: eligible D2 not exposed")

    return {
        "dataset_count": len(dataset_ids),
        "unit_count": len(units),
        "d2_eligible_dataset_count": eligible_count,
    }

"""Fail-closed validation for D0/D1/D2 dataset evidence bundles."""

from __future__ import annotations

import hashlib
import math
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_PATH = Path(__file__).with_name("dataset_profile_schema.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_D2_PEAK_BYTES = 50 * 1024**3
MAX_D2_HOURS = 12.0
NUMERIC_SUMMARY_FIELDS = {
    "n", "missing_n", "min", "p5", "p25", "median", "p75", "p95",
    "max", "mean", "standard_deviation",
}
UNIVERSAL_D2_FIGURE_IDS = {
    "scale_bytes",
    "duration_distribution",
    "file_size_distribution",
    "missingness_gt_coverage",
    "attrition_waterfall",
    "split_comparison",
    "source_overlap",
    "duplicate_matrix",
}
UNIVERSAL_D2_FIGURE_CONTRACTS = {
    "scale_bytes": ("dataset.cross_dataset_comparison", "bytes", "bar", "selected_bytes"),
    "duration_distribution": ("dataset.statistics", "seconds", "histogram_ecdf", "duration"),
    "file_size_distribution": ("dataset.statistics", "bytes", "histogram_ecdf", "file_size_bytes"),
    "missingness_gt_coverage": ("dataset.quality_issues", "percent", "heatmap", "missingness_gt_coverage"),
    "attrition_waterfall": ("dataset.attrition", "count", "waterfall", "attrition"),
    "split_comparison": ("dataset.split_statistics", "count", "grouped_bar", "split_count"),
    "source_overlap": ("dataset.leakage_audit", "count", "matrix", "source_overlap"),
    "duplicate_matrix": ("dataset.leakage_audit", "count", "matrix", "exact_near_duplicate"),
}


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


def derive_cross_split_overlap_findings(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return deterministic exact-identity findings that occur in multiple splits."""

    tracked = ("unit_id", "sha256", "source", "speaker_id", "talk_id")
    indexes: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        field: defaultdict(list) for field in tracked
    }
    for unit in units:
        split = unit.get("split")
        if split in (None, "", "unknown"):
            continue
        for field in tracked:
            value = unit.get(field)
            if value not in (None, "", "unknown"):
                indexes[field][str(value)].append(unit)

    findings: list[dict[str, Any]] = []
    for field in tracked:
        for value, occurrences in sorted(indexes[field].items()):
            splits = sorted({str(row["split"]) for row in occurrences})
            if len(splits) < 2:
                continue
            findings.append(
                {
                    "field": field,
                    "value": value,
                    "splits": splits,
                    "unit_ids": sorted({str(row.get("unit_id", "")) for row in occurrences}),
                }
            )
    return findings


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_summary(values: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, math.floor((len(ordered) - 1) * fraction))]

    return {
        "n": len(values),
        "missing_n": 0,
        "min": min(values),
        "p5": percentile(0.05),
        "p25": percentile(0.25),
        "median": median(values),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": max(values),
        "mean": mean(values),
        "standard_deviation": pstdev(values),
    }


def _require_nonnegative_int(record: Mapping[str, Any], field: str, label: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetContractError(f"{label}: invalid {field}")
    return value


def _validate_contract_record(
    record: Mapping[str, Any], definition: str, label: str, string_fields: Iterable[str]
) -> None:
    _require_schema_keys(record, definition, label)
    _reject_extra_keys(record, definition, label)
    _require_nonempty_strings(record, string_fields, label)


def validate_dataset_characterization_bundle(
    *,
    registry: Sequence[Mapping[str, Any]],
    feasibility_receipts: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    attrition: Sequence[Mapping[str, Any]],
    frozen_manifest: Sequence[Mapping[str, Any]],
    endpoint_matrix: Sequence[Mapping[str, Any]],
    statistics: Sequence[Mapping[str, Any]],
    split_statistics: Sequence[Mapping[str, Any]],
    quality_issues: Sequence[Mapping[str, Any]],
    leakage_audit: Sequence[Mapping[str, Any]],
    figure_manifest: Sequence[Mapping[str, Any]],
    cross_dataset_comparison: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> dict[str, int]:
    """Validate the semantic 00c characterization package without promoting smoke to D2."""

    summary = validate_dataset_bundle(
        registry=registry,
        feasibility_receipts=feasibility_receipts,
        units=units,
        profiles=profiles,
        attrition=attrition,
        frozen_manifest=frozen_manifest,
        endpoint_matrix=endpoint_matrix,
    )
    registry_by_id = _index(registry, "registry")
    profiles_by_id = _index(profiles, "profiles")
    attrition_by_id = _index(attrition, "attrition")
    frozen_by_id = _index(frozen_manifest, "frozen_manifest")
    statistics_by_id = _index(statistics, "statistics")
    quality_by_id = _index(quality_issues, "quality_issues")
    leakage_by_id = _index(leakage_audit, "leakage_audit")
    dataset_ids = set(profiles_by_id)
    if not dataset_ids:
        raise DatasetContractError("characterization bundle: no datasets")
    comparison_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in cross_dataset_comparison:
        dataset_id = record.get("dataset_id")
        if dataset_id not in dataset_ids:
            raise DatasetContractError("cross_dataset_comparison: unknown dataset_id")
        comparison_by_dataset[str(dataset_id)].append(record)
    for label, indexed in (
        ("statistics", statistics_by_id),
        ("quality_issues", quality_by_id),
        ("leakage_audit", leakage_by_id),
    ):
        if set(indexed) != dataset_ids:
            raise DatasetContractError(f"{label}: dataset IDs do not match profiles")
    if set(comparison_by_dataset) != dataset_ids:
        raise DatasetContractError("cross_dataset_comparison: dataset IDs do not match profiles")

    units_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    splits_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    figures_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for unit in units:
        units_by_dataset[str(unit["dataset_id"])].append(unit)
    unknown_unit_datasets = sorted(set(units_by_dataset) - dataset_ids)
    if unknown_unit_datasets:
        raise DatasetContractError(
            f"unit_profile: unknown dataset IDs {unknown_unit_datasets}"
        )
    for record in split_statistics:
        dataset_id = record.get("dataset_id")
        if dataset_id not in dataset_ids:
            raise DatasetContractError("split_statistics: unknown dataset_id")
        splits_by_dataset[str(dataset_id)].append(record)
    for record in figure_manifest:
        dataset_id = record.get("dataset_id")
        if dataset_id not in dataset_ids:
            raise DatasetContractError("figure_manifest: unknown dataset_id")
        figures_by_dataset[str(dataset_id)].append(record)
    if set(splits_by_dataset) != dataset_ids or set(figures_by_dataset) != dataset_ids:
        raise DatasetContractError("split/figure dataset IDs do not match profiles")

    comparison_contracts: dict[str, tuple[str, str, str]] = {}
    artifact_root = artifact_root.resolve()
    d2_figure_count = 0
    for dataset_id in sorted(dataset_ids):
        dataset_units = units_by_dataset.get(dataset_id, [])
        if not dataset_units:
            raise DatasetContractError(f"dataset[{dataset_id}]: no profiled units")
        profile = profiles_by_id[dataset_id]
        eligible_d2 = frozen_by_id[dataset_id]["freeze_status"] == "eligible"
        if eligible_d2:
            frozen_ids = set(frozen_by_id[dataset_id]["unit_ids"])
            profiled_ids = {str(unit["unit_id"]) for unit in dataset_units}
            if profiled_ids != frozen_ids:
                raise DatasetContractError(
                    f"dataset[{dataset_id}]: unit profile does not match frozen IDs"
                )

        stat = statistics_by_id[dataset_id]
        _validate_contract_record(
            stat,
            "statistics_record",
            f"statistics[{dataset_id}]",
            ("dataset_id", "revision", "acquisition_level", "scope", "source_table_sha256"),
        )
        selected_files = _require_nonnegative_int(stat, "selected_files", f"statistics[{dataset_id}]")
        denominator = _require_nonnegative_int(stat, "denominator", f"statistics[{dataset_id}]")
        selected_bytes = _require_nonnegative_int(stat, "selected_bytes", f"statistics[{dataset_id}]")
        if selected_files != len(dataset_units) or denominator != len(dataset_units):
            raise DatasetContractError(f"statistics[{dataset_id}]: unit denominator mismatch")
        if selected_bytes != sum(int(unit["bytes"]) for unit in dataset_units):
            raise DatasetContractError(f"statistics[{dataset_id}]: selected_bytes mismatch")
        if profile["selected_bytes"] != selected_bytes:
            raise DatasetContractError(f"profile[{dataset_id}]: selected_bytes mismatch")
        numeric = stat.get("file_size_bytes")
        if not isinstance(numeric, Mapping) or set(numeric) != NUMERIC_SUMMARY_FIELDS:
            raise DatasetContractError(f"statistics[{dataset_id}]: incomplete numeric summary")
        expected_numeric = _numeric_summary([int(unit["bytes"]) for unit in dataset_units])
        if dict(numeric) != expected_numeric:
            raise DatasetContractError(f"statistics[{dataset_id}]: numeric summary mismatch")
        extension_counts = stat.get("extension_counts")
        valid_extension_counts = (
            isinstance(extension_counts, Mapping)
            and all(
                isinstance(key, str)
                and key
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for key, value in extension_counts.items()
            )
        )
        if not valid_extension_counts or sum(extension_counts.values()) != len(dataset_units):
            raise DatasetContractError(f"statistics[{dataset_id}]: extension counts mismatch")
        expected_unit_hash = hashlib.sha256(
            json.dumps(dataset_units, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if not SHA256_RE.fullmatch(str(stat["source_table_sha256"])):
            raise DatasetContractError(f"statistics[{dataset_id}]: invalid source table hash")
        if stat["source_table_sha256"] != expected_unit_hash:
            raise DatasetContractError(f"statistics[{dataset_id}]: source table hash mismatch")
        if stat["acquisition_level"] != profile["acquisition_level"] or stat["scope"] != profile["profile"]:
            raise DatasetContractError(f"statistics[{dataset_id}]: acquisition scope mismatch")
        if eligible_d2 and stat["revision"] != registry_by_id[dataset_id]["revision"]:
            raise DatasetContractError(f"statistics[{dataset_id}]: revision disagrees with registry")

        split_rows = splits_by_dataset[dataset_id]
        seen_splits: set[str] = set()
        for row in split_rows:
            _validate_contract_record(
                row,
                "split_statistics_record",
                f"split_statistics[{dataset_id}]",
                ("dataset_id", "split", "revision", "acquisition_level", "scope"),
            )
            split = str(row["split"])
            if split in seen_splits:
                raise DatasetContractError(f"split_statistics[{dataset_id}]: duplicate split {split}")
            seen_splits.add(split)
            _require_nonnegative_int(row, "n", f"split_statistics[{dataset_id}]")
            if _require_nonnegative_int(row, "denominator", f"split_statistics[{dataset_id}]") != len(dataset_units):
                raise DatasetContractError(f"split_statistics[{dataset_id}]: denominator mismatch")
            if any(row[field] != stat[field] for field in ("revision", "acquisition_level", "scope")):
                raise DatasetContractError(f"split_statistics[{dataset_id}]: contract scope mismatch")
        if sum(int(row["n"]) for row in split_rows) != len(dataset_units):
            raise DatasetContractError(f"split_statistics[{dataset_id}]: split counts do not reconcile")
        expected_split_counts = Counter(str(unit.get("split")) for unit in dataset_units)
        reported_split_counts = {str(row["split"]): int(row["n"]) for row in split_rows}
        if reported_split_counts != dict(expected_split_counts):
            raise DatasetContractError(
                f"split_statistics[{dataset_id}]: counts disagree with unit profile"
            )
        if eligible_d2 and not seen_splits.issubset({"train", "validation", "final_test"}):
            raise DatasetContractError(f"split_statistics[{dataset_id}]: invalid D2 split")

        quality = quality_by_id[dataset_id]
        _validate_contract_record(
            quality,
            "quality_record",
            f"quality[{dataset_id}]",
            ("dataset_id", "revision", "acquisition_level", "scope", "issue_policy"),
        )
        if _require_nonnegative_int(quality, "checked_units", f"quality[{dataset_id}]") != len(dataset_units):
            raise DatasetContractError(f"quality[{dataset_id}]: checked denominator mismatch")
        issue_count = _require_nonnegative_int(quality, "issue_count", f"quality[{dataset_id}]")
        excluded_units = _require_nonnegative_int(quality, "excluded_units", f"quality[{dataset_id}]")
        if issue_count != excluded_units or excluded_units != attrition_by_id[dataset_id]["excluded"]:
            raise DatasetContractError(f"quality[{dataset_id}]: exclusions do not reconcile")
        if any(quality[field] != stat[field] for field in ("revision", "acquisition_level", "scope")):
            raise DatasetContractError(f"quality[{dataset_id}]: contract scope mismatch")

        leakage = leakage_by_id[dataset_id]
        _validate_contract_record(
            leakage,
            "leakage_audit_record",
            f"leakage[{dataset_id}]",
            (
                "dataset_id", "revision", "acquisition_level", "scope",
                "near_duplicate_status", "adjudication_status",
            ),
        )
        exact_overlap = _require_nonnegative_int(leakage, "exact_overlap_count", f"leakage[{dataset_id}]")
        duplicate_groups = _require_nonnegative_int(
            leakage, "duplicate_group_count", f"leakage[{dataset_id}]"
        )
        findings = leakage.get("findings")
        if not isinstance(findings, list):
            raise DatasetContractError(f"leakage[{dataset_id}]: findings must be an array")
        expected_overlap_findings = derive_cross_split_overlap_findings(dataset_units)
        if findings != expected_overlap_findings or exact_overlap != len(expected_overlap_findings):
            raise DatasetContractError(f"leakage[{dataset_id}]: exact overlap findings mismatch")
        expected_duplicate_groups = sum(
            count > 1
            for count in Counter(
                str(unit.get("sha256")) for unit in dataset_units if unit.get("sha256")
            ).values()
        )
        if duplicate_groups != expected_duplicate_groups:
            raise DatasetContractError(f"leakage[{dataset_id}]: duplicate groups do not reconcile")
        if any(leakage[field] != stat[field] for field in ("revision", "acquisition_level", "scope")):
            raise DatasetContractError(f"leakage[{dataset_id}]: contract scope mismatch")
        if eligible_d2 and (
            exact_overlap != 0
            or leakage["near_duplicate_status"] != "adjudicated"
            or leakage["adjudication_status"] != "complete"
        ):
            raise DatasetContractError(f"leakage[{dataset_id}]: D2 leakage audit is incomplete")

        figure_ids: set[str] = set()
        figure_paths: set[Path] = set()
        universal_figure_hashes: set[str] = set()
        for figure in figures_by_dataset[dataset_id]:
            _validate_contract_record(
                figure,
                "figure_manifest_record",
                f"figure[{dataset_id}]",
                (
                    "dataset_id", "figure_id", "path", "sha256", "source_table_sha256",
                    "revision", "acquisition_level", "split_scope", "profile", "unit",
                    "missing_policy", "exclusion_policy", "source_artifact", "code_revision",
                ),
            )
            figure_id = str(figure["figure_id"])
            if figure_id in figure_ids:
                raise DatasetContractError(f"figure[{dataset_id}]: duplicate figure_id {figure_id}")
            figure_ids.add(figure_id)
            if _require_nonnegative_int(figure, "n", f"figure[{dataset_id}]") != len(dataset_units):
                raise DatasetContractError(f"figure[{dataset_id}]: unit denominator mismatch")
            if any(figure[field] != stat[field] for field in ("revision", "acquisition_level")):
                raise DatasetContractError(f"figure[{dataset_id}]: contract scope mismatch")
            if figure["profile"] != stat["scope"]:
                raise DatasetContractError(f"figure[{dataset_id}]: profile scope mismatch")
            for hash_field in ("sha256", "source_table_sha256", "code_revision"):
                if not SHA256_RE.fullmatch(str(figure[hash_field])):
                    raise DatasetContractError(f"figure[{dataset_id}]: invalid {hash_field}")
            source_payloads = {
                "dataset.statistics": stat,
                "dataset.split_statistics": split_rows,
                "dataset.quality_issues": quality,
                "dataset.attrition": attrition_by_id[dataset_id],
                "dataset.leakage_audit": leakage,
                "dataset.cross_dataset_comparison": cross_dataset_comparison,
            }
            source_artifact = str(figure["source_artifact"])
            if source_artifact not in source_payloads:
                raise DatasetContractError(f"figure[{dataset_id}]: unknown source_artifact")
            plot_config = figure.get("plot_config")
            if not isinstance(plot_config, Mapping) or not plot_config:
                raise DatasetContractError(f"figure[{dataset_id}]: invalid plot_config")
            if any(
                not isinstance(plot_config.get(field), str) or not plot_config[field]
                for field in ("kind", "measure")
            ):
                raise DatasetContractError(f"figure[{dataset_id}]: invalid plot_config")
            if figure_id in UNIVERSAL_D2_FIGURE_CONTRACTS:
                required_source, required_unit, required_kind, required_measure = (
                    UNIVERSAL_D2_FIGURE_CONTRACTS[figure_id]
                )
                if (
                    source_artifact != required_source
                    or figure["unit"] != required_unit
                    or plot_config.get("kind") != required_kind
                    or plot_config.get("measure") != required_measure
                ):
                    raise DatasetContractError(
                        f"figure[{dataset_id}/{figure_id}]: universal semantic contract mismatch"
                    )
            expected_source_hash = hashlib.sha256(
                json.dumps(
                    source_payloads[source_artifact], sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            if figure["source_table_sha256"] != expected_source_hash:
                raise DatasetContractError(f"figure[{dataset_id}]: source table hash mismatch")
            figure_path = (artifact_root / str(figure["path"])).resolve()
            try:
                figure_path.relative_to(artifact_root)
            except ValueError as exc:
                raise DatasetContractError(f"figure[{dataset_id}]: path escapes artifact root") from exc
            if figure_path in figure_paths:
                raise DatasetContractError(f"figure[{dataset_id}]: duplicate figure path")
            figure_paths.add(figure_path)
            if figure_id in UNIVERSAL_D2_FIGURE_CONTRACTS:
                if figure["sha256"] in universal_figure_hashes:
                    raise DatasetContractError(
                        f"figure[{dataset_id}/{figure_id}]: duplicate universal figure content"
                    )
                universal_figure_hashes.add(str(figure["sha256"]))
            if not figure_path.is_file() or _sha256_file(figure_path) != figure["sha256"]:
                raise DatasetContractError(f"figure[{dataset_id}]: missing or hash mismatch")
        if eligible_d2:
            missing_figures = UNIVERSAL_D2_FIGURE_IDS - figure_ids
            if missing_figures:
                raise DatasetContractError(f"figure[{dataset_id}]: missing universal figures {sorted(missing_figures)}")
            d2_figure_count += len(figure_ids)

        seen_metric_ids: set[str] = set()
        for comparison in comparison_by_dataset[dataset_id]:
            _validate_contract_record(
                comparison,
                "cross_dataset_comparison_record",
                f"comparison[{dataset_id}]",
                (
                    "dataset_id", "metric_id", "definition_id", "unit", "revision",
                    "acquisition_level", "scope", "source_table_sha256",
                ),
            )
            value = comparison.get("value")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise DatasetContractError(f"comparison[{dataset_id}]: invalid value")
            if not SHA256_RE.fullmatch(str(comparison["source_table_sha256"])):
                raise DatasetContractError(f"comparison[{dataset_id}]: invalid source table hash")
            if comparison["source_table_sha256"] != stat["source_table_sha256"]:
                raise DatasetContractError(f"comparison[{dataset_id}]: source table hash mismatch")
            if any(comparison[field] != stat[field] for field in ("revision", "acquisition_level", "scope")):
                raise DatasetContractError(f"comparison[{dataset_id}]: contract scope mismatch")
            if _require_nonnegative_int(comparison, "denominator", f"comparison[{dataset_id}]") != len(dataset_units):
                raise DatasetContractError(f"comparison[{dataset_id}]: denominator mismatch")
            contract = (str(comparison["definition_id"]), str(comparison["unit"]), str(comparison["scope"]))
            metric_id = str(comparison["metric_id"])
            if metric_id in seen_metric_ids:
                raise DatasetContractError(f"comparison[{dataset_id}]: duplicate metric_id {metric_id}")
            seen_metric_ids.add(metric_id)
            prior = comparison_contracts.setdefault(metric_id, contract)
            if prior != contract:
                raise DatasetContractError(f"comparison[{dataset_id}]: incompatible metric definition")
            if metric_id == "selected_bytes" and comparison["value"] != selected_bytes:
                raise DatasetContractError(f"comparison[{dataset_id}]: value disagrees with statistics")

    return {
        **summary,
        "characterization_artifact_count": (
            len(statistics) + len(split_statistics) + len(quality_issues)
            + len(leakage_audit) + len(figure_manifest) + len(cross_dataset_comparison)
        ),
        "d2_figure_count": d2_figure_count,
    }

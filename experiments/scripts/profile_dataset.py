"""Build real-data smoke profiles and a clearly scoped frozen subset."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any, Sequence

from experiments.scripts.benchmark_artifacts import (
    parse_context,
    read_json_artifact,
    write_json_artifact,
    write_run_manifest,
    write_text_artifact,
)
from experiments.scripts.dataset_profile_helpers import (
    numeric_summary,
    split_for,
    write_size_figure,
)
from experiments.evaluation.dataset_contracts import (
    derive_cross_split_overlap_findings,
    validate_dataset_characterization_bundle,
)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        context = parse_context(__doc__ or "", "00c_dataset_eda_and_freeze", argv)
        feasibility = read_json_artifact(context, "dataset.feasibility_receipts")["records"]
        inventory = read_json_artifact(context, "dataset.acquisition_inventory")["records"]
        profileable_ids = {
            row["dataset_id"] for row in feasibility if row.get("eligible_for_profiling")
        }
        inventory = [row for row in inventory if row["dataset_id"] in profileable_ids]
        if not inventory:
            raise RuntimeError("no readable dataset inventory available")

        units: list[dict[str, Any]] = []
        profiles: list[dict[str, Any]] = []
        statistics: list[dict[str, Any]] = []
        split_statistics: list[dict[str, Any]] = []
        quality: list[dict[str, Any]] = []
        attrition: list[dict[str, Any]] = []
        checksum_index: dict[str, list[dict[str, str]]] = defaultdict(list)
        dataset_contract_meta: dict[str, dict[str, Any]] = {}

        for dataset in inventory:
            dataset_units = []
            for item in dataset["files"]:
                unit = {
                    "dataset_id": dataset["dataset_id"],
                    "unit_id": item["unit_id"],
                    "relative_path": item["relative_path"],
                    "split": split_for(item["relative_path"]),
                    "bytes": item["bytes"],
                    "extension": item["extension"],
                    "sha256": item["sha256"],
                    "readable": item["readable"],
                    "decoded": item["decoded"],
                    "probe_status": item["probe_status"],
                }
                units.append(unit)
                dataset_units.append(unit)
                checksum_index[unit["sha256"]].append(
                    {"dataset_id": unit["dataset_id"], "unit_id": unit["unit_id"], "split": unit["split"]}
                )
            sizes = [unit["bytes"] for unit in dataset_units]
            source_table_sha256 = hashlib.sha256(
                json.dumps(dataset_units, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            dataset_revision = f"local-inventory-{source_table_sha256[:12]}"
            dataset_contract_meta[dataset["dataset_id"]] = {
                "revision": dataset_revision,
                "source_table_sha256": source_table_sha256,
                "acquisition_level": dataset["acquisition_level"],
            }
            split_counts = Counter(unit["split"] for unit in dataset_units)
            issue_count = sum(not unit["readable"] or unit["bytes"] <= 0 for unit in dataset_units)
            eligible_count = len(dataset_units) - issue_count
            excluded_count = dataset["selected_files"] - len(dataset_units) + issue_count
            profiles.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "source": dataset["source"],
                    "acquisition_level": dataset["acquisition_level"],
                    "profile": context.profile,
                    "raw_file_count": dataset["total_files"],
                    "eligible_file_count": eligible_count,
                    "excluded_file_count": excluded_count,
                    "unresolved_file_count": dataset["total_files"] - dataset["selected_files"],
                    "selected_bytes": sum(sizes),
                    "selection_policy": dataset["selection_policy"],
                    "ground_truth_status": "unknown_not_adjudicated",
                }
            )
            statistics.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "selected_files": len(dataset_units),
                    "selected_bytes": sum(sizes),
                    "file_size_bytes": numeric_summary(sizes),
                    "extension_counts": dict(Counter(unit["extension"] for unit in dataset_units)),
                    "revision": dataset_revision,
                    "acquisition_level": dataset["acquisition_level"],
                    "scope": context.profile,
                    "denominator": len(dataset_units),
                    "source_table_sha256": source_table_sha256,
                }
            )
            split_statistics.extend(
                {
                    "dataset_id": dataset["dataset_id"],
                    "split": split,
                    "n": count,
                    "denominator": len(dataset_units),
                    "revision": dataset_revision,
                    "acquisition_level": dataset["acquisition_level"],
                    "scope": context.profile,
                }
                for split, count in sorted(split_counts.items())
            )
            quality.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "checked_units": len(dataset_units),
                    "issue_count": excluded_count,
                    "excluded_units": excluded_count,
                    "revision": dataset_revision,
                    "acquisition_level": dataset["acquisition_level"],
                    "scope": context.profile,
                    "issue_policy": "unreadable or zero-byte selected files are excluded",
                }
            )
            attrition.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "raw": dataset["total_files"],
                    "eligible": eligible_count,
                    "excluded": excluded_count,
                    "unresolved": dataset["total_files"] - dataset["selected_files"],
                    "scope": context.profile,
                }
            )

        leakage = []
        for profile in profiles:
            dataset_id = profile["dataset_id"]
            dataset_units = [unit for unit in units if unit["dataset_id"] == dataset_id]
            duplicate_groups = [
                checksum
                for checksum, occurrences in sorted(checksum_index.items())
                if len([row for row in occurrences if row["dataset_id"] == dataset_id]) > 1
            ]
            overlap_findings = derive_cross_split_overlap_findings(dataset_units)
            leakage.append(
                {
                    "dataset_id": dataset_id,
                    "revision": dataset_contract_meta[dataset_id]["revision"],
                    "acquisition_level": profile["acquisition_level"],
                    "scope": context.profile,
                    "exact_overlap_count": len(overlap_findings),
                    "duplicate_group_count": len(duplicate_groups),
                    "near_duplicate_status": "not_run",
                    "adjudication_status": "not_applicable_smoke",
                    "findings": overlap_findings,
                }
            )

        comparison = [
            {
                "dataset_id": row["dataset_id"],
                "metric_id": "selected_bytes",
                "value": row["selected_bytes"],
                "definition_id": "physical_selected_file_bytes_v1",
                "unit": "bytes",
                "revision": row["revision"],
                "acquisition_level": row["acquisition_level"],
                "scope": context.profile,
                "denominator": row["denominator"],
                "source_table_sha256": row["source_table_sha256"],
            }
            for row in statistics
        ]
        base_figure = write_size_figure(
            context,
            [
                {"dataset_id": row["dataset_id"], "selected_bytes": row["value"]}
                for row in comparison
            ],
        )
        figure_manifest = [
            {
                **base_figure,
                "dataset_id": row["dataset_id"],
                "revision": row["revision"],
                "acquisition_level": row["acquisition_level"],
                "split_scope": "all_profiled_splits",
                "n": row["denominator"],
                "missing_policy": "missing values reported, never imputed",
                "exclusion_policy": "unreadable or zero-byte selected files excluded",
                "source_artifact": "dataset.cross_dataset_comparison",
                "source_table_sha256": hashlib.sha256(
                    json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "code_revision": context.dependency_state_sha256,
                "plot_config": {
                    "kind": "bar",
                    "measure": "selected_bytes",
                    "x": "dataset_id",
                    "y": "selected_bytes",
                },
            }
            for row in comparison
        ]
        endpoint_matrix = [
            {
                "dataset_id": row["dataset_id"],
                "endpoint": "dataset_characterization",
                "eligibility": "eligible_smoke" if context.profile == "smoke" else "eligible_full_inventory",
                "scientific_selection_eligible": False,
                "reason": "local inventory only; task GT eligibility remains unadjudicated",
            }
            for row in profiles
        ]
        frozen_scope = [
            {
                "dataset_id": row["dataset_id"],
                "freeze_status": "not_eligible",
                "scientific_selection_eligible": False,
                "unit_ids": [],
                "reason": "smoke profiling does not adjudicate rights, task ground truth or media decode",
                "profile": context.profile,
            }
            for row in profiles
        ]

        validate_dataset_characterization_bundle(
            registry=profiles,
            feasibility_receipts=feasibility,
            units=units,
            profiles=profiles,
            attrition=attrition,
            frozen_manifest=frozen_scope,
            endpoint_matrix=endpoint_matrix,
            statistics=statistics,
            split_statistics=split_statistics,
            quality_issues=quality,
            leakage_audit=leakage,
            figure_manifest=figure_manifest,
            cross_dataset_comparison=comparison,
            artifact_root=context.artifact_root,
        )

        write_json_artifact(context, "dataset.profile", profiles)
        write_json_artifact(context, "dataset.unit_profile", units)
        write_json_artifact(context, "dataset.statistics", statistics)
        write_json_artifact(context, "dataset.split_statistics", split_statistics)
        write_json_artifact(context, "dataset.quality_issues", quality)
        write_json_artifact(context, "dataset.attrition", attrition)
        write_json_artifact(context, "dataset.leakage_audit", leakage)
        write_json_artifact(context, "dataset.figure_manifest", figure_manifest)
        write_json_artifact(
            context,
            "dataset.frozen_manifest",
            frozen_scope,
            metadata={
                "scope": context.profile,
                "not_final_test": True,
                "profiled_unit_count": len(units),
            },
        )
        write_json_artifact(context, "dataset.registry", profiles)
        write_json_artifact(context, "dataset.endpoint_matrix", endpoint_matrix)
        write_json_artifact(context, "dataset.cross_dataset_comparison", comparison)
        report_lines = [
            f"# Dataset EDA {context.profile} report",
            "",
            f"Profile: `{context.profile}`. This artifact is real local-file evidence, not a model benchmark result.",
            "",
            "| Dataset | Selected files | Selected bytes | GT status |",
            "|---|---:|---:|---|",
        ]
        report_lines.extend(
            f"| {row['dataset_id']} | {row['eligible_file_count']} | {row['selected_bytes']} | {row['ground_truth_status']} |"
            for row in profiles
        )
        write_text_artifact(context, "dataset.eda_report", "\n".join(report_lines))
        write_run_manifest(context)
        print(f"[OK] dataset profile: {len(profiles)} datasets, {len(units)} real files")
        return 0
    except Exception as exc:
        print(f"[FAIL] dataset profile: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

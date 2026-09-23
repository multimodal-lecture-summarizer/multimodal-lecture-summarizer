"""Build real-data smoke profiles and a clearly scoped frozen subset."""

from __future__ import annotations

from collections import Counter, defaultdict
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
from experiments.evaluation.dataset_contracts import validate_dataset_bundle


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
                }
            )
            split_statistics.extend(
                {"dataset_id": dataset["dataset_id"], "split": split, "n": count}
                for split, count in sorted(split_counts.items())
            )
            quality.append(
                {"dataset_id": dataset["dataset_id"], "checked_units": len(dataset_units), "issue_count": issue_count}
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
        for checksum, occurrences in sorted(checksum_index.items()):
            if len(occurrences) > 1:
                leakage.append({"sha256": checksum, "duplicate_count": len(occurrences), "occurrences": occurrences})
        if not leakage:
            leakage = [{"sha256": None, "duplicate_count": 0, "occurrences": [], "scope": context.profile}]

        comparison = [
            {
                "dataset_id": row["dataset_id"],
                "selected_files": row["selected_files"],
                "selected_bytes": row["selected_bytes"],
                "profile": context.profile,
            }
            for row in statistics
        ]
        figure_manifest = [write_size_figure(context, comparison)]
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

        validate_dataset_bundle(
            registry=profiles,
            feasibility_receipts=feasibility,
            units=units,
            profiles=profiles,
            attrition=attrition,
            frozen_manifest=frozen_scope,
            endpoint_matrix=endpoint_matrix,
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

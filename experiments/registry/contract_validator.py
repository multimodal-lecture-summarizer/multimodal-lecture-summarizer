"""Validate frozen research, metric, and notebook orchestration contracts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = Path(__file__).resolve().parent
DEFAULT_DECISIONS = REGISTRY_DIR / "research_decisions.yaml"
DEFAULT_METRICS = REGISTRY_DIR / "metric_registry.yaml"
DEFAULT_NOTEBOOKS = PROJECT_ROOT / "experiments" / "notebooks" / "notebook_manifest.yaml"

REQUIRED_TRACKS = {
    "asr_english",
    "scene_boundary",
    "keyframe_visual_evidence",
    "ocr_detection",
    "ocr_recognition",
    "ocr_end_to_end_spotting",
    "caption_multireference",
    "caption_grounded",
    "object_detection_closed",
    "object_detection_open",
    "tracking_fixed_detector",
    "chaptering",
    "text_summary_reference",
    "visual_summary_keyshots",
    "multimodal_summary_package",
    "retrieval",
    "qa_short_answer",
    "end_to_end_system",
}

EDA_OUTPUTS = {
    "dataset.profile",
    "dataset.unit_profile",
    "dataset.statistics",
    "dataset.split_statistics",
    "dataset.quality_issues",
    "dataset.attrition",
    "dataset.leakage_audit",
    "dataset.figure_manifest",
    "dataset.frozen_manifest",
    "dataset.registry",
    "dataset.endpoint_matrix",
    "dataset.cross_dataset_comparison",
    "dataset.eda_report",
}

ALLOWED_METRIC_DIRECTIONS = {"higher", "lower"}
ALLOWED_FAILURE_POLICIES = {"count_failure_separately", "not_eligible"}
ALLOWED_NOTEBOOK_PROFILES = {"smoke", "full"}
ALLOWED_NETWORK_POLICIES = {"none", "bounded_acquisition", "registry_controlled"}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Contract not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Contract root must be a mapping: {path}")
    return payload


def _value(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _require_values(data: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    return [
        f"{path}: expected {wanted!r}, got {_value(data, path)!r}"
        for path, wanted in expected.items()
        if _value(data, path) != wanted
    ]


def validate_research_decisions(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_sections: dict[str, Any] = {
        "schema_version": 1,
        "decision_status": "frozen",
        "language": {
            "primary": "en",
            "benchmark_track": "english",
            "non_english_policy": "separate_track_only",
            "asr_primary_metric": "wer",
            "asr_secondary_metrics": ["cer", "macro_talk_wer"],
            "normalization_contract": "english_v1_to_freeze_in_phase2",
        },
        "outputs": {
            "summary_package": "multimodal_text_and_visual",
            "shared_event_evidence_timeline": True,
            "derived_dataset_export": True,
            "derived_dataset_field_roles": ["source_metadata", "ground_truth", "prediction", "evidence"],
            "predictions_may_be_ground_truth": False,
        },
        "summary_budgets": {
            "short": 0.05,
            "medium": 0.10,
            "long": 0.20,
            "text_denominator": "transcript_word_count",
            "visual_denominator": "source_video_duration",
        },
        "ground_truth": {
            "policy": "official_or_preexisting_labels_only",
            "project_manual_annotation_allowed": False,
            "pseudo_labels_allowed_for_official_evaluation": False,
            "heuristic_references_allowed_for_official_evaluation": False,
            "missing_label_action": "fallback_or_not_eligible",
        },
        "speaker_diarization": {
            "scope": "future_ready_only",
            "official_benchmark": False,
            "selection_gate": False,
            "deferred_metrics": ["der", "jer", "cpwer"],
        },
        "compute_profiles": [
            {"id": "colab_t4", "purpose": "heavy_tasks", "accelerator": "nvidia_t4", "leaderboard_scope": "separate"},
            {"id": "local_rtx4050_6gb", "purpose": "light_tasks_and_local_smoke", "accelerator": "nvidia_rtx_4050", "vram_gb": 6, "leaderboard_scope": "separate"},
        ],
        "model_policy": {
            "default": "open_source",
            "llm_api_exception": {
                "allowed_providers": ["groq", "openrouter"],
                "stage_policy": "llm_stages_only",
                "justification_required": True,
                "exact_model_and_date_required": True,
                "token_cost_rate_limit_cap_required": True,
                "deterministic_fallback_required": True,
                "missing_cap_status": "blocked_decision",
                "compare_api_and_local_efficiency_together": False,
            },
        },
        "large_dataset_policy": {
            "vista_full_local_status": "not_feasible_local",
            "allowed_alternatives": ["labeled_subset", "registered_fallback_dataset", "separately_approved_external_storage_profile"],
        },
    }
    allowed_top_level = set(expected_sections) | {"frozen_at"}
    if set(data) != allowed_top_level:
        errors.append(f"top-level decision keys must be {sorted(allowed_top_level)}, got {sorted(data)}")
    for section, expected in expected_sections.items():
        actual = data.get(section)
        if isinstance(expected, dict) and isinstance(actual, dict):
            if set(actual) != set(expected):
                errors.append(f"{section}: keys must be {sorted(expected)}, got {sorted(actual)}")
            for field, wanted in expected.items():
                if actual.get(field) != wanted:
                    errors.append(f"{section}.{field}: expected {wanted!r}, got {actual.get(field)!r}")
        elif actual != expected:
            errors.append(f"{section}: expected {expected!r}, got {actual!r}")
    if not isinstance(data.get("frozen_at"), str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", data["frozen_at"]):
        errors.append("frozen_at must be an ISO YYYY-MM-DD string")
    return errors


def validate_metric_registry(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tracks = data.get("tracks")
    expected_top = {"schema_version", "registry_status", "allowed_directions", "allowed_failure_policies", "tracks"}
    if set(data) != expected_top:
        errors.append(f"metric registry top-level keys must be {sorted(expected_top)}")
    if data.get("schema_version") != 1 or data.get("registry_status") != "phase2_parameters_pending" or not isinstance(tracks, list):
        return ["metric registry requires schema_version=1 and a tracks list"]
    ids: set[str] = set()
    directions = set(data.get("allowed_directions", []))
    failure_policies = set(data.get("allowed_failure_policies", []))
    if directions != ALLOWED_METRIC_DIRECTIONS:
        errors.append(f"allowed_directions must be exactly {sorted(ALLOWED_METRIC_DIRECTIONS)}")
    if failure_policies != ALLOWED_FAILURE_POLICIES:
        errors.append(f"allowed_failure_policies must be exactly {sorted(ALLOWED_FAILURE_POLICIES)}")
    allowed_track_keys = {
        "id", "task_id", "gt_type", "primary_metric", "secondary_metrics", "diagnostic_metrics",
        "scorer", "aggregation_unit", "failure_policy", "practical_threshold", "hard_gates",
        "selection_mode", "metric_vector",
    }
    for index, track in enumerate(tracks):
        prefix = f"tracks[{index}]"
        if not isinstance(track, dict):
            errors.append(f"{prefix}: must be a mapping")
            continue
        track_id = str(track.get("id", ""))
        extra_keys = set(track) - allowed_track_keys
        if extra_keys:
            errors.append(f"{track_id or prefix}: unexpected keys {sorted(extra_keys)}")
        if not track_id or track_id in ids:
            errors.append(f"{prefix}: empty or duplicate id {track_id!r}")
        ids.add(track_id)
        primary = track.get("primary_metric")
        if track_id == "multimodal_summary_package":
            if primary is not None or track.get("selection_mode") != "pareto_or_gate":
                errors.append(f"{track_id}: must be a non-scalar pareto_or_gate track")
            vector = track.get("metric_vector")
            if not isinstance(vector, list) or len(vector) < 2 or not all(isinstance(item, str) and item for item in vector):
                errors.append(f"{track_id}: metric_vector requires text and visual components")
        elif not isinstance(primary, dict) or set(primary) != {"name", "direction"} or not primary.get("name"):
            errors.append(f"{track_id}: exactly one primary_metric name/direction mapping is required")
        elif primary.get("direction") not in ALLOWED_METRIC_DIRECTIONS:
            errors.append(f"{track_id}: invalid primary direction {primary.get('direction')!r}")
        if track.get("failure_policy") not in ALLOWED_FAILURE_POLICIES:
            errors.append(f"{track_id}: invalid failure_policy {track.get('failure_policy')!r}")
        scorer = track.get("scorer")
        if not isinstance(scorer, dict) or set(scorer) != {"package", "version", "parameters"}:
            errors.append(f"{track_id}: scorer must contain only package/version/parameters")
        elif not scorer.get("package") or scorer.get("version") in (None, "") or not isinstance(scorer.get("parameters"), dict):
            errors.append(f"{track_id}: non-empty scorer package/version and parameters mapping required")
        if not track.get("task_id") or not track.get("gt_type") or not track.get("aggregation_unit"):
            errors.append(f"{track_id}: task_id, gt_type and aggregation_unit required")
        for field in ("secondary_metrics", "diagnostic_metrics"):
            values = track.get(field)
            if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values) or len(values) != len(set(values)):
                errors.append(f"{track_id}: {field} must be a unique non-empty-string list")
        if track.get("practical_threshold") in (None, ""):
            errors.append(f"{track_id}: practical_threshold required")
        if "hard_gates" in track and (
            not isinstance(track["hard_gates"], list)
            or not all(isinstance(item, str) and item for item in track["hard_gates"])
        ):
            errors.append(f"{track_id}: hard_gates must be a non-empty-string list")
    missing = REQUIRED_TRACKS - ids
    extra = ids - REQUIRED_TRACKS
    if missing:
        errors.append(f"metric registry missing tracks: {sorted(missing)}")
    if extra:
        errors.append(f"metric registry has unexpected tracks: {sorted(extra)}")
    by_id = {track.get("id"): track for track in tracks if isinstance(track, dict)}
    if _value(by_id.get("asr_english", {}), "primary_metric.name") != "corpus_wer":
        errors.append("asr_english primary metric must be corpus_wer")
    multimodal = by_id.get("multimodal_summary_package", {})
    if _value(multimodal, "scorer.parameters.scalar_merge_forbidden") is not True:
        errors.append("multimodal summary must forbid scalar metric merging")
    return errors


def _cycle_nodes(edges: dict[str, set[str]]) -> set[str]:
    remaining = {node: set(deps) for node, deps in edges.items()}
    while True:
        ready = {node for node, deps in remaining.items() if not deps}
        if not ready:
            return set(remaining)
        for node in ready:
            remaining.pop(node)
        for deps in remaining.values():
            deps.difference_update(ready)
        if not remaining:
            return set()


def validate_notebook_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    notebooks = data.get("notebooks")
    expected_top = {
        "schema_version", "manifest_status", "official_default_profile", "allowed_profiles",
        "allowed_network_policies", "notebooks",
    }
    if set(data) != expected_top:
        errors.append(f"notebook manifest top-level keys must be {sorted(expected_top)}")
    if data.get("schema_version") != 1 or data.get("manifest_status") != "infrastructure_contract" or not isinstance(notebooks, list):
        return ["notebook manifest requires schema_version=1 and a notebooks list"]
    if len(notebooks) != 22:
        errors.append(f"notebook manifest must contain 22 entries, got {len(notebooks)}")
    ids: set[str] = set()
    paths: set[str] = set()
    producer: dict[str, str] = {}
    allowed_profiles = set(data.get("allowed_profiles", []))
    allowed_network = set(data.get("allowed_network_policies", []))
    if allowed_profiles != ALLOWED_NOTEBOOK_PROFILES:
        errors.append(f"allowed_profiles must be exactly {sorted(ALLOWED_NOTEBOOK_PROFILES)}")
    if allowed_network != ALLOWED_NETWORK_POLICIES:
        errors.append(f"allowed_network_policies must be exactly {sorted(ALLOWED_NETWORK_POLICIES)}")
    allowed_entry_keys = {
        "id", "group", "task_id", "path", "runner", "responsibility", "requires", "produces",
        "profiles", "timeout_seconds", "network_policy", "expected_schema_version",
    }
    task_ids: set[str] = set()
    for entry in notebooks:
        if not isinstance(entry, dict):
            errors.append("notebook entry must be a mapping")
            continue
        notebook_id = str(entry.get("id", ""))
        path = str(entry.get("path", ""))
        if set(entry) != allowed_entry_keys:
            errors.append(f"{notebook_id or '<unknown>'}: notebook keys must be {sorted(allowed_entry_keys)}")
        if not re.fullmatch(r"\d{2}[a-z]?_[a-z0-9_]+", notebook_id) or notebook_id in ids:
            errors.append(f"empty or duplicate notebook id {notebook_id!r}")
        if not re.fullmatch(r"experiments/notebooks/[a-z0-9_]+\.ipynb", path) or path in paths:
            errors.append(f"{notebook_id}: empty or duplicate path {path!r}")
        ids.add(notebook_id)
        paths.add(path)
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in task_ids:
            errors.append(f"{notebook_id}: task_id must be a unique non-empty string")
        task_ids.add(task_id)
        if not isinstance(entry.get("runner"), str) or not re.fullmatch(r"[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)+", entry["runner"]):
            errors.append(f"{notebook_id}: runner must be a dotted module path")
        if not isinstance(entry.get("responsibility"), str) or not entry["responsibility"].strip():
            errors.append(f"{notebook_id}: task_id, runner and responsibility required")
        if not isinstance(entry.get("group"), str) or not re.fullmatch(r"\d{2}", entry["group"]):
            errors.append(f"{notebook_id}: group must be a two-digit string")
        if not set(entry.get("profiles", [])) <= ALLOWED_NOTEBOOK_PROFILES or not entry.get("profiles"):
            errors.append(f"{notebook_id}: invalid profiles")
        if entry.get("network_policy") not in ALLOWED_NETWORK_POLICIES:
            errors.append(f"{notebook_id}: invalid network policy")
        if entry.get("expected_schema_version") != 1 or int(entry.get("timeout_seconds", 0)) <= 0:
            errors.append(f"{notebook_id}: schema version 1 and positive timeout required")
        for field in ("requires", "produces"):
            values = entry.get(field)
            if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values) or len(values) != len(set(values)):
                errors.append(f"{notebook_id}: {field} must be a unique non-empty-string list")
        if not entry.get("produces"):
            errors.append(f"{notebook_id}: produces cannot be empty")
        for artifact in entry.get("produces", []):
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            if artifact in producer:
                errors.append(f"output collision: {artifact} from {producer[artifact]} and {notebook_id}")
            producer[artifact] = notebook_id
    valid_entries = [entry for entry in notebooks if isinstance(entry, dict)]
    edges = {entry.get("id", ""): set() for entry in valid_entries}
    for entry in valid_entries:
        for artifact in entry.get("requires", []):
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            if artifact not in producer:
                errors.append(f"{entry.get('id')}: unresolved required artifact {artifact}")
            elif producer[artifact] != entry.get("id"):
                edges[entry.get("id", "")].add(producer[artifact])
    cycles = _cycle_nodes(edges)
    if cycles:
        errors.append(f"notebook dependency cycle: {sorted(cycles)}")
    eda = next((entry for entry in valid_entries if entry.get("id") == "00c_dataset_eda_and_freeze"), {})
    missing_eda = EDA_OUTPUTS - set(eda.get("produces", []))
    if missing_eda:
        errors.append(f"00c missing required EDA outputs: {sorted(missing_eda)}")
    if data.get("official_default_profile") != "full":
        errors.append("official_default_profile must be full")
    return errors


def validate_contracts(
    decisions_path: Path = DEFAULT_DECISIONS,
    metrics_path: Path = DEFAULT_METRICS,
    notebooks_path: Path = DEFAULT_NOTEBOOKS,
) -> list[str]:
    validators: Iterable[tuple[str, Path, Any]] = (
        ("research_decisions", decisions_path, validate_research_decisions),
        ("metric_registry", metrics_path, validate_metric_registry),
        ("notebook_manifest", notebooks_path, validate_notebook_manifest),
    )
    errors: list[str] = []
    for name, path, validator in validators:
        try:
            errors.extend(f"{name}: {error}" for error in validator(load_yaml(path)))
        except Exception as exc:
            errors.append(f"{name}: load failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--notebooks", type=Path, default=DEFAULT_NOTEBOOKS)
    args = parser.parse_args()
    errors = validate_contracts(args.decisions, args.metrics, args.notebooks)
    if errors:
        print(f"[FAIL] {len(errors)} contract error(s)")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("[OK] Research, metric, and notebook contracts are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

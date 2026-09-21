"""Execute or inspect the notebook DAG declared by notebook_manifest.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import nbformat
import yaml
from nbclient import NotebookClient

from experiments.registry.contract_validator import (
    DEFAULT_NOTEBOOKS,
    load_yaml,
    validate_notebook_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "executed_notebooks"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "artifacts"
ELIGIBLE_ARTIFACT_STATUS = "real"
EVIDENCE_STATUSES = {
    "real", "dry", "proxy", "partial", "failed", "not_run", "not_eligible", "blocked_decision"
}
USABLE_DIAGNOSTIC_STATUSES = {"real", "dry", "proxy", "partial"}


class ManifestError(ValueError):
    """Raised when the notebook execution contract is invalid."""


def load_manifest(path: Path = DEFAULT_NOTEBOOKS) -> dict[str, Any]:
    """Load and fail closed on an invalid notebook manifest."""
    try:
        manifest = load_yaml(path)
        errors = validate_notebook_manifest(manifest)
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError, KeyError) as exc:
        raise ManifestError(f"Cannot load notebook manifest {path}: {exc}") from exc
    if errors:
        raise ManifestError("Invalid notebook manifest:\n- " + "\n- ".join(errors))
    return manifest


def _producer_map(entries: Sequence[dict[str, Any]]) -> dict[str, str]:
    return {
        artifact: entry["id"]
        for entry in entries
        for artifact in entry["produces"]
    }


def topological_order(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable manifest-order topological sort."""
    by_id = {entry["id"]: entry for entry in entries}
    producers = _producer_map(entries)
    dependencies = {
        entry["id"]: {
            producers[artifact]
            for artifact in entry["requires"]
            if artifact in producers and producers[artifact] != entry["id"]
        }
        for entry in entries
    }
    ordered: list[dict[str, Any]] = []
    remaining = set(by_id)
    while remaining:
        notebook_id = next(
            (
                entry["id"]
                for entry in entries
                if entry["id"] in remaining and not (dependencies[entry["id"]] & remaining)
            ),
            None,
        )
        if notebook_id is None:
            raise ManifestError(f"Notebook dependency cycle: {sorted(remaining)}")
        ordered.append(by_id[notebook_id])
        remaining.remove(notebook_id)
    return ordered


def select_entries(
    manifest: dict[str, Any],
    *,
    notebook: str | None = None,
    group: str | None = None,
    task: str | None = None,
    profile: str | None = None,
) -> list[dict[str, Any]]:
    """Select manifest entries without maintaining a second notebook list."""
    selected = list(manifest["notebooks"])
    if notebook:
        selected = [
            entry
            for entry in selected
            if notebook in {entry["id"], entry["path"], Path(entry["path"]).name}
        ]
    elif group:
        selected = [entry for entry in selected if entry["group"] == group]
    elif task:
        selected = [entry for entry in selected if entry["task_id"] == task]
    if profile:
        selected = [entry for entry in selected if profile in entry["profiles"]]
    if not selected:
        selector = notebook or group or task or profile
        raise ManifestError(f"No notebook matches selector {selector!r}")
    return topological_order(selected)


def build_plan(
    entries: Sequence[dict[str, Any]],
    profile: str,
    all_entries: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a serializable, side-effect-free execution plan."""
    producers = _producer_map(all_entries or load_manifest()["notebooks"])
    return [
        {
            "order": index,
            "id": entry["id"],
            "task_id": entry["task_id"],
            "group": entry["group"],
            "profile": profile,
            "path": entry["path"],
            "source_exists": (PROJECT_ROOT / entry["path"]).is_file(),
            "timeout_seconds": entry["timeout_seconds"],
            "network_policy": entry["network_policy"],
            "disk_budget": {
                "status": "not_declared",
                "estimated_peak_bytes": None,
                "must_be_resolved_by": "00a_environment_preflight",
            },
            "requires": entry["requires"],
            "upstream_notebooks": sorted(
                {producers[item] for item in entry["requires"] if item in producers}
            ),
            "produces": entry["produces"],
        }
        for index, entry in enumerate(entries, start=1)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_config_sha256(entry: dict[str, Any], profile: str, run_namespace: str) -> str:
    payload = {
        "entry": entry,
        "profile": profile,
        "run_namespace": run_namespace,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _namespace_root(artifact_root: Path, run_namespace: str, profile: str) -> Path:
    return artifact_root / run_namespace / profile


def _artifact_receipt_path(
    artifact_root: Path, run_namespace: str, profile: str, artifact_id: str
) -> Path:
    return _namespace_root(artifact_root, run_namespace, profile) / f"{artifact_id}.receipt.json"


def _validate_payload(payload_path: Path, artifact_id: str) -> tuple[list[str], int | None]:
    if not payload_path.is_file() or payload_path.stat().st_size == 0:
        return [f"{artifact_id}: payload is missing or empty: {payload_path}"], 0
    suffix = payload_path.suffix.lower()
    try:
        if suffix == ".json":
            value = json.loads(payload_path.read_text(encoding="utf-8"))
            if value in (None, {}, []):
                return [f"{artifact_id}: JSON payload has no records"], 0
            if isinstance(value, list):
                return [], len(value)
            for key in ("records", "rows", "items", "predictions", "metrics", "results"):
                if isinstance(value, dict) and isinstance(value.get(key), list):
                    return [], len(value[key])
            return [], 1
        elif suffix in {".csv", ".tsv"}:
            rows = [line for line in payload_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(rows) < 2:
                return [f"{artifact_id}: tabular payload requires a header and data row"], 0
            return [], len(rows) - 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"{artifact_id}: payload is not parseable: {exc}"], 0
    return [], None


def _validate_artifacts(
    artifact_ids: Sequence[str],
    schema_version: int,
    artifact_root: Path,
    profile: str,
    run_namespace: str,
    expected_provenance: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    root = artifact_root.resolve()
    for artifact_id in artifact_ids:
        receipt_path = _artifact_receipt_path(root, run_namespace, profile, artifact_id)
        if not receipt_path.is_file():
            errors.append(f"{artifact_id}: missing receipt {receipt_path}")
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{artifact_id}: unreadable receipt: {exc}")
            continue
        expected: dict[str, Any] = {
            "artifact_id": artifact_id,
            "schema_version": schema_version,
            "profile": profile,
            "run_namespace": run_namespace,
        }
        status = receipt.get("status")
        if run_namespace == "official" and status != ELIGIBLE_ARTIFACT_STATUS:
            errors.append(f"{artifact_id}: official artifact status must be real, got {status!r}")
            continue
        if run_namespace == "diagnostic" and status not in USABLE_DIAGNOSTIC_STATUSES:
            errors.append(f"{artifact_id}: unusable diagnostic artifact status {status!r}")
            continue
        if expected_provenance:
            expected.update(expected_provenance)
        mismatches = [
            f"{key}={receipt.get(key)!r} (expected {wanted!r})"
            for key, wanted in expected.items()
            if receipt.get(key) != wanted
        ]
        if mismatches:
            errors.append(f"{artifact_id}: receipt mismatch: {', '.join(mismatches)}")
            continue
        payload_value = receipt.get("payload_path")
        expected_hash = receipt.get("sha256")
        if not isinstance(payload_value, str) or not isinstance(expected_hash, str):
            errors.append(f"{artifact_id}: payload_path and sha256 are required")
            continue
        payload_path = (root / payload_value).resolve()
        try:
            payload_path.relative_to(root)
        except ValueError:
            errors.append(f"{artifact_id}: payload_path escapes artifact root")
            continue
        payload_errors, observed_records = _validate_payload(payload_path, artifact_id)
        if payload_errors:
            errors.extend(payload_errors)
        elif _sha256(payload_path) != expected_hash:
            errors.append(f"{artifact_id}: payload checksum mismatch")
        record_count = receipt.get("record_count")
        denominator = receipt.get("denominator")
        if not isinstance(record_count, int) or record_count <= 0:
            errors.append(f"{artifact_id}: record_count must be a positive integer")
        elif observed_records is not None and record_count != observed_records:
            errors.append(
                f"{artifact_id}: record_count={record_count} does not match payload rows={observed_records}"
            )
        if not isinstance(denominator, int) or denominator <= 0:
            errors.append(f"{artifact_id}: denominator must be a positive integer")
        elif isinstance(record_count, int) and denominator < record_count:
            errors.append(f"{artifact_id}: denominator cannot be smaller than record_count")
    return errors


def validate_required_artifacts(
    entry: dict[str, Any], artifact_root: Path, profile: str, run_namespace: str,
    expected_producers: dict[str, str] | None = None,
) -> list[str]:
    """Validate every required artifact receipt, payload, schema, status and hash."""
    errors = _validate_artifacts(
        entry["requires"], entry["expected_schema_version"], artifact_root, profile, run_namespace
    )
    for artifact_id in entry["requires"]:
        receipt_path = _artifact_receipt_path(
            artifact_root.resolve(), run_namespace, profile, artifact_id
        )
        if not receipt_path.is_file():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        producer = receipt.get("producer_notebook_id")
        if not isinstance(producer, str) or not producer:
            errors.append(f"{artifact_id}: producer_notebook_id is required")
        elif expected_producers and producer != expected_producers.get(artifact_id):
            errors.append(
                f"{artifact_id}: producer {producer!r} does not match manifest "
                f"{expected_producers.get(artifact_id)!r}"
            )
        run_id = receipt.get("run_id")
        try:
            uuid.UUID(str(run_id))
        except (ValueError, TypeError, AttributeError):
            errors.append(f"{artifact_id}: run_id must be a UUID")
        for field in ("source_sha256", "config_sha256", "dependency_state_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))):
                errors.append(f"{artifact_id}: {field} must be a SHA-256 hex digest")
    return errors


def validate_produced_artifacts(
    entry: dict[str, Any], artifact_root: Path, profile: str, run_namespace: str,
    run_context: dict[str, Any],
) -> list[str]:
    """Reject successful kernels that did not emit complete scientific evidence."""
    errors = _validate_artifacts(
        entry["produces"], entry["expected_schema_version"], artifact_root, profile, run_namespace,
        {
            "producer_notebook_id": entry["id"],
            "run_id": run_context["run_id"],
            "source_sha256": run_context["source_sha256"],
            "config_sha256": run_context["config_sha256"],
            "dependency_state_sha256": run_context["dependency_state_sha256"],
            "status": run_context["status"],
        },
    )
    runner_manifest = (
        _namespace_root(artifact_root.resolve(), run_namespace, profile)
        / "runs"
        / f"{entry['id']}.run.json"
    )
    if not runner_manifest.is_file():
        errors.append(f"run manifest missing: {runner_manifest}")
        return errors
    try:
        payload = json.loads(runner_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"run manifest unreadable: {exc}")
        return errors
    expected = {
        "schema_version": 1,
        "notebook_id": entry["id"],
        "profile": profile,
        "run_namespace": run_namespace,
        "status": run_context["status"],
        "config_sha256": run_context["config_sha256"],
        "run_id": run_context["run_id"],
        "source_sha256": run_context["source_sha256"],
        "dependency_state_sha256": run_context["dependency_state_sha256"],
    }
    mismatches = [key for key, wanted in expected.items() if payload.get(key) != wanted]
    if mismatches:
        errors.append(f"run manifest mismatch: {mismatches}")
    return errors


def dependency_state_sha256(
    entry: dict[str, Any], artifact_root: Path, profile: str, run_namespace: str
) -> str:
    state = []
    for artifact_id in sorted(entry["requires"]):
        receipt_path = _artifact_receipt_path(
            artifact_root.resolve(), run_namespace, profile, artifact_id
        )
        state.append({"artifact_id": artifact_id, "receipt_sha256": _sha256(receipt_path)})
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def produced_state_sha256(
    entry: dict[str, Any], artifact_root: Path, profile: str, run_namespace: str
) -> str:
    paths = [
        _artifact_receipt_path(artifact_root.resolve(), run_namespace, profile, artifact_id)
        for artifact_id in sorted(entry["produces"])
    ]
    state = [{"path": str(path), "sha256": _sha256(path)} for path in paths]
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_receipt_path(destination: Path) -> Path:
    return destination.with_suffix(".run.json")


def can_resume(
    entry: dict[str, Any], source: Path, destination: Path, profile: str,
    run_namespace: str, dependency_sha256: str, evidence_status: str,
    produced_sha256: str,
) -> tuple[bool, str]:
    """Accept resume only for a parseable output backed by matching hashes."""
    receipt_path = _run_receipt_path(destination)
    if not source.is_file():
        return False, "source notebook is missing"
    if not destination.is_file() or not receipt_path.is_file():
        return False, "executed notebook or run receipt is missing"
    try:
        nbformat.read(destination, as_version=4)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, nbformat.reader.NotJSONError) as exc:
        return False, f"invalid executed output or receipt: {exc}"
    expected = {
        "schema_version": 1,
        "notebook_id": entry["id"],
        "profile": profile,
        "run_namespace": run_namespace,
        "status": evidence_status,
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
        "config_sha256": _entry_config_sha256(entry, profile, run_namespace),
        "dependency_state_sha256": dependency_sha256,
        "produced_state_sha256": produced_sha256,
    }
    mismatches = [key for key, wanted in expected.items() if receipt.get(key) != wanted]
    return (not mismatches, "matching receipt" if not mismatches else f"receipt mismatch: {mismatches}")


def _write_run_receipt(
    entry: dict[str, Any], source: Path, destination: Path, profile: str,
    run_namespace: str, run_context: dict[str, Any], produced_sha256: str,
) -> None:
    receipt = {
        "schema_version": 1,
        "notebook_id": entry["id"],
        "profile": profile,
        "run_namespace": run_namespace,
        "status": run_context["status"],
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(destination),
        "config_sha256": _entry_config_sha256(entry, profile, run_namespace),
        "dependency_state_sha256": run_context["dependency_state_sha256"],
        "produced_state_sha256": produced_sha256,
        "run_id": run_context["run_id"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _run_receipt_path(destination).write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def _write_start_record(
    entry: dict[str, Any], artifact_root: Path, profile: str,
    run_namespace: str, run_context: dict[str, Any],
) -> None:
    path = (
        _namespace_root(artifact_root.resolve(), run_namespace, profile)
        / "runs"
        / f"{entry['id']}.start.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run_context, indent=2) + "\n", encoding="utf-8")


def execute_notebook(
    entry: dict[str, Any],
    source: Path,
    destination: Path,
    timeout: int,
    profile: str,
    run_namespace: str,
    artifact_root: Path,
    run_context: dict[str, Any],
) -> tuple[bool, float, str | None]:
    """Execute in a clean kernel and preserve the source notebook."""
    started = time.perf_counter()
    try:
        notebook = nbformat.read(source, as_version=4)
        NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
        ).execute(
            env={
                **os.environ,
                "MLS_BENCHMARK_PROFILE": profile,
                "MLS_RUN_NAMESPACE": run_namespace,
                "MLS_NOTEBOOK_ID": entry["id"],
                "MLS_CONFIG_SHA256": run_context["config_sha256"],
                "MLS_ARTIFACT_ROOT": str(artifact_root.resolve()),
                "MLS_RUN_ID": run_context["run_id"],
                "MLS_SOURCE_SHA256": run_context["source_sha256"],
                "MLS_DEPENDENCY_STATE_SHA256": run_context["dependency_state_sha256"],
                "MLS_EVIDENCE_STATUS": run_context["status"],
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, destination)
        return True, time.perf_counter() - started, None
    except Exception as exc:  # nbclient exposes several backend-specific errors
        return False, time.perf_counter() - started, str(exc)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_NOTEBOOKS)
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument("--notebook", help="Notebook ID, path, or filename")
    selectors.add_argument("--group", help="Two-digit notebook group")
    selectors.add_argument("--task", help="Exact task_id")
    parser.add_argument("--profile", choices=("smoke", "full"))
    parser.add_argument("--list", action="store_true", help="List selected notebooks")
    parser.add_argument("--dry-plan", action="store_true", help="Print the plan without executing")
    parser.add_argument("--resume", action="store_true", help="Skip an existing executed output")
    parser.add_argument("--run-namespace", choices=("official", "diagnostic"), default="official")
    parser.add_argument("--evidence-status", choices=sorted(EVIDENCE_STATUSES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        profile = args.profile or manifest["official_default_profile"]
        evidence_status = args.evidence_status or (
            ELIGIBLE_ARTIFACT_STATUS if args.run_namespace == "official" else "dry"
        )
        if args.run_namespace == "official" and evidence_status != ELIGIBLE_ARTIFACT_STATUS:
            raise ManifestError("official namespace requires evidence-status=real")
        if args.run_namespace == "diagnostic" and evidence_status not in USABLE_DIAGNOSTIC_STATUSES:
            raise ManifestError(
                f"diagnostic execution requires one of {sorted(USABLE_DIAGNOSTIC_STATUSES)}"
            )
        entries = select_entries(
            manifest,
            notebook=args.notebook,
            group=args.group,
            task=args.task,
            profile=profile,
        )
    except (FileNotFoundError, ValueError, ManifestError) as exc:
        print(f"[FAIL] {exc}")
        return 2

    plan = build_plan(entries, profile, manifest["notebooks"])
    if args.list:
        for item in plan:
            print(f"{item['id']}\t{item['task_id']}\t{item['path']}")
        return 0
    if args.dry_plan:
        print(json.dumps({"profile": profile, "notebooks": plan}, indent=2))
        return 0

    failed = False
    outcomes: dict[str, str] = {}
    selected_ids = {entry["id"] for entry in entries}
    producers = _producer_map(manifest["notebooks"])
    for entry in entries:
        source = PROJECT_ROOT / entry["path"]
        destination = args.output_root / args.run_namespace / profile / f"{entry['id']}.ipynb"
        upstream = {
            producers[artifact]
            for artifact in entry["requires"]
            if artifact in producers and producers[artifact] in selected_ids
        }
        blocked_by = sorted(
            notebook_id for notebook_id in upstream if outcomes.get(notebook_id) in {"failed", "blocked"}
        )
        if blocked_by:
            print(f"[BLOCKED] {entry['id']}: failed upstream notebooks: {blocked_by}")
            outcomes[entry["id"]] = "blocked"
            failed = True
            continue
        if not source.is_file():
            print(f"[FAIL] {entry['id']}: source notebook not found: {source}")
            outcomes[entry["id"]] = "failed"
            failed = True
            continue
        artifact_errors = validate_required_artifacts(
            entry, args.artifact_root, profile, args.run_namespace, producers
        )
        if artifact_errors:
            print(f"[FAIL] {entry['id']}: required artifact validation failed")
            for error in artifact_errors:
                print(f"  - {error}")
            outcomes[entry["id"]] = "failed"
            failed = True
            continue
        dependency_sha256 = dependency_state_sha256(
            entry, args.artifact_root, profile, args.run_namespace
        )
        if args.resume:
            try:
                prior_receipt = json.loads(
                    _run_receipt_path(destination).read_text(encoding="utf-8")
                )
                resume_context = {
                    key: prior_receipt[key]
                    for key in (
                        "run_id", "status", "source_sha256", "config_sha256",
                        "dependency_state_sha256",
                    )
                }
                produced_errors = validate_produced_artifacts(
                    entry, args.artifact_root, profile, args.run_namespace, resume_context
                )
                produced_sha256 = produced_state_sha256(
                    entry, args.artifact_root, profile, args.run_namespace
                ) if not produced_errors else ""
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError):
                produced_errors = ["missing or invalid prior run receipt"]
                produced_sha256 = ""
            resumable, reason = can_resume(
                entry, source, destination, profile, args.run_namespace,
                dependency_sha256, evidence_status, produced_sha256,
            )
            if resumable and not produced_errors:
                print(f"[SKIP] {entry['id']}: {reason}")
                outcomes[entry["id"]] = evidence_status
                continue
            detail = reason if not resumable else f"invalid produced evidence: {produced_errors}"
            print(f"[RERUN] {entry['id']}: {detail}")
        run_context = {
            "schema_version": 1,
            "notebook_id": entry["id"],
            "profile": profile,
            "run_namespace": args.run_namespace,
            "status": evidence_status,
            "run_id": str(uuid.uuid4()),
            "source_sha256": _sha256(source),
            "config_sha256": _entry_config_sha256(entry, profile, args.run_namespace),
            "dependency_state_sha256": dependency_sha256,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_start_record(
            entry, args.artifact_root, profile, args.run_namespace, run_context
        )
        ok, elapsed, error = execute_notebook(
            entry,
            source,
            destination,
            entry["timeout_seconds"],
            profile,
            args.run_namespace,
            args.artifact_root,
            run_context,
        )
        if ok:
            produced_errors = validate_produced_artifacts(
                entry, args.artifact_root, profile, args.run_namespace, run_context
            )
            if produced_errors:
                print(f"[FAIL] {entry['id']}: produced evidence validation failed")
                for produced_error in produced_errors:
                    print(f"  - {produced_error}")
                outcomes[entry["id"]] = "failed"
                failed = True
                continue
            produced_sha256 = produced_state_sha256(
                entry, args.artifact_root, profile, args.run_namespace
            )
            _write_run_receipt(
                entry, source, destination, profile, args.run_namespace,
                run_context, produced_sha256,
            )
            print(f"[OK] {entry['id']} ({elapsed:.1f}s) -> {destination}")
            outcomes[entry["id"]] = evidence_status
        else:
            print(f"[FAIL] {entry['id']} ({elapsed:.1f}s): {error}")
            outcomes[entry["id"]] = "failed"
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

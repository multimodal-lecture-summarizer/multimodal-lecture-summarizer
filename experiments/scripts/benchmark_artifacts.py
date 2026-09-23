"""Shared fail-closed artifact writer for manifest-driven benchmark runners."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class RunnerContext:
    profile: str
    run_namespace: str
    status: str
    artifact_root: Path
    notebook_id: str
    run_id: str
    source_sha256: str
    config_sha256: str
    dependency_state_sha256: str

    @property
    def namespace_root(self) -> Path:
        return self.artifact_root.resolve() / self.run_namespace / self.profile


def runner_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--profile", required=True, choices=("smoke", "full"))
    parser.add_argument("--run-namespace", required=True, choices=("official", "diagnostic"))
    parser.add_argument("--evidence-status", required=True)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--dependency-state-sha256", required=True)
    return parser


def parse_context(
    description: str,
    expected_notebook_id: str,
    argv: Sequence[str] | None = None,
) -> RunnerContext:
    args = runner_parser(description).parse_args(argv)
    if args.notebook_id != expected_notebook_id:
        raise ValueError(
            f"runner {expected_notebook_id} cannot serve notebook {args.notebook_id}"
        )
    if args.run_namespace == "official" and args.evidence_status != "real":
        raise ValueError("official namespace requires evidence-status=real")
    if args.evidence_status not in {"real", "dry", "proxy", "partial"}:
        raise ValueError(f"runner cannot emit status {args.evidence_status!r}")
    uuid.UUID(args.run_id)
    for field in ("source_sha256", "config_sha256", "dependency_state_sha256"):
        if not SHA256_PATTERN.fullmatch(getattr(args, field)):
            raise ValueError(f"{field} must be a SHA-256 hex digest")
    return RunnerContext(
        profile=args.profile,
        run_namespace=args.run_namespace,
        status=args.evidence_status,
        artifact_root=args.artifact_root,
        notebook_id=args.notebook_id,
        run_id=args.run_id,
        source_sha256=args.source_sha256,
        config_sha256=args.config_sha256,
        dependency_state_sha256=args.dependency_state_sha256,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_payload(
    context: RunnerContext,
    artifact_id: str,
    content: str,
    suffix: str,
    record_count: int,
    denominator: int | None,
) -> Path:
    if record_count <= 0:
        raise ValueError(f"{artifact_id}: record_count must be positive")
    denominator = denominator if denominator is not None else record_count
    if denominator < record_count:
        raise ValueError(f"{artifact_id}: denominator cannot be smaller than record_count")
    payload_dir = context.namespace_root / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"{artifact_id}{suffix}"
    temporary_payload = payload_path.with_suffix(payload_path.suffix + f".{context.run_id}.tmp")
    temporary_payload.write_text(content, encoding="utf-8")
    temporary_payload.replace(payload_path)
    relative_payload = payload_path.resolve().relative_to(context.artifact_root.resolve())
    receipt = {
        "artifact_id": artifact_id,
        "schema_version": 1,
        "profile": context.profile,
        "run_namespace": context.run_namespace,
        "status": context.status,
        "producer_notebook_id": context.notebook_id,
        "run_id": context.run_id,
        "source_sha256": context.source_sha256,
        "config_sha256": context.config_sha256,
        "dependency_state_sha256": context.dependency_state_sha256,
        "payload_path": relative_payload.as_posix(),
        "sha256": sha256_file(payload_path),
        "record_count": record_count,
        "denominator": denominator,
    }
    receipt_path = context.namespace_root / f"{artifact_id}.receipt.json"
    temporary_receipt = receipt_path.with_suffix(receipt_path.suffix + f".{context.run_id}.tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary_receipt.replace(receipt_path)
    return payload_path


def write_json_artifact(
    context: RunnerContext,
    artifact_id: str,
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
    denominator: int | None = None,
) -> Path:
    if not records:
        raise ValueError(f"{artifact_id}: records cannot be empty")
    payload = {"schema_version": 1, "records": records}
    if metadata:
        payload["metadata"] = metadata
    return _write_payload(
        context,
        artifact_id,
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        ".json",
        len(records),
        denominator,
    )


def write_text_artifact(
    context: RunnerContext,
    artifact_id: str,
    text: str,
    *,
    record_count: int = 1,
    denominator: int | None = None,
) -> Path:
    if not text.strip():
        raise ValueError(f"{artifact_id}: text cannot be empty")
    return _write_payload(
        context, artifact_id, text.rstrip() + "\n", ".md", record_count, denominator
    )


def read_json_artifact(context: RunnerContext, artifact_id: str) -> dict[str, Any]:
    receipt_path = context.namespace_root / f"{artifact_id}.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("artifact_id") != artifact_id:
        raise ValueError(f"{artifact_id}: receipt identity mismatch")
    producer = receipt.get("producer_notebook_id")
    run_manifest_path = context.namespace_root / "runs" / f"{producer}.run.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "schema_version": 1,
        "notebook_id": producer,
        "profile": context.profile,
        "run_namespace": context.run_namespace,
    }
    for key, wanted in expected_manifest.items():
        if run_manifest.get(key) != wanted:
            raise ValueError(f"{artifact_id}: upstream run manifest {key} mismatch")
    for key in (
        "run_id", "status", "source_sha256", "config_sha256", "dependency_state_sha256"
    ):
        if receipt.get(key) != run_manifest.get(key):
            raise ValueError(f"{artifact_id}: {key} does not match upstream run manifest")
    payload_path = (context.artifact_root.resolve() / receipt["payload_path"]).resolve()
    payload_path.relative_to(context.artifact_root.resolve())
    if sha256_file(payload_path) != receipt.get("sha256"):
        raise ValueError(f"{artifact_id}: payload checksum mismatch")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError(f"{artifact_id}: payload has no records")
    return payload


def write_run_manifest(context: RunnerContext) -> Path:
    runs_dir = context.namespace_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{context.notebook_id}.run.json"
    payload = {
        "schema_version": 1,
        "notebook_id": context.notebook_id,
        "profile": context.profile,
        "run_namespace": context.run_namespace,
        "status": context.status,
        "config_sha256": context.config_sha256,
        "run_id": context.run_id,
        "source_sha256": context.source_sha256,
        "dependency_state_sha256": context.dependency_state_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(path.suffix + f".{context.run_id}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path

"""Offline TED-LIUM Release 3 D1 qualification over validation/test splits.

This adapter does not acquire data, inspect train, or claim D2 eligibility.
Its unit outputs, event log, and checkpoint are operational resume state.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from experiments.scripts.dataset_qualification_runtime import (
    CheckpointFingerprint,
    CheckpointStore,
    CheckpointTrigger,
    ExclusiveOperationLock,
    JsonlEventLogger,
    RunBudget,
    run_interruptible,
)


PILOT_SIZE = 20
HASH_CHUNK_BYTES = 16 * 1024 * 1024
IGNORE_MARKER = "ignore_time_segment_in_scoring"
SPLIT_MAP = {"validation": "validation", "test": "final_test"}

SplitLoader = Callable[[Path, str], Sequence[Mapping[str, Any]]]
AudioDecoder = Callable[[bytes], Mapping[str, Any]]


class SetupSoftStop(RuntimeError):
    """Internal control flow for a budget stop before a checkpoint can exist."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_setup_budget(
    budget: Any,
    logger: JsonlEventLogger,
    started: float,
    *,
    stage: str,
    estimated_seconds: float = 0.1,
    context: Mapping[str, Any] | None = None,
) -> None:
    if budget.should_start_next(estimated_seconds):
        return
    details = {"stage": stage, "reason": getattr(budget, "stop_reason", None)}
    details.update(context or {})
    logger.emit(
        event="setup_soft_stop",
        status="partial_resumable",
        elapsed_sec=time.monotonic() - started,
        context=details,
    )
    raise SetupSoftStop(stage)


def _sha256_arrow_with_budget(
    path: Path, budget: Any, logger: JsonlEventLogger, started: float, *, original_split: str
) -> str:
    size = path.stat().st_size
    logger.emit(
        event="setup_progress",
        status="running",
        elapsed_sec=time.monotonic() - started,
        context={
            "stage": "arrow_hash_started",
            "original_split": original_split,
            "file": path.name,
            "bytes_total": size,
        },
    )
    digest = hashlib.sha256()
    hashed = 0
    with path.open("rb") as handle:
        while True:
            _require_setup_budget(
                budget,
                logger,
                started,
                stage="arrow_hash",
                estimated_seconds=0.25,
                context={
                    "original_split": original_split,
                    "file": path.name,
                    "bytes_hashed": hashed,
                    "bytes_total": size,
                },
            )
            block = handle.read(HASH_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
            hashed += len(block)
            logger.emit(
                event="setup_progress",
                status="running",
                elapsed_sec=time.monotonic() - started,
                context={
                    "stage": "arrow_hash_progress",
                    "original_split": original_split,
                    "file": path.name,
                    "bytes_hashed": hashed,
                    "bytes_total": size,
                },
            )
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _default_split_loader(dataset_root: Path, split: str) -> Sequence[Mapping[str, Any]]:
    from datasets import Audio, load_from_disk

    dataset = load_from_disk(str(dataset_root / split))
    return dataset.cast_column("audio", Audio(decode=False))


def _default_audio_decoder(audio_bytes: bytes) -> Mapping[str, Any]:
    import soundfile as sf

    with sf.SoundFile(io.BytesIO(audio_bytes)) as audio:
        sample_rate = int(audio.samplerate)
        frames = int(audio.frames)
        channels = int(audio.channels)
        while audio.read(frames=65_536, dtype="float32", always_2d=True).size:
            pass
    return {
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "channels": channels,
        "duration_sec": frames / sample_rate if sample_rate else 0.0,
    }


def _disk_used_bytes(path: Path) -> int:
    import psutil

    return int(psutil.disk_usage(str(path.resolve().anchor)).used)


def _snapshot_manifest(
    dataset_root: Path, budget: Any, logger: JsonlEventLogger, started: float
) -> dict[str, Any]:
    _require_setup_budget(budget, logger, started, stage="provenance_read")
    provenance_path = dataset_root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required_provenance = ("dataset_id", "hf_dataset", "hf_splits", "license")
    if any(not provenance.get(field) for field in required_provenance):
        raise ValueError("TED-LIUM provenance is missing required identity fields")
    if provenance["dataset_id"] != "tedlium3" or provenance["hf_dataset"] != "distil-whisper/tedlium":
        raise ValueError("TED-LIUM provenance identity mismatch")
    if not all(split in provenance["hf_splits"] for split in SPLIT_MAP):
        raise ValueError("TED-LIUM provenance does not declare validation/test")
    splits: dict[str, Any] = {}
    for original_split in SPLIT_MAP:
        _require_setup_budget(
            budget,
            logger,
            started,
            stage="split_metadata_read",
            context={"original_split": original_split},
        )
        split_root = dataset_root / original_split
        state_path = split_root / "state.json"
        info_path = split_root / "dataset_info.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info.get("config_name") != "release3":
            raise ValueError(f"TED-LIUM {original_split} is not release3")
        if state.get("_split") != original_split:
            raise ValueError(f"TED-LIUM split identity mismatch: {original_split}")
        arrow_files = sorted(split_root.glob("*.arrow"))
        if not state.get("_fingerprint") or not arrow_files:
            raise ValueError(f"TED-LIUM {original_split} has no trustworthy Arrow snapshot")
        splits[original_split] = {
            "hf_fingerprint": state.get("_fingerprint"),
            "state_sha256": _sha256_file(state_path),
            "dataset_info_sha256": _sha256_file(info_path),
            "arrow_files": [
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_arrow_with_budget(
                        path, budget, logger, started, original_split=original_split
                    ),
                }
                for path in arrow_files
            ],
        }
    manifest = {
        "dataset_id": "tedlium3",
        "release": "release3",
        "source_dataset": provenance.get("hf_dataset"),
        "license": provenance.get("license"),
        "provenance_sha256": _sha256_file(provenance_path),
        "splits": splits,
    }
    manifest["local_snapshot_fingerprint"] = _canonical_sha256(manifest)
    return manifest


def _setup_partial_result(budget: Any, stage: str, elapsed_sec: float) -> dict[str, Any]:
    """Return an explicit non-evidence result when setup cannot safely finish."""
    return {
        "dataset_id": "tedlium3",
        "source": "local_huggingface_saved_arrow",
        "acquisition_level": "D1_local_pilot",
        "profile": "full",
        "root": "experiments/datasets/tedlium3",
        "release": "release3",
        "files": [],
        "selected_files": 0,
        "readable_files": 0,
        "status": "partial_resumable",
        "stop_reason": getattr(budget, "stop_reason", None) or "setup_budget_exhausted",
        "stopped_during_setup": True,
        "setup_stage": stage,
        "elapsed_sec": max(0.0, elapsed_sec),
        "scientific_selection_eligible": False,
        "eligibility_reason": "setup did not complete; no D1 qualification evidence emitted",
        "network_used": False,
    }


def _candidate_metadata(
    rows: Sequence[Mapping[str, Any]], original_split: str
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    ignored = 0
    for index in range(len(rows)):
        row = rows[index]
        text = str(row.get("text") or "").strip()
        if text == IGNORE_MARKER:
            ignored += 1
            continue
        upstream_id = str(row.get("id") or "").strip()
        if not upstream_id:
            raise ValueError(f"TED-LIUM {original_split} row {index} has no id")
        unit_id = f"{SPLIT_MAP[original_split]}:{upstream_id}"
        candidates.append(
            {"index": index, "unit_id": unit_id, "rank": _sha256_bytes(unit_id.encode())}
        )
    candidates.sort(key=lambda item: (item["rank"], item["unit_id"]))
    return candidates, ignored


def _pilot_order(candidates: Mapping[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    per_split = PILOT_SIZE // len(SPLIT_MAP)
    if any(len(candidates[split]) < per_split for split in SPLIT_MAP):
        raise ValueError(f"TED-LIUM D1 requires {per_split} eligible rows per split")
    return [
        (split, candidates[split][index])
        for index in range(per_split)
        for split in SPLIT_MAP
    ]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = serialized.encode("utf-8")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return _sha256_bytes(encoded)


def _unit_output(
    row: Mapping[str, Any], *, original_split: str, unit_id: str, decoder: AudioDecoder
) -> dict[str, Any]:
    audio = row.get("audio")
    if not isinstance(audio, Mapping) or not isinstance(audio.get("bytes"), bytes):
        raise ValueError("TED-LIUM row has no embedded audio bytes")
    audio_bytes = audio["bytes"]
    decoded = dict(decoder(audio_bytes))
    upstream_source = Path(str(row.get("file") or "unknown.sph")).name
    talk_id = Path(upstream_source).stem
    upstream_id = str(row["id"])
    return {
        "unit_id": unit_id,
        "relative_path": f"tedlium3/{original_split}/{upstream_id}.wav",
        "bytes": len(audio_bytes),
        "extension": ".wav",
        "sha256": _sha256_bytes(audio_bytes),
        "audio_sha256": _sha256_bytes(audio_bytes),
        "readable": True,
        "decoded": True,
        "probe_status": "soundfile_decoded",
        "split": SPLIT_MAP[original_split],
        "original_split": original_split,
        "text": str(row.get("text") or ""),
        "speaker_id": str(row.get("speaker_id") or "unknown"),
        "talk_id": talk_id,
        "source_id": upstream_source,
        "sample_rate_hz": int(decoded["sample_rate_hz"]),
        "frames": int(decoded["frames"]),
        "channels": int(decoded["channels"]),
        "duration_sec": float(decoded["duration_sec"]),
    }


def qualify_tedlium(
    dataset_root: Path,
    *,
    run_root: Path,
    run_id: str,
    config_sha256: str,
    code_sha256: str,
    split_loader: SplitLoader = _default_split_loader,
    audio_decoder: AudioDecoder = _default_audio_decoder,
    budget: Any | None = None,
    checkpoint_trigger: CheckpointTrigger | None = None,
) -> dict[str, Any]:
    """Run or resume a deterministic, offline 20-utterance D1 pilot."""
    dataset_root = Path(dataset_root)
    run_root = Path(run_root)
    operation_root = run_root / "operations" / "00b_dataset_feasibility"
    operation_root.mkdir(parents=True, exist_ok=True)
    budget = budget or RunBudget(target_seconds=3600, shutdown_reserve_seconds=300)
    with ExclusiveOperationLock(operation_root / ".qualification.lock"):
        return _qualify_tedlium_locked(
            dataset_root,
            run_root=run_root,
            run_id=run_id,
            config_sha256=config_sha256,
            code_sha256=code_sha256,
            split_loader=split_loader,
            audio_decoder=audio_decoder,
            budget=budget,
            checkpoint_trigger=checkpoint_trigger,
        )


def _qualify_tedlium_locked(
    dataset_root: Path,
    *,
    run_root: Path,
    run_id: str,
    config_sha256: str,
    code_sha256: str,
    split_loader: SplitLoader,
    audio_decoder: AudioDecoder,
    budget: Any,
    checkpoint_trigger: CheckpointTrigger | None,
) -> dict[str, Any]:
    operation_root = run_root / "operations" / "00b_dataset_feasibility"
    started = time.monotonic()
    logger = JsonlEventLogger(
        operation_root / "logs" / f"{run_id}.jsonl",
        run_root=run_root,
        run_id=run_id,
        task_id="tedlium3-d1-pilot",
        shard_id="validation-test",
    )
    logger.emit(event="setup_started", status="running", elapsed_sec=0, context={"network_used": False})
    try:
        snapshot = _snapshot_manifest(dataset_root, budget, logger, started)
    except SetupSoftStop as exc:
        return _setup_partial_result(budget, exc.stage, time.monotonic() - started)
    logger.emit(
        event="setup_progress",
        status="running",
        elapsed_sec=time.monotonic() - started,
        context={"stage": "snapshot_hashed"},
    )
    rows_by_split: dict[str, Sequence[Mapping[str, Any]]] = {}
    for split in SPLIT_MAP:
        try:
            _require_setup_budget(
                budget,
                logger,
                started,
                stage="split_load_start",
                estimated_seconds=1.0,
                context={"original_split": split},
            )
        except SetupSoftStop as exc:
            return _setup_partial_result(budget, exc.stage, time.monotonic() - started)
        logger.emit(
            event="setup_progress",
            status="running",
            elapsed_sec=time.monotonic() - started,
            context={"stage": "split_load_started", "original_split": split},
        )
        rows_by_split[split] = split_loader(dataset_root, split)
        logger.emit(
            event="setup_progress",
            status="running",
            elapsed_sec=time.monotonic() - started,
            context={"stage": "split_loaded", "original_split": split, "rows": len(rows_by_split[split])},
        )
        try:
            _require_setup_budget(
                budget,
                logger,
                started,
                stage="split_load_finished",
                context={"original_split": split, "rows": len(rows_by_split[split])},
            )
        except SetupSoftStop as exc:
            return _setup_partial_result(budget, exc.stage, time.monotonic() - started)
    candidate_sets: dict[str, list[dict[str, Any]]] = {}
    ignored_by_split: dict[str, int] = {}
    for split, rows in rows_by_split.items():
        candidate_sets[split], ignored_by_split[split] = _candidate_metadata(rows, split)
    ordered_candidates = _pilot_order(candidate_sets)
    fingerprint = CheckpointFingerprint(
        dataset_revision="release3",
        config_sha256=config_sha256,
        code_sha256=code_sha256,
        input_manifest_sha256=_canonical_sha256(
            {
                "snapshot": snapshot["local_snapshot_fingerprint"],
                "candidates": [
                    {"original_split": split, "unit_id": item["unit_id"]}
                    for split, item in ordered_candidates
                ],
                "pilot_size": PILOT_SIZE,
                "ignore_marker": IGNORE_MARKER,
            }
        ),
    )
    checkpoint = CheckpointStore(
        run_root, "operations/00b_dataset_feasibility/checkpoint.json", fingerprint
    )
    state = checkpoint.load() or checkpoint.new_state()
    candidate_ids = {item["unit_id"] for _split, item in ordered_candidates}
    attempted_before = set(state["units"]) & candidate_ids
    completed_before = checkpoint.completed_unit_ids(state)
    trigger = checkpoint_trigger or CheckpointTrigger(interval_sec=600, every_completed_units=5)
    peak_disk_used = _disk_used_bytes(run_root)
    initial_disk_used = peak_disk_used
    decode_attempts = 0
    decode_failures = 0
    invocation_bytes = 0
    durations: list[float] = []

    def save_checkpoint(status: str, *, force: bool = False) -> None:
        nonlocal peak_disk_used
        completed = len(set(state["units"]) & candidate_ids)
        now = time.monotonic()
        if not trigger.is_due(now=now, completed_units=completed, force=force):
            return
        state["status"] = status
        checkpoint.save(state)
        trigger.mark_saved(now=now, completed_units=completed)
        peak_disk_used = max(peak_disk_used, _disk_used_bytes(run_root))
        logger.emit(
            event="checkpoint_saved",
            status=status,
            elapsed_sec=now - started,
            context={"completed_units": completed},
        )

    save_checkpoint("running", force=True)
    logger.emit(
        event="task_started",
        status="running",
        elapsed_sec=0,
        context={
            "release": "release3",
            "network_used": False,
            "target_units": PILOT_SIZE,
            "snapshot_fingerprint": snapshot["local_snapshot_fingerprint"],
        },
    )

    def process() -> str:
        nonlocal decode_attempts, decode_failures, invocation_bytes, peak_disk_used
        attempted = set(state["units"]) & candidate_ids
        for original_split, candidate in ordered_candidates:
            unit_id = candidate["unit_id"]
            if unit_id in attempted:
                continue
            estimate = sum(durations[-5:]) / len(durations[-5:]) if durations else 1.0
            if not budget.should_start_next(estimate):
                save_checkpoint("partial_resumable", force=True)
                logger.emit(
                    event="soft_stop",
                    status="partial_resumable",
                    elapsed_sec=time.monotonic() - started,
                    context={"reason": budget.stop_reason, "attempted_units": len(attempted)},
                )
                return "partial_resumable"
            unit_started = time.monotonic()
            logger.emit(
                event="unit_started",
                status="running",
                elapsed_sec=unit_started - started,
                unit_id=unit_id,
                context={"original_split": original_split},
            )
            decode_attempts += 1
            row = rows_by_split[original_split][candidate["index"]]
            audio = row.get("audio")
            audio_bytes = audio.get("bytes") if isinstance(audio, Mapping) else None
            processed_bytes = len(audio_bytes) if isinstance(audio_bytes, bytes) else 0
            invocation_bytes += processed_bytes
            try:
                output = _unit_output(
                    row, original_split=original_split, unit_id=unit_id, decoder=audio_decoder
                )
                duration = time.monotonic() - unit_started
                output["outcome"] = "completed"
                output["processing_elapsed_sec"] = duration
                output["observed_disk_used_bytes"] = _disk_used_bytes(run_root)
                output_name = f"{_sha256_bytes(unit_id.encode())}.json"
                relative_output = Path("operations/00b_dataset_feasibility/units") / output_name
                output_sha = _atomic_json(run_root / relative_output, output)
                checkpoint.record_unit(
                    state,
                    unit_id,
                    status="completed",
                    output_sha256=output_sha,
                    output_path=relative_output,
                )
                attempted.add(unit_id)
                durations.append(duration)
                peak_disk_used = max(peak_disk_used, int(output["observed_disk_used_bytes"]))
                logger.emit(
                    event="unit_completed",
                    status="completed",
                    elapsed_sec=time.monotonic() - started,
                    unit_id=unit_id,
                    context={
                        "duration_sec": duration,
                        "attempted_units": len(attempted),
                        "audio_bytes": output["bytes"],
                    },
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                decode_failures += 1
                duration = time.monotonic() - unit_started
                failure = {
                    "unit_id": unit_id,
                    "outcome": "failed",
                    "error_type": type(exc).__name__,
                    "split": SPLIT_MAP[original_split],
                    "original_split": original_split,
                    "processed_audio_bytes": processed_bytes,
                    "processing_elapsed_sec": duration,
                    "observed_disk_used_bytes": _disk_used_bytes(run_root),
                }
                output_name = f"{_sha256_bytes(unit_id.encode())}.json"
                relative_output = Path("operations/00b_dataset_feasibility/attempts") / output_name
                output_sha = _atomic_json(run_root / relative_output, failure)
                checkpoint.record_unit(
                    state,
                    unit_id,
                    status="failed",
                    output_sha256=output_sha,
                    output_path=relative_output,
                    error=type(exc).__name__,
                )
                attempted.add(unit_id)
                durations.append(duration)
                peak_disk_used = max(peak_disk_used, int(failure["observed_disk_used_bytes"]))
                logger.emit(
                    event="unit_failed",
                    status="failed",
                    elapsed_sec=time.monotonic() - started,
                    unit_id=unit_id,
                    context={"error_type": type(exc).__name__},
                )
            save_checkpoint("running")
        final_status = "completed" if len(attempted) == PILOT_SIZE else "failed"
        save_checkpoint(final_status, force=True)
        logger.emit(
            event="task_finished",
            status=final_status,
            elapsed_sec=time.monotonic() - started,
            context={"attempted_units": len(attempted)},
        )
        return final_status

    def on_interrupt() -> None:
        save_checkpoint("partial_resumable", force=True)
        logger.emit(
            event="interrupted",
            status="partial_resumable",
            elapsed_sec=time.monotonic() - started,
        )

    status = run_interruptible(process, on_interrupt=on_interrupt)
    attempt_records: list[dict[str, Any]] = []
    for unit_id in sorted(set(state["units"]) & candidate_ids):
        record = state["units"][unit_id]
        output_path = run_root / record["output_path"]
        if not output_path.is_file() or _sha256_file(output_path) != record["output_sha256"]:
            raise RuntimeError(f"TED-LIUM attempt artifact failed integrity validation: {unit_id}")
        attempt_records.append(json.loads(output_path.read_text(encoding="utf-8")))
    units = [item for item in attempt_records if item["outcome"] == "completed"]
    units.sort(key=lambda item: (item["split"], item["unit_id"]))
    elapsed = max(0.0, time.monotonic() - started)
    selected_bytes = sum(int(item["bytes"]) for item in units)
    aggregate_attempts = len(attempt_records)
    aggregate_failures = sum(item["outcome"] == "failed" for item in attempt_records)
    aggregate_processing_elapsed = sum(float(item["processing_elapsed_sec"]) for item in attempt_records)
    aggregate_processed_bytes = sum(
        int(item.get("bytes", item.get("processed_audio_bytes", 0))) for item in attempt_records
    )
    aggregate_peak_disk = max(
        [initial_disk_used, *(int(item["observed_disk_used_bytes"]) for item in attempt_records)]
    )
    fixed_success_rate = len(units) / PILOT_SIZE
    return {
        "dataset_id": "tedlium3",
        "source": "local_huggingface_saved_arrow",
        "acquisition_level": "D1_local_pilot",
        "profile": "full",
        "root": "experiments/datasets/tedlium3",
        "release": "release3",
        "local_snapshot_fingerprint": snapshot["local_snapshot_fingerprint"],
        "provenance": snapshot,
        "total_files": sum(len(rows) for rows in rows_by_split.values()),
        "total_bytes": selected_bytes,
        "selected_files": len(units),
        "readable_files": len(units),
        "selection_policy": "deterministic_sha256_rank_balanced_validation_final_test_pilot20",
        "files": units,
        "split_counts": dict(sorted(Counter(item["split"] for item in units).items())),
        "ignore_policy": {
            "marker": IGNORE_MARKER,
            "action": "exclude_before_pilot_selection",
            "excluded_count": sum(ignored_by_split.values()),
            "excluded_by_original_split": ignored_by_split,
        },
        "decode_attempts": aggregate_attempts,
        "decode_attempts_this_run": decode_attempts,
        "decode_failures": aggregate_failures,
        "decode_failures_this_run": decode_failures,
        "decode_success_rate": fixed_success_rate,
        "decode_gate_denominator": PILOT_SIZE,
        "decode_gate_threshold": 0.95,
        "decode_gate_passed": status == "completed" and fixed_success_rate >= 0.95,
        "elapsed_sec": elapsed,
        "elapsed_sec_this_run": elapsed,
        "processing_elapsed_sec": aggregate_processing_elapsed,
        "processing_elapsed_sec_this_run": sum(durations),
        "processed_audio_bytes": aggregate_processed_bytes,
        "processed_audio_bytes_this_run": invocation_bytes,
        "throughput_audio_bytes_per_sec": aggregate_processed_bytes / aggregate_processing_elapsed if aggregate_processing_elapsed else 0.0,
        "peak_disk_used_bytes": aggregate_peak_disk,
        "peak_disk_used_bytes_this_run": peak_disk_used,
        "peak_disk_delta_bytes": max(0, peak_disk_used - initial_disk_used),
        "resumed_units": len(attempted_before),
        "resumed_successful_units": len(completed_before),
        "status": status,
        "stop_reason": getattr(budget, "stop_reason", None),
        "scientific_selection_eligible": False,
        "eligibility_reason": "D1 local decode pilot only; rights and endpoint GT remain unadjudicated",
        "network_used": False,
    }

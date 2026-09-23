"""Tests for the shared, fail-closed dataset qualification runtime."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path

import pytest

from experiments.scripts.dataset_qualification_runtime import (
    CheckpointError,
    CheckpointFingerprint,
    CheckpointStore,
    CheckpointTrigger,
    JsonlEventLogger,
    RunBudget,
    resolve_under_root,
    run_interruptible,
)


def _fingerprint(*, revision: str = "release-3") -> CheckpointFingerprint:
    return CheckpointFingerprint(
        dataset_revision=revision,
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        input_manifest_sha256="c" * 64,
    )


def _write_output(root: Path, relative_path: str, content: bytes = b"result") -> str:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_jsonl_logger_emits_ordered_required_fields_and_redacts(tmp_path: Path) -> None:
    log_path = tmp_path / "run" / "events.jsonl"
    logger = JsonlEventLogger(
        log_path,
        run_root=tmp_path,
        run_id="run-1",
        task_id="integrity-scan",
        shard_id="shard-0",
        wall_clock=lambda: "2026-09-23T07:00:00Z",
    )

    first = logger.emit(
        event="unit_started",
        status="running",
        elapsed_sec=1.25,
        unit_id="talk-1",
        context={
            "path": "dev/talk-1.sph",
            "api_token": "do-not-log",
            "nested": {"password": "secret", "count": 2},
        },
    )
    second = logger.emit(
        event="unit_completed",
        status="completed",
        elapsed_sec=2.5,
        unit_id="talk-1",
    )

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records == [first, second]
    assert [record["sequence"] for record in records] == [1, 2]
    assert set(records[0]) == {
        "schema_version",
        "timestamp",
        "sequence",
        "run_id",
        "task_id",
        "shard_id",
        "unit_id",
        "event",
        "status",
        "elapsed_sec",
        "context",
    }
    assert records[0]["context"] == {
        "path": "dev/talk-1.sph",
        "api_token": "[REDACTED]",
        "nested": {"password": "[REDACTED]", "count": 2},
    }
    assert "do-not-log" not in log_path.read_text(encoding="utf-8")


def test_logger_propagates_write_or_flush_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    logger = JsonlEventLogger(
        tmp_path / "events.jsonl",
        run_root=tmp_path,
        run_id="run-1",
        task_id="scan",
        shard_id="0",
    )

    class BrokenFile:
        def __enter__(self) -> "BrokenFile":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def write(self, _value: str) -> int:
            return 1

        def flush(self) -> None:
            raise OSError("disk full")

        def fileno(self) -> int:
            return 1

    original_open = Path.open

    def broken_append(target: Path, *args: object, **kwargs: object) -> object:
        if args and args[0] == "a":
            return BrokenFile()
        return original_open(target, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken_append)
    with pytest.raises(OSError, match="disk full"):
        logger.emit(event="heartbeat", status="running", elapsed_sec=1.0)


def test_logger_rolls_back_partial_append_to_last_durable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    logger.emit(event="started", status="running", elapsed_sec=0)
    durable_bytes = path.read_bytes()
    original_open = Path.open

    class PartialWriter:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self) -> "PartialWriter":
            self.handle.__enter__()  # type: ignore[attr-defined]
            return self

        def __exit__(self, *args: object) -> object:
            return self.handle.__exit__(*args)  # type: ignore[attr-defined]

        def write(self, value: str) -> int:
            self.handle.write(value[:8])  # type: ignore[attr-defined]
            self.handle.flush()  # type: ignore[attr-defined]
            raise OSError("simulated short write")

        def flush(self) -> None:
            self.handle.flush()  # type: ignore[attr-defined]

        def fileno(self) -> int:
            return self.handle.fileno()  # type: ignore[attr-defined,no-any-return]

    def partial_open(target: Path, *args: object, **kwargs: object) -> object:
        handle = original_open(target, *args, **kwargs)
        if args and args[0] == "a":
            return PartialWriter(handle)
        return handle

    monkeypatch.setattr(Path, "open", partial_open)
    with pytest.raises(OSError, match="short write"):
        logger.emit(event="heartbeat", status="running", elapsed_sec=1)
    assert path.read_bytes() == durable_bytes

    monkeypatch.setattr(Path, "open", original_open)
    resumed = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    resumed.emit(event="resumed", status="running", elapsed_sec=2)
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [1, 2]


def test_logger_resume_continues_sequence_without_overwriting(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    first.emit(event="started", status="running", elapsed_sec=0)
    resumed = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    resumed.emit(event="resumed", status="running", elapsed_sec=10)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["sequence"] for record in records] == [1, 2]


def test_checkpoint_round_trip_resume_and_noncomplete_units(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "checkpoints/state.json", _fingerprint())
    state = store.new_state()
    output_hash = _write_output(tmp_path, "outputs/talk-1.json")
    store.record_unit(
        state,
        "talk-1",
        status="completed",
        output_sha256=output_hash,
        output_path="outputs/talk-1.json",
    )
    store.record_unit(state, "talk-2", status="retryable", error="decode timeout")
    store.record_unit(state, "talk-3", status="failed", error="invalid audio")
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert store.completed_unit_ids(loaded) == {"talk-1"}
    assert loaded["units"]["talk-2"]["status"] == "retryable"
    assert loaded["units"]["talk-3"]["status"] == "failed"
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_checkpoint_corruption_and_fingerprint_mismatch_fail_closed(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "state.json", _fingerprint())
    store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(CheckpointError, match="corrupt"):
        store.load()

    store.path.unlink()
    store.save(store.new_state())
    changed = CheckpointStore(tmp_path, "state.json", _fingerprint(revision="release-3-hotfix"))
    with pytest.raises(CheckpointError, match="fingerprint mismatch"):
        changed.load()


def test_completed_unit_requires_output_hash(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "state.json", _fingerprint())
    state = store.new_state()
    with pytest.raises(CheckpointError, match="output_sha256"):
        store.record_unit(state, "talk-1", status="completed")
    with pytest.raises(CheckpointError, match="status"):
        store.record_unit(state, "talk-1", status="unknown")


def test_completed_unit_resume_rehashes_output_and_rejects_stale_artifact(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "state.json", _fingerprint())
    state = store.new_state()
    output_hash = _write_output(tmp_path, "outputs/talk-1.json")
    store.record_unit(
        state,
        "talk-1",
        status="completed",
        output_sha256=output_hash,
        output_path="outputs/talk-1.json",
    )
    assert store.completed_unit_ids(state) == {"talk-1"}

    (tmp_path / "outputs/talk-1.json").write_bytes(b"stale or tampered")
    assert store.completed_unit_ids(state) == set()
    (tmp_path / "outputs/talk-1.json").unlink()
    assert store.completed_unit_ids(state) == set()


def test_checkpoint_trigger_supports_time_count_and_force() -> None:
    trigger = CheckpointTrigger(interval_sec=300, every_completed_units=20)
    trigger.mark_saved(now=10.0, completed_units=4)
    assert not trigger.is_due(now=200.0, completed_units=10)
    assert trigger.is_due(now=311.0, completed_units=10)
    trigger.mark_saved(now=311.0, completed_units=10)
    assert trigger.is_due(now=312.0, completed_units=30)
    assert trigger.is_due(now=312.0, completed_units=11, force=True)


def test_run_budget_adapts_to_estimate_and_soft_stops() -> None:
    now = [100.0]
    budget = RunBudget(
        target_seconds=3600,
        shutdown_reserve_seconds=300,
        monotonic=lambda: now[0],
    )
    assert budget.should_start_next(estimated_seconds=3000)
    now[0] += 301
    assert not budget.should_start_next(estimated_seconds=3000)
    assert budget.status == "partial_resumable"
    assert budget.stop_reason == "insufficient_time_for_next_task"
    with pytest.raises(ValueError, match="estimated_seconds"):
        budget.should_start_next(estimated_seconds=-1)


def test_keyboard_interrupt_flushes_checkpoint_and_returns_partial() -> None:
    calls: list[str] = []

    def interrupted() -> None:
        raise KeyboardInterrupt

    result = run_interruptible(interrupted, on_interrupt=lambda: calls.append("saved"))
    assert result == "partial_resumable"
    assert calls == ["saved"]


@pytest.mark.parametrize("path", ["../escape.json", "nested/../../escape.json"])
def test_paths_cannot_escape_run_root(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError, match="run root"):
        resolve_under_root(tmp_path, path)


def test_absolute_paths_are_rejected_even_when_inside_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="relative"):
        resolve_under_root(tmp_path, tmp_path / "state.json")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_runtime_values_fail_closed(tmp_path: Path, value: float) -> None:
    logger = JsonlEventLogger(
        tmp_path / "events.jsonl",
        run_root=tmp_path,
        run_id="run-1",
        task_id="scan",
        shard_id="0",
    )
    with pytest.raises(ValueError, match="non-negative number"):
        logger.emit(event="heartbeat", status="running", elapsed_sec=value)
    assert not logger.path.exists()

    with pytest.raises(ValueError, match="finite"):
        RunBudget(target_seconds=value)
    with pytest.raises(ValueError, match="finite"):
        CheckpointTrigger(interval_sec=value)


def test_logger_rejects_sequence_corruption_and_identity_drift(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    logger.emit(event="started", status="running", elapsed_sec=0)
    logger.emit(event="heartbeat", status="running", elapsed_sec=1)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["sequence"] = 2
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        JsonlEventLogger(
            path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
        )

    records[0]["sequence"] = 1
    records[0]["run_id"] = "different-run"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corrupt"):
        JsonlEventLogger(
            path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
        )


def test_logger_recovers_only_non_newline_torn_final_suffix(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    first.emit(event="started", status="running", elapsed_sec=0)
    durable_bytes = path.read_bytes()
    torn_suffix = b'{"sequence":2,"event":"heart'
    with path.open("ab") as handle:
        handle.write(torn_suffix)
        handle.flush()
        os.fsync(handle.fileno())

    recovered = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    assert path.read_bytes() == durable_bytes
    quarantines = list(tmp_path.glob(".events.jsonl.torn.*"))
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == torn_suffix

    recovered.emit(event="resumed", status="running", elapsed_sec=1)
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [1, 2]


def test_logger_never_repairs_corruption_in_a_complete_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(
        path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
    )
    logger.emit(event="started", status="running", elapsed_sec=0)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["sequence"] = 9
    corrupt_complete = json.dumps(records[0]).encode("utf-8") + b"\n"
    original = corrupt_complete + b'{"sequence":2'
    path.write_bytes(original)

    with pytest.raises(RuntimeError, match="corrupt"):
        JsonlEventLogger(
            path, run_root=tmp_path, run_id="run-1", task_id="scan", shard_id="0"
        )
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".events.jsonl.torn.*"))


@pytest.mark.parametrize("owner_state", [True, None])
def test_logger_never_clears_live_or_unknown_owner_lock(
    tmp_path: Path, owner_state: object
) -> None:
    path = tmp_path / "events.jsonl"
    lock = tmp_path / ".events.jsonl.lock"
    owner = {"pid": 4242, "token": "existing-owner"}
    lock.write_text(json.dumps(owner), encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing recovery"):
        JsonlEventLogger(
            path,
            run_root=tmp_path,
            run_id="run-1",
            task_id="scan",
            shard_id="0",
            process_alive=lambda _pid: owner_state,  # type: ignore[return-value]
        )
    assert json.loads(lock.read_text(encoding="utf-8")) == owner


def test_logger_clears_only_proven_dead_owner_lock(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    lock = tmp_path / ".events.jsonl.lock"
    lock.write_text(json.dumps({"pid": 4242, "token": "dead-owner"}), encoding="utf-8")
    observed: list[int] = []

    logger = JsonlEventLogger(
        path,
        run_root=tmp_path,
        run_id="run-1",
        task_id="scan",
        shard_id="0",
        process_alive=lambda pid: observed.append(pid) or False,
    )
    assert observed == [4242]
    assert not lock.exists()
    logger.emit(event="started", status="running", elapsed_sec=0)


def test_logger_redacts_sensitive_values_not_only_sensitive_keys(tmp_path: Path) -> None:
    logger = JsonlEventLogger(
        tmp_path / "events.jsonl",
        run_root=tmp_path,
        run_id="run-1",
        task_id="scan",
        shard_id="0",
    )
    record = logger.emit(
        event="retry",
        status="retryable",
        elapsed_sec=1,
        context={
            "message": "request failed: Bearer abc123 token=xyz",
            "url": "https://example.invalid/upload?api_key=private&part=1",
        },
    )
    serialized = logger.path.read_text(encoding="utf-8")
    assert record["context"]["message"] == (
        "request failed: Bearer [REDACTED] token=[REDACTED]"
    )
    assert "api_key=[REDACTED]" in record["context"]["url"]
    assert all(secret not in serialized for secret in ("abc123", "xyz", "private"))


def test_checkpoint_rejects_extra_state_and_unit_fields(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "state.json", _fingerprint())
    state = store.new_state()
    state["unexpected"] = True
    with pytest.raises(CheckpointError, match="unsupported fields"):
        store.save(state)

    state = store.new_state()
    state["units"]["talk-1"] = {"status": "pending", "unexpected": True}
    with pytest.raises(CheckpointError, match="unsupported fields"):
        store.save(state)


def test_replace_failure_preserves_prior_checkpoint_and_cleans_unique_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CheckpointStore(tmp_path, "checkpoints/state.json", _fingerprint())
    original = store.new_state()
    store.save(original)
    original_bytes = store.path.read_bytes()

    replacement = store.new_state()
    output_hash = _write_output(tmp_path, "outputs/talk-1.json")
    store.record_unit(
        replacement,
        "talk-1",
        status="completed",
        output_sha256=output_hash,
        output_path="outputs/talk-1.json",
    )
    sources: list[Path] = []

    def reject_replace(source: Path, _destination: Path) -> None:
        sources.append(Path(source))
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", reject_replace)
    with pytest.raises(OSError, match="replace failure"):
        store.save(replacement)

    assert store.path.read_bytes() == original_bytes
    assert sources and not sources[0].exists()
    assert not list(store.path.parent.glob(".*.tmp"))


def test_preexisting_legacy_temp_does_not_collide_with_checkpoint_save(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "checkpoints/state.json", _fingerprint())
    legacy_temp = store.path.with_name(f".{store.path.name}.tmp")
    legacy_temp.write_text("unrelated writer", encoding="utf-8")
    store.save(store.new_state())
    assert legacy_temp.read_text(encoding="utf-8") == "unrelated writer"
    assert store.load() == store.new_state()


def test_checkpoint_errors_are_redacted_before_persistence(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, "state.json", _fingerprint())
    state = store.new_state()
    store.record_unit(
        state,
        "talk-1",
        status="retryable",
        error="upload failed with Bearer top-secret and password=hunter2",
    )
    store.save(state)
    serialized = store.path.read_text(encoding="utf-8")
    assert "top-secret" not in serialized
    assert "hunter2" not in serialized
    assert serialized.count("[REDACTED]") == 2


def test_interrupt_hook_failure_propagates_instead_of_claiming_resumable() -> None:
    def interrupted() -> None:
        raise KeyboardInterrupt

    def checkpoint_failed() -> None:
        raise OSError("checkpoint disk full")

    with pytest.raises(OSError, match="checkpoint disk full"):
        run_interruptible(interrupted, on_interrupt=checkpoint_failed)

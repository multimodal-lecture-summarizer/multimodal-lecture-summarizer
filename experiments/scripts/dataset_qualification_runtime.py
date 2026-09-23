"""Shared reliability primitives for long-running dataset qualification tasks.

The module is intentionally stdlib-only.  It does not decide scientific
eligibility; it makes progress and telemetry durable enough for 00b/00c
qualification runners to resume without silently changing their inputs.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, TypeVar, Union


EVENT_SCHEMA_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "1.0"
PARTIAL_RESUMABLE = "partial_resumable"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:authorization|cookie|credential|password|secret|token|api[_-]?key)",
    re.IGNORECASE,
)
_UNIT_STATUSES = {"pending", "running", "completed", "retryable", "failed"}
_CHECKPOINT_STATUSES = {"running", PARTIAL_RESUMABLE, "completed", "failed"}
_EVENT_KEYS = {
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
_SENSITIVE_VALUE_RES = (
    re.compile(r"(?i)\b(bearer\s+)[^\s,;]+"),
    re.compile(
        r"(?i)\b((?:authorization|credential|password|secret|token|api[_-]?key)\s*[=:]\s*)[^\s&,;]+"
    ),
    re.compile(
        r"(?i)([?&](?:authorization|credential|password|secret|token|api[_-]?key)=)[^&#\s]+"
    ),
)


class CheckpointError(RuntimeError):
    """A checkpoint cannot be trusted or does not match the current run."""


def _read_lock_owner(path: Path) -> Optional[dict[str, Union[int, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"pid", "token"}
        or isinstance(payload["pid"], bool)
        or not isinstance(payload["pid"], int)
        or payload["pid"] <= 0
        or not isinstance(payload["token"], str)
        or not payload["token"]
    ):
        return None
    return payload


def _process_is_alive(pid: int) -> Optional[bool]:
    """Return False only when the OS proves that ``pid`` is not alive."""

    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
                process_query_limited_information, False, pid
            )
            if handle:
                try:
                    exit_code = ctypes.c_ulong()
                    if not ctypes.windll.kernel32.GetExitCodeProcess(  # type: ignore[attr-defined]
                        handle, ctypes.byref(exit_code)
                    ):
                        return None
                    return exit_code.value == still_active
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
            # ERROR_INVALID_PARAMETER proves no process has this PID. Access
            # denied and every other result are deliberately treated unknown.
            return False if ctypes.windll.kernel32.GetLastError() == 87 else None  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return None
    return True


class ExclusiveOperationLock:
    """Hold one filesystem-backed lock for an entire qualification operation."""

    def __init__(
        self,
        path: Path,
        *,
        process_alive: Callable[[int], Optional[bool]] = _process_is_alive,
    ) -> None:
        self.path = Path(path)
        self._process_alive = process_alive
        self._owner: dict[str, Union[int, str]] = {
            "pid": os.getpid(),
            "token": uuid.uuid4().hex,
        }

    def __enter__(self) -> "ExclusiveOperationLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(self._owner, sort_keys=True).encode("utf-8")
        for _attempt in range(2):
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(descriptor, encoded)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                return self
            except FileExistsError:
                existing = _read_lock_owner(self.path)
                if existing is None or self._process_alive(int(existing["pid"])) is not False:
                    raise RuntimeError(f"operation is already locked: {self.path}")
                if _read_lock_owner(self.path) != existing:
                    raise RuntimeError(f"operation lock changed during recovery: {self.path}")
                self.path.unlink()
        raise RuntimeError(f"could not acquire operation lock: {self.path}")

    def __exit__(self, *_args: object) -> None:
        if _read_lock_owner(self.path) == self._owner:
            self.path.unlink()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def resolve_under_root(run_root: Path, relative_path: Union[str, Path]) -> Path:
    """Resolve a relative artifact path and reject traversal outside run_root."""

    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("artifact path must be relative to the run root")
    root = Path(run_root).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("artifact path escapes the run root") from exc
    return candidate


def _redact(value: Any, *, key: Optional[str] = None) -> Any:
    if key is not None and _SENSITIVE_KEY_RE.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in _SENSITIVE_VALUE_RES:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        return redacted
    return value


class JsonlEventLogger:
    """Append durable, sequence-numbered structured events to one JSONL file."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        run_root: Path,
        run_id: str,
        task_id: str,
        shard_id: str,
        wall_clock: Callable[[], str] = _utc_now,
        process_alive: Callable[[int], Optional[bool]] = lambda pid: _process_is_alive(pid),
    ) -> None:
        raw_path = Path(path)
        if raw_path.is_absolute():
            root = Path(run_root).resolve()
            try:
                relative_path = raw_path.resolve().relative_to(root)
            except ValueError as exc:
                raise ValueError("event log path escapes the run root") from exc
        else:
            relative_path = raw_path
        self.path = resolve_under_root(run_root, relative_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = _required_text(run_id, "run_id")
        self.task_id = _required_text(task_id, "task_id")
        self.shard_id = _required_text(shard_id, "shard_id")
        self._wall_clock = wall_clock
        self._process_alive = process_alive
        lock_owner = self._acquire_lock()
        try:
            self._sequence = self._last_sequence(recover_torn_suffix=True)
        finally:
            self._release_lock(lock_owner)

    @property
    def _lock_path(self) -> Path:
        return self.path.with_name(f".{self.path.name}.lock")

    def _acquire_lock(self) -> bytes:
        owner = json.dumps(
            {"pid": os.getpid(), "token": uuid.uuid4().hex},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for _attempt in range(2):
            owner_candidate = self._lock_path.with_name(
                f".{self._lock_path.name}.owner.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with owner_candidate.open("xb") as handle:
                    handle.write(owner)
                    handle.flush()
                    os.fsync(handle.fileno())
                # Hard-linking a fully fsynced owner record makes lock
                # publication atomic: a hard kill can leave no lock or a lock
                # with a complete PID/token, never a half-written owner.
                os.link(owner_candidate, self._lock_path)
            except FileExistsError as exc:
                existing_owner = _read_lock_owner(self._lock_path)
                if existing_owner is None:
                    raise RuntimeError(
                        f"event log lock owner is unknown; refusing recovery: {self.path}"
                    ) from exc
                alive = self._process_alive(existing_owner["pid"])
                if alive is not False:
                    state = "active" if alive else "unknown"
                    raise RuntimeError(
                        f"event log has an {state} writer; refusing recovery: {self.path}"
                    ) from exc
                # Re-read before unlinking so a replaced/live lock is never
                # removed based on a stale owner observation.
                if _read_lock_owner(self._lock_path) != existing_owner:
                    raise RuntimeError(
                        f"event log lock changed during recovery: {self.path}"
                    ) from exc
                try:
                    self._lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            finally:
                try:
                    owner_candidate.unlink()
                except FileNotFoundError:
                    pass
            return owner
        raise RuntimeError(f"could not acquire event log writer lock: {self.path}")

    def _release_lock(self, owner: bytes) -> None:
        try:
            if self._lock_path.read_bytes() == owner:
                self._lock_path.unlink()
        except FileNotFoundError:
            pass

    def _last_sequence(self, *, recover_torn_suffix: bool = False) -> int:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        try:
            payload = self.path.read_bytes()
            torn_suffix = b""
            complete_payload = payload
            if not payload.endswith(b"\n"):
                boundary = payload.rfind(b"\n")
                complete_payload = payload[: boundary + 1] if boundary >= 0 else b""
                torn_suffix = payload[boundary + 1 :]
            lines = complete_payload.splitlines()
            for expected_sequence, raw_line in enumerate(lines, start=1):
                record = json.loads(raw_line.decode("utf-8"))
                if not isinstance(record, Mapping) or set(record) != _EVENT_KEYS:
                    raise ValueError("event record fields do not match the schema")
                if record["sequence"] != expected_sequence:
                    raise ValueError("event sequence is not contiguous")
                if record["schema_version"] != EVENT_SCHEMA_VERSION:
                    raise ValueError("event schema_version is unsupported")
                for field, expected in (
                    ("run_id", self.run_id),
                    ("task_id", self.task_id),
                    ("shard_id", self.shard_id),
                ):
                    if record[field] != expected:
                        raise ValueError(f"event {field} does not match the logger")
                _required_text(record["timestamp"], "timestamp")
                _required_text(record["event"], "event")
                _required_text(record["status"], "status")
                _nonnegative_number(record["elapsed_sec"], "elapsed_sec")
                if record["unit_id"] is not None:
                    _required_text(record["unit_id"], "unit_id")
                if not isinstance(record["context"], Mapping):
                    raise ValueError("event context must be an object")
            if torn_suffix:
                if not recover_torn_suffix:
                    raise ValueError("event log has an incomplete final suffix")
                quarantine = self.path.with_name(
                    f".{self.path.name}.torn.{uuid.uuid4().hex}"
                )
                with quarantine.open("xb") as handle:
                    handle.write(torn_suffix)
                    handle.flush()
                    os.fsync(handle.fileno())
                truncate_fd = os.open(self.path, os.O_WRONLY)
                try:
                    os.ftruncate(truncate_fd, len(complete_payload))
                    os.fsync(truncate_fd)
                finally:
                    os.close(truncate_fd)
            sequence = len(lines)
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"existing event log is corrupt: {self.path}") from exc
        return sequence

    def emit(
        self,
        *,
        event: str,
        status: str,
        elapsed_sec: float,
        unit_id: Optional[str] = None,
        context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        lock_owner = self._acquire_lock()
        try:
            # Re-read while holding the lock so independently constructed
            # logger instances cannot emit duplicate sequence numbers.
            sequence = self._last_sequence(recover_torn_suffix=True) + 1
            record = {
                "schema_version": EVENT_SCHEMA_VERSION,
                "timestamp": self._wall_clock(),
                "sequence": sequence,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "shard_id": self.shard_id,
                "unit_id": unit_id,
                "event": _required_text(event, "event"),
                "status": _required_text(status, "status"),
                "elapsed_sec": _nonnegative_number(elapsed_sec, "elapsed_sec"),
                "context": _redact(dict(context or {})),
            }
            line = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            # Opening per event avoids buffering telemetry in process memory.
            previous_size = self.path.stat().st_size if self.path.exists() else 0
            try:
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                # A short write must not poison all future resumes. Best-effort
                # rollback preserves the last fully fsynced JSONL boundary;
                # the original I/O exception remains the caller-visible cause.
                try:
                    rollback_fd = os.open(self.path, os.O_WRONLY)
                    try:
                        os.ftruncate(rollback_fd, previous_size)
                        os.fsync(rollback_fd)
                    finally:
                        os.close(rollback_fd)
                except OSError:
                    pass
                raise
            self._sequence = sequence
            return record
        finally:
            self._release_lock(lock_owner)


@dataclass(frozen=True)
class CheckpointFingerprint:
    """Inputs whose change makes an old checkpoint unsafe to resume."""

    dataset_revision: str
    config_sha256: str
    code_sha256: str
    input_manifest_sha256: str

    def __post_init__(self) -> None:
        _required_text(self.dataset_revision, "dataset_revision")
        for name in ("config_sha256", "code_sha256", "input_manifest_sha256"):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")


class CheckpointStore:
    """Atomically persist and validate unit-level qualification progress."""

    def __init__(
        self,
        run_root: Path,
        relative_path: Union[str, Path],
        fingerprint: CheckpointFingerprint,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.path = resolve_under_root(self.run_root, relative_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint

    def new_state(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": asdict(self.fingerprint),
            "status": "running",
            "units": {},
        }

    def load(self) -> Optional[dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint is corrupt: {self.path}") from exc
        self._validate_state(payload)
        if payload["fingerprint"] != asdict(self.fingerprint):
            raise CheckpointError("checkpoint fingerprint mismatch; refusing unsafe resume")
        return payload

    def save(self, state: Mapping[str, Any]) -> None:
        self._validate_state(state)
        if state["fingerprint"] != asdict(self.fingerprint):
            raise CheckpointError("checkpoint fingerprint mismatch; refusing write")
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        serialized = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            # If writing or replacing fails, never leave a temp file that could
            # be mistaken for a resumable checkpoint.
            if temporary.exists():
                temporary.unlink()

    def record_unit(
        self,
        state: MutableMapping[str, Any],
        unit_id: str,
        *,
        status: str,
        output_sha256: Optional[str] = None,
        output_path: Optional[Union[str, Path]] = None,
        error: Optional[str] = None,
    ) -> None:
        self._validate_state(state)
        identifier = _required_text(unit_id, "unit_id")
        if status not in _UNIT_STATUSES:
            raise CheckpointError(f"unsupported unit status: {status}")
        if status == "completed" and not _is_sha256(output_sha256):
            raise CheckpointError("completed unit requires a lowercase output_sha256")
        if status == "completed" and output_path is None:
            raise CheckpointError("completed unit requires a relative output_path")
        record: dict[str, Any] = {"status": status}
        if output_sha256 is not None:
            if not _is_sha256(output_sha256):
                raise CheckpointError("output_sha256 must be a lowercase SHA-256 digest")
            record["output_sha256"] = output_sha256
        if output_path is not None:
            raw_output_path = Path(output_path)
            if raw_output_path.is_absolute():
                raise CheckpointError("output_path must be relative to the run root")
            try:
                resolved_output = resolve_under_root(self.run_root, raw_output_path)
            except ValueError as exc:
                raise CheckpointError("output_path escapes the run root") from exc
            if status == "completed":
                if not resolved_output.is_file():
                    raise CheckpointError("completed unit output_path is not a file")
                if _sha256_file(resolved_output) != output_sha256:
                    raise CheckpointError("completed unit output_sha256 does not match output_path")
            record["output_path"] = raw_output_path.as_posix()
        if error is not None:
            record["error"] = _redact(str(error))
        state["units"][identifier] = record

    def completed_unit_ids(self, state: Mapping[str, Any]) -> set[str]:
        self._validate_state(state)
        completed: set[str] = set()
        for unit_id, record in state["units"].items():
            if record["status"] != "completed":
                continue
            output = resolve_under_root(self.run_root, record["output_path"])
            if output.is_file() and _sha256_file(output) == record["output_sha256"]:
                completed.add(unit_id)
        return completed

    def _validate_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise CheckpointError("checkpoint must be a JSON object")
        if set(state) != {"schema_version", "fingerprint", "status", "units"}:
            raise CheckpointError("checkpoint contains missing or unsupported fields")
        if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointError("checkpoint schema_version is unsupported")
        fingerprint = state.get("fingerprint")
        expected_keys = set(asdict(self.fingerprint))
        if not isinstance(fingerprint, Mapping) or set(fingerprint) != expected_keys:
            raise CheckpointError("checkpoint fingerprint is missing required fields")
        if state.get("status") not in _CHECKPOINT_STATUSES:
            raise CheckpointError("checkpoint status is unsupported")
        units = state.get("units")
        if not isinstance(units, Mapping):
            raise CheckpointError("checkpoint units must be an object")
        for unit_id, record in units.items():
            if not isinstance(unit_id, str) or not unit_id or not isinstance(record, Mapping):
                raise CheckpointError("checkpoint contains an invalid unit record")
            status = record.get("status")
            if status not in _UNIT_STATUSES:
                raise CheckpointError(f"checkpoint contains unsupported unit status: {status}")
            if not set(record).issubset({"status", "output_sha256", "output_path", "error"}):
                raise CheckpointError("checkpoint unit contains unsupported fields")
            if status == "completed" and not _is_sha256(record.get("output_sha256")):
                raise CheckpointError("completed checkpoint unit lacks valid output_sha256")
            if status == "completed" and not isinstance(record.get("output_path"), str):
                raise CheckpointError("completed checkpoint unit lacks output_path")
            if "output_sha256" in record and not _is_sha256(record["output_sha256"]):
                raise CheckpointError("checkpoint unit contains invalid output_sha256")
            if "output_path" in record:
                if not isinstance(record["output_path"], str) or not record["output_path"]:
                    raise CheckpointError("checkpoint unit contains invalid output_path")
                try:
                    resolve_under_root(self.run_root, record["output_path"])
                except ValueError as exc:
                    raise CheckpointError("checkpoint unit output_path escapes the run root") from exc
            if "error" in record:
                if not isinstance(record["error"], str):
                    raise CheckpointError("checkpoint unit error must be a string")
                if _redact(record["error"]) != record["error"]:
                    raise CheckpointError("checkpoint unit error contains sensitive text")


@dataclass
class CheckpointTrigger:
    """Expose flexible time-, progress-, or caller-triggered checkpointing."""

    interval_sec: float = 600.0
    every_completed_units: int = 0
    _last_saved_at: Optional[float] = None
    _last_completed_units: int = 0

    def __post_init__(self) -> None:
        if not _is_finite_number(self.interval_sec) or self.interval_sec < 0:
            raise ValueError("checkpoint trigger thresholds must be finite and non-negative")
        if (
            isinstance(self.every_completed_units, bool)
            or not isinstance(self.every_completed_units, int)
            or self.every_completed_units < 0
        ):
            raise ValueError("checkpoint trigger thresholds cannot be negative")

    def mark_saved(self, *, now: float, completed_units: int) -> None:
        _finite_number(now, "now")
        _nonnegative_integer(completed_units, "completed_units")
        self._last_saved_at = now
        self._last_completed_units = completed_units

    def is_due(self, *, now: float, completed_units: int, force: bool = False) -> bool:
        _finite_number(now, "now")
        _nonnegative_integer(completed_units, "completed_units")
        if force:
            return True
        if self._last_saved_at is None:
            return True
        time_due = self.interval_sec > 0 and now - self._last_saved_at >= self.interval_sec
        count_due = (
            self.every_completed_units > 0
            and completed_units - self._last_completed_units >= self.every_completed_units
        )
        return time_due or count_due


class RunBudget:
    """Decide whether another task fits while preserving clean shutdown time."""

    def __init__(
        self,
        *,
        target_seconds: float = 3600.0,
        shutdown_reserve_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not _is_finite_number(target_seconds) or not _is_finite_number(shutdown_reserve_seconds):
            raise ValueError("run budget values must be finite")
        if target_seconds <= 0 or shutdown_reserve_seconds < 0:
            raise ValueError("run budget must be positive and reserve cannot be negative")
        if shutdown_reserve_seconds >= target_seconds:
            raise ValueError("shutdown reserve must be smaller than the run budget")
        self.target_seconds = target_seconds
        self.shutdown_reserve_seconds = shutdown_reserve_seconds
        self._monotonic = monotonic
        self._started_at = _finite_number(monotonic(), "monotonic clock")
        self.status = "running"
        self.stop_reason: Optional[str] = None

    @property
    def elapsed_sec(self) -> float:
        current = _finite_number(self._monotonic(), "monotonic clock")
        return max(0.0, current - self._started_at)

    def should_start_next(self, estimated_seconds: float) -> bool:
        if not _is_finite_number(estimated_seconds) or estimated_seconds < 0:
            raise ValueError("estimated_seconds must be finite and non-negative")
        if self.status != "running":
            return False
        fits = self.elapsed_sec + estimated_seconds + self.shutdown_reserve_seconds <= self.target_seconds
        if not fits:
            self.request_soft_stop("insufficient_time_for_next_task")
        return fits

    def request_soft_stop(self, reason: str = "requested") -> str:
        self.status = PARTIAL_RESUMABLE
        self.stop_reason = _required_text(reason, "stop_reason")
        return self.status


T = TypeVar("T")


def run_interruptible(operation: Callable[[], T], *, on_interrupt: Callable[[], None]) -> Union[T, str]:
    """Checkpoint on Ctrl+C and convert it to an explicit resumable outcome.

    Exceptions from ``on_interrupt`` intentionally propagate: claiming a safe
    partial result would be incorrect when the emergency checkpoint failed.
    """

    try:
        return operation()
    except KeyboardInterrupt:
        on_interrupt()
        return PARTIAL_RESUMABLE


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonnegative_number(value: object, name: str) -> float:
    if not _is_finite_number(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return float(value)


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _finite_number(value: object, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value

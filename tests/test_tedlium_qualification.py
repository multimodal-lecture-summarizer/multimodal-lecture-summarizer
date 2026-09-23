"""Contract tests for the local TED-LIUM Release 3 D1 pilot."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import wave

import pytest

from experiments.scripts.benchmark_artifacts import (
    RunnerContext,
    read_json_artifact,
    write_json_artifact,
    write_run_manifest,
)
from experiments.scripts.dataset_qualification_runtime import CheckpointError, ExclusiveOperationLock
from experiments.scripts.profile_dataset import main as profile_dataset_main
from experiments.scripts.tedlium_qualification import qualify_tedlium


def _wav_bytes(*, frames: int = 160) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * frames)
    return output.getvalue()


def _rows(split: str, count: int, *, start: int = 0) -> list[dict[str, object]]:
    audio = _wav_bytes()
    return [
        {
            "id": f"S{index:03d}-{index}.00-{index + 1}.00",
            "audio": {"bytes": audio, "path": None},
            "text": f"utterance {index}",
            "speaker_id": f"S{index % 4:03d}",
            "file": f"/upstream/{split}/talk-{index % 3}.sph",
        }
        for index in range(start, start + count)
    ]


def _provenance(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "dataset_id": "tedlium3",
                "hf_dataset": "distil-whisper/tedlium",
                "hf_splits": ["train", "validation", "test"],
                "license": "CC BY-NC-ND 3.0",
            }
        ),
        encoding="utf-8",
    )
    for split, fingerprint in (("validation", "dev-fp"), ("test", "test-fp")):
        directory = root / split
        directory.mkdir()
        (directory / "state.json").write_text(
            json.dumps({"_fingerprint": fingerprint, "_split": split}), encoding="utf-8"
        )
        (directory / "dataset_info.json").write_text(
            json.dumps({"config_name": "release3"}), encoding="utf-8"
        )
        (directory / f"{split}-00000-of-00001.arrow").write_bytes(
            f"fixture-{split}".encode("utf-8")
        )


def _loader(rows_by_split: dict[str, list[dict[str, object]]], calls: list[str]):
    def load(_root: Path, split: str) -> list[dict[str, object]]:
        calls.append(split)
        return rows_by_split[split]

    return load


def test_d1_pilot_is_deterministic_balanced_and_never_loads_train(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    calls: list[str] = []
    rows = {"validation": _rows("dev", 14), "test": _rows("test", 14, start=100)}

    first = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run-a",
        run_id="run-a",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, calls),
    )
    second = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run-b",
        run_id="run-b",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, calls),
    )

    assert calls == ["validation", "test", "validation", "test"]
    assert [unit["unit_id"] for unit in first["files"]] == [
        unit["unit_id"] for unit in second["files"]
    ]
    assert first["selected_files"] == 20
    assert first["split_counts"] == {"final_test": 10, "validation": 10}
    assert {unit["split"] for unit in first["files"]} == {"validation", "final_test"}
    assert {unit["original_split"] for unit in first["files"]} == {"validation", "test"}
    assert first["acquisition_level"] == "D1_local_pilot"
    assert first["scientific_selection_eligible"] is False
    assert first["network_used"] is False


def test_ignore_marker_is_predeclared_and_corrupt_audio_is_measured(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    validation = _rows("dev", 11)
    validation[0]["text"] = "ignore_time_segment_in_scoring"
    validation[1]["audio"] = {"bytes": b"not-audio", "path": None}
    rows = {"validation": validation, "test": _rows("test", 11, start=100)}

    result = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="run",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, []),
    )

    assert result["selected_files"] == 19
    assert result["ignore_policy"]["marker"] == "ignore_time_segment_in_scoring"
    assert result["ignore_policy"]["excluded_count"] == 1
    assert result["decode_attempts"] == 20
    assert result["decode_failures"] == 1
    assert result["decode_success_rate"] == 19 / 20
    assert result["decode_gate_denominator"] == 20
    assert result["decode_gate_passed"] is True
    assert all(unit["text"] != "ignore_time_segment_in_scoring" for unit in result["files"])
    assert all(unit["decoded"] for unit in result["files"])
    expected = hashlib.sha256(_wav_bytes()).hexdigest()
    assert {unit["audio_sha256"] for unit in result["files"]} == {expected}
    assert all(unit["sample_rate_hz"] == 16_000 for unit in result["files"])


def test_resume_reuses_hash_verified_unit_outputs_without_duplicates(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    rows = {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)}
    decode_calls: list[str] = []

    def decoder(audio_bytes: bytes) -> dict[str, object]:
        decode_calls.append(hashlib.sha256(audio_bytes).hexdigest())
        return {"sample_rate_hz": 16_000, "frames": 160, "channels": 1, "duration_sec": 0.01}

    common = {
        "run_root": tmp_path / "run",
        "config_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "split_loader": _loader(rows, []),
        "audio_decoder": decoder,
    }
    first = qualify_tedlium(dataset_root, run_id="run-1", **common)
    second = qualify_tedlium(dataset_root, run_id="run-2", **common)

    assert len(decode_calls) == 20
    assert first["files"] == second["files"]
    assert len({unit["unit_id"] for unit in second["files"]}) == 20
    assert second["resumed_units"] == 20
    assert second["decode_attempts"] == 20
    assert second["decode_attempts_this_run"] == 0
    assert second["processed_audio_bytes"] == first["processed_audio_bytes"]
    assert second["processing_elapsed_sec"] == first["processing_elapsed_sec"]


def test_budget_soft_stop_returns_partial_resumable_and_checkpoint(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    rows = {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)}

    class ThreeUnitBudget:
        status = "running"
        stop_reason = None

        def __init__(self) -> None:
            self.calls = 0

        @property
        def elapsed_sec(self) -> float:
            return float(self.calls)

        def should_start_next(self, _estimated_seconds: float) -> bool:
            self.calls += 1
            # Eleven setup checks for the two one-chunk fixture splits, then
            # permit exactly three decode units.
            if self.calls <= 14:
                return True
            self.status = "partial_resumable"
            self.stop_reason = "test_budget"
            return False

    result = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="run",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, []),
        budget=ThreeUnitBudget(),
    )

    assert result["status"] == "partial_resumable"
    assert result["selected_files"] == 3
    checkpoint = json.loads(
        (tmp_path / "run" / "operations" / "00b_dataset_feasibility" / "checkpoint.json").read_text()
    )
    assert checkpoint["status"] == "partial_resumable"
    assert len(checkpoint["units"]) == 3

    resumed = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="resume",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, []),
    )
    assert resumed["status"] == "completed"
    assert resumed["decode_attempts"] == 20
    assert resumed["decode_attempts_this_run"] == 17
    assert resumed["resumed_units"] == 3
    assert resumed["processed_audio_bytes"] > resumed["processed_audio_bytes_this_run"]
    assert resumed["processing_elapsed_sec"] >= resumed["processing_elapsed_sec_this_run"]


def test_fixed_candidates_are_not_replaced_after_decode_failure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    validation = _rows("dev", 20)
    test = _rows("test", 20, start=100)
    ranked = sorted(
        validation,
        key=lambda row: hashlib.sha256(f"validation:{row['id']}".encode()).hexdigest(),
    )
    ranked[0]["audio"] = {"bytes": b"broken", "path": None}

    result = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="fixed",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader({"validation": validation, "test": test}, []),
    )

    assert result["status"] == "completed"
    assert result["decode_attempts"] == 20
    assert result["decode_failures"] == 1
    assert result["selected_files"] == 19
    assert result["decode_success_rate"] == 0.95


def test_arrow_content_change_invalidates_resume_fingerprint(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    rows = {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)}
    common = {
        "run_root": tmp_path / "run",
        "config_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "split_loader": _loader(rows, []),
    }
    qualify_tedlium(dataset_root, run_id="first", **common)
    arrow = dataset_root / "validation" / "validation-00000-of-00001.arrow"
    original = arrow.read_bytes()
    arrow.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    assert arrow.stat().st_size == len(original)

    with pytest.raises(CheckpointError, match="fingerprint mismatch"):
        qualify_tedlium(dataset_root, run_id="second", **common)


def test_operation_lock_rejects_concurrent_namespace_writer(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    rows = {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)}
    run_root = tmp_path / "run"
    lock_path = run_root / "operations" / "00b_dataset_feasibility" / ".qualification.lock"

    with ExclusiveOperationLock(lock_path):
        with pytest.raises(RuntimeError, match="already locked"):
            qualify_tedlium(
                dataset_root,
                run_root=run_root,
                run_id="concurrent",
                config_sha256="a" * 64,
                code_sha256="b" * 64,
                split_loader=_loader(rows, []),
            )


def test_setup_events_are_durable_and_detailed_before_split_load(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    rows = {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)}

    qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="setup",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(rows, []),
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "operations" / "00b_dataset_feasibility" / "logs" / "setup.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["event"] == "setup_started"
    stages = [event["context"].get("stage") for event in events if event["event"] == "setup_progress"]
    assert stages.count("arrow_hash_started") == 2
    assert stages.count("arrow_hash_progress") == 2
    assert stages.count("split_load_started") == 2
    assert stages.count("split_loaded") == 2
    assert "snapshot_hashed" in stages


def test_exhausted_budget_interrupts_arrow_hash_before_any_split_load(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    arrow = dataset_root / "validation" / "validation-00000-of-00001.arrow"
    arrow.write_bytes(b"a" * (16 * 1024 * 1024 + 1))
    loader_calls: list[str] = []

    class StopDuringArrowHash:
        status = "running"
        stop_reason = None

        def __init__(self) -> None:
            self.calls = 0

        @property
        def elapsed_sec(self) -> float:
            return float(self.calls)

        def should_start_next(self, _estimated_seconds: float) -> bool:
            self.calls += 1
            if self.calls <= 3:
                return True
            self.status = "partial_resumable"
            self.stop_reason = "test_setup_budget"
            return False

    result = qualify_tedlium(
        dataset_root,
        run_root=tmp_path / "run",
        run_id="setup-stop",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        split_loader=_loader(
            {"validation": _rows("dev", 10), "test": _rows("test", 10, start=100)},
            loader_calls,
        ),
        budget=StopDuringArrowHash(),
    )

    assert result["status"] == "partial_resumable"
    assert result["stopped_during_setup"] is True
    assert result["setup_stage"] == "arrow_hash"
    assert result["selected_files"] == 0
    assert loader_calls == []
    assert not (tmp_path / "run" / "operations" / "00b_dataset_feasibility" / "checkpoint.json").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "operations" / "00b_dataset_feasibility" / "logs" / "setup-stop.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert any(event["event"] == "setup_soft_stop" for event in events)


def test_failed_attempt_metrics_remain_historical_on_resume(tmp_path: Path) -> None:
    dataset_root = tmp_path / "tedlium3"
    _provenance(dataset_root)
    validation = _rows("dev", 10)
    validation[0]["audio"] = {"bytes": b"broken", "path": None}
    rows = {"validation": validation, "test": _rows("test", 10, start=100)}
    common = {
        "run_root": tmp_path / "run",
        "config_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "split_loader": _loader(rows, []),
    }

    first = qualify_tedlium(dataset_root, run_id="first", **common)
    decoder_calls: list[bytes] = []

    def now_working_decoder(audio: bytes) -> dict[str, object]:
        decoder_calls.append(audio)
        return {"sample_rate_hz": 16_000, "frames": 160, "channels": 1, "duration_sec": 0.01}

    resumed = qualify_tedlium(
        dataset_root, run_id="resumed", audio_decoder=now_working_decoder, **common
    )

    assert decoder_calls == []
    assert first["decode_attempts"] == resumed["decode_attempts"] == 20
    assert first["decode_failures"] == resumed["decode_failures"] == 1
    assert resumed["decode_attempts_this_run"] == 0
    assert resumed["decode_failures_this_run"] == 0
    assert resumed["processed_audio_bytes"] == first["processed_audio_bytes"]
    assert resumed["processing_elapsed_sec"] == first["processing_elapsed_sec"]


def test_00b_artifact_to_00c_runner_preserves_explicit_and_original_split(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    shared = {
        "profile": "full",
        "run_namespace": "diagnostic",
        "status": "real",
        "artifact_root": artifact_root,
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "dependency_state_sha256": "c" * 64,
    }
    upstream = RunnerContext(
        notebook_id="00b_dataset_feasibility",
        run_id="00000000-0000-4000-8000-000000000001",
        **shared,
    )
    item = {
        "unit_id": "final_test:sample",
        "relative_path": "tedlium3/test/sample.wav",
        "split": "final_test",
        "original_split": "test",
        "bytes": 42,
        "extension": ".wav",
        "sha256": "d" * 64,
        "readable": True,
        "decoded": True,
        "probe_status": "soundfile_decoded",
    }
    inventory = {
        "dataset_id": "tedlium3",
        "source": "local_huggingface_saved_arrow",
        "acquisition_level": "D1_local_pilot",
        "profile": "full",
        "root": "experiments/datasets/tedlium3",
        "total_files": 1,
        "total_bytes": 42,
        "selected_files": 1,
        "readable_files": 1,
        "selection_policy": "integration_fixture",
        "files": [item],
    }
    feasibility = {
        "dataset_id": "tedlium3",
        "acquisition_level": "D1_local_pilot",
        "selected_files": 1,
        "readable_files": 1,
        "readability_rate": 1.0,
        "eligible_for_profiling": True,
        "scientific_selection_eligible": False,
        "eligibility_reason": "integration fixture",
        "network_used": False,
        "qualification_status": "completed",
    }
    write_json_artifact(upstream, "dataset.feasibility_receipts", [feasibility])
    write_json_artifact(upstream, "dataset.acquisition_inventory", [inventory])
    write_run_manifest(upstream)

    downstream_run_id = "00000000-0000-4000-8000-000000000002"
    argv = [
        "--profile", "full",
        "--run-namespace", "diagnostic",
        "--evidence-status", "real",
        "--artifact-root", str(artifact_root),
        "--notebook-id", "00c_dataset_eda_and_freeze",
        "--run-id", downstream_run_id,
        "--source-sha256", "a" * 64,
        "--config-sha256", "b" * 64,
        "--dependency-state-sha256", "c" * 64,
    ]
    assert profile_dataset_main(argv) == 0

    downstream = RunnerContext(
        notebook_id="00c_dataset_eda_and_freeze",
        run_id=downstream_run_id,
        **shared,
    )
    records = read_json_artifact(downstream, "dataset.unit_profile")["records"]
    assert records[0]["split"] == "final_test"
    assert records[0]["original_split"] == "test"

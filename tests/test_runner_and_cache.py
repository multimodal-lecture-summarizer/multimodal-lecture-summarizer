"""Unit tests for experiment runner, budget guardrails, manifest manager, and feature store."""

import json
from pathlib import Path
import numpy as np
import pytest

from benchmarks.core.feature_store import (
    FeatureCache,
    MultimodalFeatureSchema,
    MultimodalProvenance,
    OCRItem,
    TranscriptChunk,
)
from benchmarks.core.judge import QualityAuditRecord, SingleAuthorAuditTool
from benchmarks.core.manifest_manager import FrozenManifestManager, verify_manifest_integrity
from benchmarks.core.runner import (
    ExperimentConfig,
    ResumableExperimentRunner,
    assert_budget,
)


def test_assert_budget_pass():
    cfg = ExperimentConfig(
        variant_id="S1_fixed_chunk",
        rq_category="RQ2_summarization",
        dataset_name="TIB",
        dataset_split="test",
        source_tokens=32000,
        output_tokens=512,
        max_frames=200,
        frame_resolution_px=448,
    )
    # Should not raise
    assert_budget(cfg)


def test_assert_budget_fail_on_token_inflation():
    cfg = {
        "variant_id": "S3_predicted_hierarchy",
        "source_tokens": 64000,  # Violated!
        "output_tokens": 512,
        "max_frames": 200,
        "frame_resolution_px": 448,
    }
    with pytest.raises(AssertionError, match="Budget mismatch"):
        assert_budget(cfg)


def test_feature_cache_roundtrip(tmp_path: Path):
    cache = FeatureCache(tmp_path)
    vid = "test_vid_001"
    rev = "v1"

    schema = MultimodalFeatureSchema(
        video_id=vid,
        transcript_chunks=[
            TranscriptChunk(id=0, start_sec=0.0, end_sec=5.0, text="Hello world"),
            TranscriptChunk(id=1, start_sec=5.0, end_sec=10.0, text="This is a lecture"),
        ],
        acoustic_features=np.zeros((10, 16), dtype=np.float32),
        visual_features=np.zeros((10, 384), dtype=np.float32),  # DINOv2 384-dim
        ocr_items=[
            OCRItem(text="Introduction", confidence=0.95, bbox=[[0, 0], [10, 0], [10, 10], [0, 10]], timestamp_sec=2.0)
        ],
        provenance=MultimodalProvenance(
            dataset_name="YTSeg",
            dataset_revision=rev,
            video_id=vid,
        ),
    )

    # Save
    cache.save_features(rev, schema)
    assert cache.exists(rev, vid)

    # Load
    loaded = cache.load_features(rev, vid)
    assert loaded.video_id == vid
    assert len(loaded.transcript_chunks) == 2
    assert loaded.visual_features.shape == (10, 384)

    # Schema validation
    val = cache.validate_schema_compliance(rev, vid)
    assert val["schema_compliant"] is True
    assert val["visual_dim_compliant"] is True


def test_resumable_runner_execution(tmp_path: Path):
    ckpt_file = tmp_path / "checkpoint.json"
    runner = ResumableExperimentRunner(ckpt_file)

    cfg = ExperimentConfig(
        variant_id="C1_text_only",
        rq_category="RQ1_chaptering",
        dataset_name="YTSeg",
        dataset_split="test",
    )

    items = [
        {"id": "v1", "context_length": 1500},
        {"id": "v2", "context_length": 2500},
        {"id": "v3_fail", "context_length": 500},
    ]

    def dummy_infer(model, item):
        if "fail" in item["id"]:
            raise RuntimeError("Simulated model execution failure")
        return {"boundaries": [10.0, 30.0]}

    results = runner.run_variant(cfg, items, dummy_infer)
    assert len(results) == 3
    assert results[0].status == "ok"
    assert results[1].status == "ok"
    assert results[2].status == "failed"
    assert results[2].error is not None

    # Check resumption
    runner2 = ResumableExperimentRunner(ckpt_file)
    assert len(runner2.results) == 3


def test_manifest_verification(tmp_path: Path):
    manifest_path = Path("benchmarks/manifests/frozen_manifest_v1.json")
    if manifest_path.exists():
        res = verify_manifest_integrity(manifest_path, check_existing_files=False)
        assert res["integrity_healthy"] is True
        assert res["leakage_free"] is True


def test_single_author_audit_tool(tmp_path: Path):
    audit_file = tmp_path / "audit.json"
    tool = SingleAuthorAuditTool(audit_file)

    tool.add_record(QualityAuditRecord(
        item_id="item_1",
        dataset="VISTA",
        source_support=2,
        coverage=2,
        style="summary-like",
        action="keep",
    ))
    tool.add_record(QualityAuditRecord(
        item_id="item_2",
        dataset="VISTA",
        source_support=0,
        coverage=0,
        style="boilerplate",
        action="exclude",
    ))

    stats = tool.summary_statistics()
    assert stats["total"] == 2
    assert stats["keep_count"] == 1
    assert stats["exclude_count"] == 1
    assert tool.get_exclusion_list() == ["item_2"]

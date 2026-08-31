"""Multimodal Feature Store & Cache Protocol.

Implements the frozen feature cache schema per 03-colab-runbook.md §6 & decisions-log.md D-T04:
- transcript.json: ASR text + timestamped chunks (Whisper)
- acoustic.npy: Pitch, energy, silence/pause features
- visual.npy: DINOv2 ViT-S/14 frame embeddings (384-dim, projected or pooled)
- ocr.json: PaddleOCR v3 ch_PP-OCRv4, confidence threshold >= 0.6
- provenance.json: Model revisions, extractor git hashes, precision, timestamps.

Enforces HF weights on local SSD and precomputed feature stores on Drive/Local (D-T09).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptChunk:
    id: int
    start_sec: float
    end_sec: float
    text: str
    tokens: Optional[List[int]] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class OCRItem:
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    timestamp_sec: float = 0.0


@dataclass
class MultimodalProvenance:
    dataset_name: str
    dataset_revision: str
    video_id: str
    extractor_revisions: Dict[str, str] = field(default_factory=lambda: {
        "visual_model": "dinov2_vits14",
        "ocr_model": "ch_PP-OCRv4",
        "ocr_conf_threshold": "0.6",
        "asr_model": "whisper-small",
    })
    created_at: str = ""
    checksum_sha256: str = ""


@dataclass
class MultimodalFeatureSchema:
    video_id: str
    transcript_chunks: List[TranscriptChunk]
    acoustic_features: np.ndarray  # Shape: [T_acoustic, D_acoustic]
    visual_features: np.ndarray    # Shape: [T_frames, 384] (DINOv2 ViT-S/14)
    ocr_items: List[OCRItem]
    provenance: MultimodalProvenance


class FeatureCache:
    """Manages storage, retrieval, and validation of multimodal feature caches."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_feature_dir(self, dataset_revision: str, video_id: str) -> Path:
        """Return standardized path: features/{dataset_revision}/{video_id}/"""
        clean_vid = video_id.replace("/", "_").replace("\\", "_")
        return self.base_dir / dataset_revision / clean_vid

    def exists(self, dataset_revision: str, video_id: str) -> bool:
        """Check if all required feature files exist for this video."""
        d = self.get_feature_dir(dataset_revision, video_id)
        required = ["transcript.json", "acoustic.npy", "visual.npy", "ocr.json", "provenance.json"]
        return d.is_dir() and all((d / f).exists() for f in required)

    def save_features(
        self,
        dataset_revision: str,
        features: MultimodalFeatureSchema,
    ) -> Path:
        """Save all multimodal modalities to the standardized cache."""
        target_dir = self.get_feature_dir(dataset_revision, features.video_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save transcript.json
        transcript_data = [asdict(c) for c in features.transcript_chunks]
        (target_dir / "transcript.json").write_text(
            json.dumps({"video_id": features.video_id, "chunks": transcript_data}, indent=2),
            encoding="utf-8",
        )

        # 2. Save acoustic.npy
        np.save(target_dir / "acoustic.npy", features.acoustic_features)

        # 3. Save visual.npy
        np.save(target_dir / "visual.npy", features.visual_features)

        # 4. Save ocr.json
        ocr_data = [asdict(item) for item in features.ocr_items]
        (target_dir / "ocr.json").write_text(
            json.dumps({"video_id": features.video_id, "items": ocr_data}, indent=2),
            encoding="utf-8",
        )

        # 5. Save provenance.json
        (target_dir / "provenance.json").write_text(
            json.dumps(asdict(features.provenance), indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved multimodal features for {features.video_id} to {target_dir}")
        return target_dir

    def load_features(self, dataset_revision: str, video_id: str) -> MultimodalFeatureSchema:
        """Load features from cache directory."""
        target_dir = self.get_feature_dir(dataset_revision, video_id)
        if not self.exists(dataset_revision, video_id):
            raise FileNotFoundError(f"Feature cache incomplete or missing at {target_dir}")

        # 1. Transcript
        t_data = json.loads((target_dir / "transcript.json").read_text(encoding="utf-8"))
        chunks = [TranscriptChunk(**c) for c in t_data.get("chunks", [])]

        # 2. Acoustic
        acoustic = np.load(target_dir / "acoustic.npy")

        # 3. Visual
        visual = np.load(target_dir / "visual.npy")

        # 4. OCR
        ocr_data = json.loads((target_dir / "ocr.json").read_text(encoding="utf-8"))
        ocr_items = [OCRItem(**item) for item in ocr_data.get("items", [])]

        # 5. Provenance
        prov_data = json.loads((target_dir / "provenance.json").read_text(encoding="utf-8"))
        provenance = MultimodalProvenance(**prov_data)

        return MultimodalFeatureSchema(
            video_id=video_id,
            transcript_chunks=chunks,
            acoustic_features=acoustic,
            visual_features=visual,
            ocr_items=ocr_items,
            provenance=provenance,
        )

    def validate_schema_compliance(self, dataset_revision: str, video_id: str) -> Dict[str, Any]:
        """Validate shapes and types for C5/C6 encoder inputs."""
        feats = self.load_features(dataset_revision, video_id)
        
        # Visual dim check (DINOv2 ViT-S/14 is 384-dim)
        is_visual_valid = feats.visual_features.ndim == 2 and feats.visual_features.shape[1] == 384
        is_acoustic_valid = feats.acoustic_features.ndim == 2
        is_transcript_valid = len(feats.transcript_chunks) > 0
        
        return {
            "video_id": video_id,
            "transcript_chunks_count": len(feats.transcript_chunks),
            "acoustic_shape": list(feats.acoustic_features.shape),
            "visual_shape": list(feats.visual_features.shape),
            "visual_dim_compliant": is_visual_valid,
            "ocr_items_count": len(feats.ocr_items),
            "schema_compliant": is_visual_valid and is_acoustic_valid and is_transcript_valid,
        }

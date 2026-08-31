"""Frozen Manifest Manager and Integrity Verifier.

Manages versioned, immutable dataset manifests (Tier A–E) for the scientific benchmark:
- Generates/loads frozen manifests with SHA-256 hashes per record/file.
- Verifies dataset integrity to prevent data corruption or drift.
- Asserts strict split segregation (train / validation / test) to prevent data leakage.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ManifestItem:
    item_id: str
    dataset: str
    split: str  # 'train' | 'val' | 'test'
    media_url: Optional[str] = None
    local_path: Optional[str] = None
    checksum_sha256: Optional[str] = None
    duration_sec: float = 0.0
    num_transcript_words: int = 0
    num_slides: int = 0
    num_qa_pairs: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FrozenManifest:
    version: str
    created_at: str
    description: str
    tiers: Dict[str, Dict[str, Any]]
    items: List[ManifestItem]


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a local file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class FrozenManifestManager:
    """Manager for versioned frozen manifests."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path)
        self._data: Optional[Dict[str, Any]] = None

    def load(self) -> Dict[str, Any]:
        """Load manifest from JSON file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")
        self._data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._data

    def save(self, manifest_dict: Dict[str, Any]) -> None:
        """Save manifest dict to JSON file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest_dict, indent=2), encoding="utf-8")
        self._data = manifest_dict
        logger.info(f"Manifest saved to {self.manifest_path}")

    def verify_split_leakage(self) -> Dict[str, Any]:
        """Verify zero leakage across splits within every dataset."""
        if self._data is None:
            self.load()

        leakage_report: Dict[str, Any] = {"passed": True, "details": {}}
        items = self._data.get("items", [])

        # Group by dataset
        by_dataset: Dict[str, Dict[str, set]] = {}
        for item in items:
            ds = item.get("dataset", "unknown")
            split = item.get("split", "unknown")
            item_id = item.get("item_id")
            if ds not in by_dataset:
                by_dataset[ds] = {"train": set(), "val": set(), "test": set()}
            if split in by_dataset[ds]:
                by_dataset[ds][split].add(item_id)

        for ds, splits in by_dataset.items():
            train_test_overlap = splits["train"].intersection(splits["test"])
            train_val_overlap = splits["train"].intersection(splits["val"])
            val_test_overlap = splits["val"].intersection(splits["test"])

            has_leakage = bool(train_test_overlap or train_val_overlap or val_test_overlap)
            if has_leakage:
                leakage_report["passed"] = False

            leakage_report["details"][ds] = {
                "train_count": len(splits["train"]),
                "val_count": len(splits["val"]),
                "test_count": len(splits["test"]),
                "train_test_overlap": list(train_test_overlap),
                "train_val_overlap": list(train_val_overlap),
                "val_test_overlap": list(val_test_overlap),
                "leakage_free": not has_leakage,
            }

        return leakage_report


def verify_manifest_integrity(manifest_path: Path, check_existing_files: bool = False) -> Dict[str, Any]:
    """Verify manifest integrity per 03-colab-runbook.md §4.

    Args:
        manifest_path: Path to the JSON manifest.
        check_existing_files: If True, checks that local_path exists and matches checksum.

    Returns:
        Verification summary dict.
    """
    manager = FrozenManifestManager(manifest_path)
    data = manager.load()
    leakage = manager.verify_split_leakage()

    missing_files: List[str] = []
    checksum_mismatches: List[str] = []

    if check_existing_files:
        for item in data.get("items", []):
            loc = item.get("local_path")
            expected_hash = item.get("checksum_sha256")
            if loc:
                p = Path(loc)
                if not p.exists():
                    missing_files.append(item["item_id"])
                elif expected_hash:
                    actual_hash = compute_file_sha256(p)
                    if actual_hash != expected_hash:
                        checksum_mismatches.append(item["item_id"])

    return {
        "manifest_version": data.get("version", "unknown"),
        "total_items": len(data.get("items", [])),
        "leakage_free": leakage["passed"],
        "leakage_details": leakage["details"],
        "missing_files_count": len(missing_files),
        "missing_files": missing_files,
        "checksum_mismatch_count": len(checksum_mismatches),
        "checksum_mismatches": checksum_mismatches,
        "integrity_healthy": leakage["passed"] and len(checksum_mismatches) == 0,
    }

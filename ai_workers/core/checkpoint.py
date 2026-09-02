"""Checkpoint and intermediate persistence manager for Celery workers.

Saves and loads pipeline intermediate states atomically, allowing the pipeline
to resume execution without re-running heavy ASR, Visual, Fusion, or Quality stages.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ai_workers.core.checkpoint")

PIPELINE_VERSION = "2.0.0"
STAGE_QUALITY_COMPLETE = "quality_complete"
STATUS_INTERMEDIATE_READY = "intermediate_ready"
STATUS_SUMMARIZING = "summarizing"
DEFAULT_CHECKPOINT_DIR = "./storage/checkpoints"


class CheckpointManager:
    """Manages atomic saving, loading, and validation of pipeline checkpoints."""

    def __init__(self, checkpoint_dir: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir or DEFAULT_CHECKPOINT_DIR).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, job_id: str) -> Path:
        sanitized_job_id = "".join(c for c in str(job_id) if c.isalnum() or c in ("-", "_"))
        return self.checkpoint_dir / f"{sanitized_job_id}.json"

    def is_checkpoint_valid(
        self,
        checkpoint_dict: Any,
        expected_stage: str = STAGE_QUALITY_COMPLETE,
        expected_version: str = PIPELINE_VERSION,
    ) -> bool:
        """Validate structural integrity, schema, stage, and pipeline version of a checkpoint."""
        if not isinstance(checkpoint_dict, dict):
            return False

        required_keys = ["job_id", "pipeline_version", "stage", "status", "data"]
        for k in required_keys:
            if k not in checkpoint_dict:
                logger.warning(f"[Checkpoint] Missing required key '{k}' in checkpoint payload.")
                return False

        if checkpoint_dict.get("pipeline_version") != expected_version:
            logger.warning(
                f"[Checkpoint] Pipeline version mismatch: expected {expected_version}, "
                f"got {checkpoint_dict.get('pipeline_version')}"
            )
            return False

        if checkpoint_dict.get("stage") != expected_stage:
            logger.warning(
                f"[Checkpoint] Stage mismatch: expected {expected_stage}, "
                f"got {checkpoint_dict.get('stage')}"
            )
            return False

        data = checkpoint_dict.get("data")
        if not isinstance(data, dict):
            return False

        # Validate minimum necessary data fields for resume
        data_keys = ["utterances", "chapters", "keyframes"]
        for dk in data_keys:
            if dk not in data or not isinstance(data[dk], list):
                logger.warning(f"[Checkpoint] Missing or invalid data key '{dk}' in checkpoint.")
                return False

        return True

    def save_checkpoint(
        self,
        job_id: str,
        stage: str = STAGE_QUALITY_COMPLETE,
        status: str = STATUS_INTERMEDIATE_READY,
        data: Optional[dict[str, Any]] = None,
        video_id: Optional[str] = None,
    ) -> bool:
        """Atomically persist checkpoint to disk and verify file integrity before returning.

        Args:
            job_id: Unique pipeline/celery job identifier.
            stage: Pipeline stage identifier (default: quality_complete).
            status: Job state (default: intermediate_ready).
            data: Intermediate state dictionary (utterances, chapters, keyframes, etc.).
            video_id: Optional video identifier.

        Returns:
            True if checkpoint was saved and verified successfully, False otherwise.
        """
        if not job_id:
            logger.error("[Checkpoint] Cannot save checkpoint without a valid job_id.")
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        filepath = self._get_checkpoint_path(job_id)

        # Retain original created_at if checkpoint already existed
        created_at = now_iso
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict) and "created_at" in existing:
                        created_at = existing["created_at"]
            except Exception:
                pass

        payload = {
            "job_id": str(job_id),
            "video_id": str(video_id) if video_id else str(job_id),
            "pipeline_version": PIPELINE_VERSION,
            "stage": stage,
            "status": status,
            "created_at": created_at,
            "updated_at": now_iso,
            "data": data or {},
        }

        # Atomic write: write to temp file in the same directory, then rename
        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                prefix=f"{filepath.name}.", suffix=".tmp", dir=str(self.checkpoint_dir)
            )
            tmp_path = Path(tmp_path_str)

            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
            tmp_fd = None  # os.fdopen took ownership and closed tmp_fd

            # Atomic replace
            os.replace(tmp_path, filepath)

            # Verification: Read back and validate
            with open(filepath, "r", encoding="utf-8") as verify_f:
                verified_data = json.load(verify_f)

            if not self.is_checkpoint_valid(verified_data, expected_stage=stage):
                logger.error(f"[Checkpoint] Checkpoint verification failed for job {job_id}.")
                return False

            logger.info(
                f"[Checkpoint] Successfully persisted and verified checkpoint for job {job_id} "
                f"(stage={stage}, status={status}) at {filepath}"
            )
            return True

        except Exception as e:
            logger.error(f"[Checkpoint] Failed to save checkpoint for job {job_id}: {e}", exc_info=True)
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except Exception:
                    pass
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return False

    def load_checkpoint(
        self,
        job_id: str,
        expected_stage: str = STAGE_QUALITY_COMPLETE,
        expected_version: str = PIPELINE_VERSION,
    ) -> Optional[dict[str, Any]]:
        """Load and validate an existing checkpoint.

        Returns:
            Checkpoint dictionary if valid, None if not found or invalid.
        """
        filepath = self._get_checkpoint_path(job_id)
        if not filepath.exists():
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if self.is_checkpoint_valid(data, expected_stage=expected_stage, expected_version=expected_version):
                logger.info(
                    f"[Checkpoint] Loaded valid checkpoint for job {job_id} "
                    f"(stage={data.get('stage')}, pipeline_version={data.get('pipeline_version')})"
                )
                return data
            else:
                logger.warning(f"[Checkpoint] Checkpoint for job {job_id} is invalid or version mismatched.")
                return None

        except Exception as e:
            logger.warning(f"[Checkpoint] Failed to read/parse checkpoint for job {job_id}: {e}")
            return None

    def delete_checkpoint(self, job_id: str) -> bool:
        """Delete checkpoint file for given job_id."""
        filepath = self._get_checkpoint_path(job_id)
        if filepath.exists():
            try:
                filepath.unlink()
                logger.info(f"[Checkpoint] Deleted checkpoint for job {job_id}.")
                return True
            except Exception as e:
                logger.error(f"[Checkpoint] Failed to delete checkpoint for job {job_id}: {e}")
                return False
        return False

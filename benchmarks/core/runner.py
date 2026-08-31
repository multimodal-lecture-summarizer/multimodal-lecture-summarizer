"""Resumable Experiment Runner and Budget Enforcement.

Implements:
- Strict Equal-Token-Budget & Equal-Frame-Budget Assertion (03-colab-runbook.md §8, D-T08).
- Memory-safe model execution lifecycle with peak VRAM tracking (03-colab-runbook.md §7).
- Resumable checkpointing per item: OOM and timeouts remain explicit result rows in the denominator (03-colab-runbook.md §13).
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Standard experiment budget constants per 03-colab-runbook.md §8
STANDARD_BUDGET = {
    "source_tokens": 32000,
    "output_tokens": 512,
    "max_frames": 200,
    "frame_resolution_px": 448,
}


@dataclass
class ExperimentConfig:
    variant_id: str  # e.g., 'C1_text_only', 'C5_learned_fusion', 'S1_fixed_chunk', 'S3_predicted_hierarchy'
    rq_category: str  # 'RQ1_chaptering', 'RQ2_summarization', 'RQ3_retrieval_qa', 'RQ4_efficiency'
    dataset_name: str
    dataset_split: str
    seed: int = 42
    source_tokens: int = 32000
    output_tokens: int = 512
    max_frames: int = 200
    frame_resolution_px: int = 448
    timeout_sec: float = 120.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


def assert_budget(variant_config: Dict[str, Any] | ExperimentConfig) -> None:
    """Assert equal token/frame budgets before any S0–S4 or Q0–Q3 run per 03-colab-runbook.md §8.

    Raises:
        AssertionError: If any budget parameter does not match the standard frozen budget.
    """
    cfg = asdict(variant_config) if isinstance(variant_config, ExperimentConfig) else variant_config
    variant_id = cfg.get("variant_id", "unknown_variant")

    for k, v in STANDARD_BUDGET.items():
        actual_v = cfg.get(k)
        if actual_v is not None and actual_v != v:
            raise AssertionError(
                f"Budget mismatch for {variant_id}: {k} = {actual_v}, expected strictly {v}. "
                "Per decisions-log.md D-T08, hierarchical methods must stay within equal token budgets."
            )


@dataclass
class ItemResult:
    item_id: str
    prediction: Optional[Any]
    status: str  # 'ok' | 'oom' | 'timeout' | 'failed'
    error: Optional[str]
    latency_sec: float
    peak_allocated_gb: float
    peak_reserved_gb: float
    context_length: int
    timestamp: str = ""


class ResumableExperimentRunner:
    """Resumable experiment execution manager with memory safety and checkpointing."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[ItemResult] = []
        self._completed_ids: set[str] = set()
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load existing results from checkpoint JSON if present."""
        if self.checkpoint_path.exists():
            try:
                data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                for row in data.get("results", []):
                    res = ItemResult(**row)
                    self.results.append(res)
                    self._completed_ids.add(res.item_id)
                logger.info(f"Resumed {len(self.results)} items from checkpoint: {self.checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint ({e}), starting fresh.")

    def save_checkpoint(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist current results to disk."""
        payload = {
            "checkpoint_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_items": len(self.results),
            "metadata": metadata or {},
            "results": [asdict(r) for r in self.results],
        }
        self.checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_variant(
        self,
        config: ExperimentConfig,
        items: List[Dict[str, Any]],
        infer_fn: Callable[[Any, Dict[str, Any]], Any],
        load_model_fn: Optional[Callable[[], Any]] = None,
    ) -> List[ItemResult]:
        """Execute model inference with strict memory lifecycle, OOM retention and timeouts."""
        # 1. Validate equal budget
        assert_budget(config)

        # 2. Try importing torch safely
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            torch = None
            has_cuda = False

        # 3. Load model
        model = load_model_fn() if load_model_fn else None

        for idx, item in enumerate(items):
            item_id = str(item.get("id") or item.get("item_id") or f"item_{idx}")
            if item_id in self._completed_ids:
                logger.debug(f"Skipping already completed item: {item_id}")
                continue

            if has_cuda and torch is not None:
                torch.cuda.reset_peak_memory_stats()

            started = time.perf_counter()
            prediction = None
            status = "ok"
            error = None

            try:
                prediction = infer_fn(model, item)
            except Exception as exc:
                exc_type_name = type(exc).__name__
                if "OutOfMemoryError" in exc_type_name or "CUDA out of memory" in str(exc):
                    status = "oom"
                elif isinstance(exc, TimeoutError) or "timeout" in str(exc).lower():
                    status = "timeout"
                else:
                    status = "failed"
                error = repr(exc)
                logger.error(f"Execution error on item {item_id} [{status}]: {error}")

            latency = time.perf_counter() - started
            peak_alloc = (
                torch.cuda.max_memory_allocated() / 1e9 if (has_cuda and torch is not None) else 0.0
            )
            peak_res = (
                torch.cuda.max_memory_reserved() / 1e9 if (has_cuda and torch is not None) else 0.0
            )

            result = ItemResult(
                item_id=item_id,
                prediction=prediction,
                status=status,
                error=error,
                latency_sec=latency,
                peak_allocated_gb=round(peak_alloc, 4),
                peak_reserved_gb=round(peak_res, 4),
                context_length=int(item.get("context_length", 0)),
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )

            self.results.append(result)
            self._completed_ids.add(item_id)
            self.save_checkpoint(metadata={"variant_id": config.variant_id})

            # Intermediate cleanup
            if has_cuda and torch is not None:
                torch.cuda.empty_cache()

        # 4. Final model unload and cleanup
        if model is not None:
            del model
        gc.collect()
        if has_cuda and torch is not None:
            torch.cuda.empty_cache()

        return self.results
